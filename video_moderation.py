#!/usr/bin/env python3
"""
video_moderation.py -- pre-upload video moderation for flashgramfo.

What it does
------------
Sits in front of the upload: the browser sends the video HERE first
instead of straight to Cloudinary. The endpoint

1. runs a fast keyword pre-check on the filename + caption
   (same list the app already uses in the browser),
2. extracts frames with OpenCV, one every second by default
   (never more than 30 frames per video -- see below),
3. scores each frame for 18+/NSFW content and stops at the FIRST
   frame scoring strictly above 70%. The check is the AI Vision API
   (Google Cloud Vision SafeSearch) when AI_VISION_API_KEY is set --
   frames are sent in ONE batched request, up to 16 images at a time.
   Without a key, the local open-source ONNX detector (NudeNet 320n,
   ~12 MB -- no TensorFlow, no GPU, CPU-friendly) does the same job,
4. if every frame is clean, forwards the SAME file to Cloudinary
   with the existing unsigned preset (no API secret needed) and
   returns the secure_url + thumbnail URL,
5. if even one frame is flagged, rejects the upload at once
   (HTTP 400), deletes the temporary file, and nothing is
   forwarded to Cloudinary or saved to Supabase.

So the file that passes the check is the exact file that gets
published -- the frontend never uploads a second, unchecked copy.

Flow
----
  browser  --multipart-->  POST /moderate-upload  --pass-->  Cloudinary
     |                          |  (unsigned preset)          |
     |                          v                             v
     +-- 400 "Video rejected: Contains 18+ ..."      secure_url
     +-- 200 {video_url, thumbnail_url} --> insert post row
            into Supabase (frontend, unchanged)

Install (a small Python host: VPS / Railway / Render / Fly.io --
it does NOT fit well on Vercel serverless: OpenCV + the ONNX model
blow past the function size limit and cold-start budget)
----------------------------------------------------------------------
  pip install -r requirements-moderation.txt
  pip install --no-deps "git+https://github.com/notAI-tech/NudeNet.git"

  # optional: the classic nsfw-detector (Yahoo model, TensorFlow --
  # ~500 MB install, much heavier). Only if you install it yourself:
  pip install nsfw-detector

Run
---
  python video_moderation.py serve --host 0.0.0.0 --port 8000
  python video_moderation.py check myvideo.mp4 --caption "beach day"
  python video_moderation.py --self-test

Endpoints
---------
  POST /moderate-upload   multipart: file=<video>, caption=<text>
      200 safe   -> {"ok": true,  "verdict": "safe",
                     "video_url": ..., "thumbnail_url": ...,
                     "frames_checked": N, "max_score": 0.0, "ms": 123}
      400 unsafe -> {"ok": false, "verdict": "unsafe",
                     "error": "Video rejected: Contains inappropriate
                               content violating community guidelines.",
                     "reason": "model"}
      413 too large, 422 unreadable video, 503 detector unavailable
  GET  /health            {"ok": true, "detector": "nudenet", ...}
  GET  /                  short usage text

Environment
-----------
  CLOUDINARY_CLOUD_NAME    default "pwqyc668"
  CLOUDINARY_UPLOAD_PRESET default "sohel98" (unsigned -- no secret)
  CLOUDINARY_UPLOAD_URL    override the upload URL (tests use a local
                           fake server through this)
  MODERATION_MAX_MB        default 100
  MODERATION_FRAME_INTERVAL_S
                           default 1.0 (seconds between sampled
                           frames; 1-2 s recommended)
  MODERATION_MAX_FRAMES    default 30 (cap on frames per video;
                           long videos are thinned evenly so the
                           whole video stays covered)
  MODERATION_MIN_SCORE     default 0.70 (a frame is flagged only when
                           its score is STRICTLY above this)
  AI_VISION_API_KEY      the AI Vision API key -- from the environment
                           ONLY. Never in the repo, never in chat, never
                           in logs. The provider bills per analysed image;
                           restrict the key in the provider console
                           (API restrictions) before deploying.
  AI_VISION_API_URL      default the Google Cloud Vision
                           images:annotate endpoint; point it at another
                           compatible provider if yours differs
  AI_VISION_TIMEOUT_S    default 30 (seconds per batched request)
  AI_VISION_BATCH_SIZE   default 16 (images per request)
  AI_VISION_FLAG_VIOLENCE
                           set to "1" to also flag LIKELY violence
                           (off by default -- adult/racy only)
  MODERATION_DETECTOR      "auto" (default: AI Vision when a key is set,
                           else the local model) | "vision" | "nudenet"
                           | "nsfw-detector"
  MODERATION_CORS_ORIGINS  comma list, default the deployed app origin
  MODERATION_MODEL_PATH    optional path to a different 320n .onnx file

Safety policy
-------------
Fail CLOSED: if no detector can be loaded, or the video cannot be
read, the upload is refused -- never passed through unchecked.
The keyword gate runs first because it costs nothing and catches
the obvious cases before any decoding happens.
The AI Vision API key is read ONLY from the AI_VISION_API_KEY
environment variable, sent only in request headers to the Vision
endpoint, and never written to a log, a response, or the repo. If
the Vision API itself errors (bad key, quota, network), the upload
is refused with 503 -- it is never passed through on a failed check.

Performance notes
-----------------
* Frames are sampled every 1 s (MODERATION_FRAME_INTERVAL_S), never
  more than 30 per video: a 10-minute upload costs the same as a
  60-second one, and long videos are thinned evenly (first and last
  samples kept) so the tail is never skipped entirely.
* Scoring stops at the FIRST flagged frame -- a bad video fails in
  about a second instead of paying for the rest of the frames.
* AI Vision frames are BATCHED: one HTTP request carries up to 16
  images (each a small JPEG, max 1024 px), so a 30-frame video costs
  at most 2 API calls, not 30. The request runs inside the worker
  thread pool, never on the event loop.
* Frames are scored at 320x320 inside the detector; a frame takes
  ~0.02-0.10 s on plain CPU (measured in this sandbox).
* Inference runs in a worker thread pool (run_in_executor), so the
  HTTP server keeps answering /health while a video is being checked.
* The model is loaded once and kept in memory; the temporary upload
  file is deleted in a finally block -- on pass, on rejection, and
  on any error -- so a refused video leaves nothing on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

KEYWORDS = [
    "18+", "nsfw", "porn", "nudity", "nude",
    "inappropriate", "violence", "gore", "explicit",
]
# NudeNet detector labels that count as 18+ content. Covered body parts
# (BREAST_COVERED, BUTTOCKS_COVERED, ...) and ordinary skin (BELLY,
# ARMPITS, FEET, MALE_BREAST) are NOT flagged.
SENSITIVE_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
}
REJECT_MESSAGE = "Video rejected: Contains inappropriate content violating community guidelines."


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Config:
    """Everything tunable, read from the environment at use time."""

    def __init__(self) -> None:
        self.cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "pwqyc668")
        self.upload_preset = os.environ.get("CLOUDINARY_UPLOAD_PRESET", "sohel98")
        self.upload_url = os.environ.get(
            "CLOUDINARY_UPLOAD_URL",
            f"https://api.cloudinary.com/v1_1/{self.cloud_name}/video/upload",
        )
        self.max_mb = _env_int("MODERATION_MAX_MB", 100)
        self.frame_interval_s = min(
            10.0, max(0.25, _env_float("MODERATION_FRAME_INTERVAL_S", 1.0))
        )
        self.max_frames = max(1, _env_int("MODERATION_MAX_FRAMES", 30))
        self.min_score = _env_float("MODERATION_MIN_SCORE", 0.70)
        self.detector_name = os.environ.get("MODERATION_DETECTOR", "auto").lower()
        self.model_path = os.environ.get("MODERATION_MODEL_PATH") or None
        origins = os.environ.get(
            "MODERATION_CORS_ORIGINS", "https://flashgram-five.vercel.app"
        )
        self.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]


CONFIG = Config()


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class ModerationError(Exception):
    """Base class; `status` is the HTTP code the endpoint should use."""


class DetectorUnavailable(ModerationError):
    status = 503

    def __init__(self, detail: str = "Moderation service unavailable") -> None:
        super().__init__(detail)


class UnreadableVideo(ModerationError):
    status = 422

    def __init__(self, detail: str = "Could not read this video file") -> None:
        super().__init__(detail)


class UploadTooLarge(ModerationError):
    status = 413

    def __init__(self, max_mb: int) -> None:
        super().__init__(f"Video is too large (max {max_mb} MB)")
        self.max_mb = max_mb


class CloudinaryError(ModerationError):
    status = 502

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


# --------------------------------------------------------------------------
# Keyword pre-check
# --------------------------------------------------------------------------

def keyword_hit(filename: str, caption: str) -> str | None:
    """Return the first blocked keyword found in filename/caption, else None."""
    text = f"{filename or ''} {caption or ''}".lower()
    for word in KEYWORDS:
        if word in text:
            return word
    return None


# --------------------------------------------------------------------------
# Frame extraction (OpenCV)
# --------------------------------------------------------------------------

def extract_frames(path: str, max_frames: int = 30, interval_s: float = 1.0):
    """Sample frames at a fixed interval (one every `interval_s`
    seconds), up to `max_frames` per video.

    Long videos are thinned evenly (first and last samples kept) so
    the whole video stays covered without scoring hundreds of frames.
    Returns (frames, duration_s, total_frames) where frames is a list
    of (time_s, bgr_frame). Raises UnreadableVideo if nothing can be read.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise UnreadableVideo()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration = total / fps if fps and total else 0.0
        if duration <= 0:
            # No duration metadata (e.g. a raw stream): read frames in
            # order until the cap is hit.
            frames = []
            while len(frames) < max(1, max_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append((round(len(frames) * interval_s, 2), frame))
            if not frames:
                raise UnreadableVideo()
            return frames, 0.0, total
        # One sample in the middle of each interval slice, so the
        # samples stay evenly spread and a short bad clip cannot hide
        # exactly between two sampled instants.
        times = []
        t = interval_s / 2.0
        while t < duration:
            times.append(round(t, 2))
            t += interval_s
        if not times:
            times = [round(duration / 2.0, 2)]
        if len(times) > max_frames:
            if max_frames <= 1:
                times = times[:1]
            else:
                # Thin evenly, keeping the first and last instant. The
                # step is > 1 here (len(times) > max_frames), so the
                # rounded indices are strictly increasing -- no dupes.
                idx = [
                    round(i * (len(times) - 1) / (max_frames - 1))
                    for i in range(max_frames)
                ]
                times = [times[j] for j in idx]
        frames = []
        for when in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, when * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((when, frame))
        if not frames:
            raise UnreadableVideo()
        return frames, round(duration, 2), total
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

class NudeNetBackend:
    """NudeNet 320n ONNX detector (~12 MB, CPU, no TensorFlow).

    Returns, per frame, the highest score among the sensitive labels.
    """

    name = "nudenet"

    def __init__(self, model_path: str | None = None) -> None:
        from nudenet import NudeDetector

        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        self._detector = NudeDetector(**kwargs)

    def score_frame(self, frame: np.ndarray):
        detections = self._detector.detect(frame) or []
        sensitive = [
            {"label": d["class"], "score": round(float(d["score"]), 3)}
            for d in detections
            if d["class"] in SENSITIVE_LABELS
        ]
        top = max((d["score"] for d in sensitive), default=0.0)
        return top, sensitive


class NsfwDetectorBackend:
    """Classic nsfw-detector (Yahoo model, TensorFlow -- heavy).

    Optional only: import it yourself with `pip install nsfw-detector`.
    Unsafe score = porn + hentai probabilities.
    """

    name = "nsfw-detector"

    def __init__(self) -> None:
        from nsfw_detector import predict

        self._predict = predict

    def score_frame(self, frame: np.ndarray):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            cv2.imwrite(tmp.name, frame)
            result = self._predict.classify(self._predict.load_model(), tmp.name)
            scores = result.get(tmp.name, {}) or {}
            unsafe = float(scores.get("porn", 0) or 0) + float(
                scores.get("hentai", 0) or 0
            )
            return round(unsafe, 3), [
                {"label": k, "score": round(float(v), 3)}
                for k, v in scores.items()
            ]
        finally:
            os.unlink(tmp.name)


class VisionAPIError(ModerationError):
    status = 503

    def __init__(self, detail: str = "AI Vision API unavailable") -> None:
        super().__init__(detail)


# Google Vision SafeSearch likelihood -> unsafe score (0..1).
# Only LIKELY / VERY_LIKELY cross the strict >0.70 gate; POSSIBLE
# (0.5) and below pass.
_VISION_LIKELIHOOD_SCORE = {
    "UNKNOWN": 0.0,
    "VERY_UNLIKELY": 0.05,
    "UNLIKELY": 0.2,
    "POSSIBLE": 0.5,
    "LIKELY": 0.9,
    "VERY_LIKELY": 1.0,
}


class VisionBackend:
    """AI Vision API (Google Cloud Vision SafeSearch by default).

    The API key comes ONLY from the AI_VISION_API_KEY environment
    variable -- never hardcoded, never logged, never returned in any
    response. Frames are sent in ONE batched request (up to
    AI_VISION_BATCH_SIZE images) instead of one call per frame, so a
    long video costs a couple of API calls, not one per second.
    """

    name = "ai-vision"

    def __init__(self, api_key=None, api_url=None, timeout_s=None,
                 batch_size=None, flag_violence=None) -> None:
        import httpx

        self._httpx = httpx
        key = api_key if api_key is not None else os.environ.get("AI_VISION_API_KEY", "")
        if not key:
            raise DetectorUnavailable(
                "AI_VISION_API_KEY is not set -- set it in the environment "
                "(never in the repo or in chat)"
            )
        self.api_key = key
        self.api_url = api_url or os.environ.get(
            "AI_VISION_API_URL",
            "https://vision.googleapis.com/v1/images:annotate",
        )
        self.timeout_s = float(
            os.environ.get("AI_VISION_TIMEOUT_S", timeout_s if timeout_s is not None else 30.0)
        )
        self.batch_size = max(
            1, int(os.environ.get("AI_VISION_BATCH_SIZE", batch_size if batch_size is not None else 16))
        )
        self.flag_violence = (
            flag_violence if flag_violence is not None
            else os.environ.get("AI_VISION_FLAG_VIOLENCE", "0") == "1"
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        import base64

        h, w = frame.shape[:2]
        scale = min(1.0, 1024.0 / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise VisionAPIError("Could not encode a frame for the AI Vision API")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _annotate(self, b64_images):
        payload = {
            "requests": [
                {
                    "image": {"content": img},
                    "features": [{"type": "SAFE_SEARCH_DETECTION"}],
                }
                for img in b64_images
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            # trust_env=False: never inherit stray HTTP(S)_PROXY variables
            # from the host -- the service talks to the API directly.
            with self._httpx.Client(trust_env=False, timeout=self.timeout_s) as client:
                resp = client.post(self.api_url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001 -- network failure
            raise VisionAPIError(f"AI Vision API request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise VisionAPIError(
                f"AI Vision API rejected the request ({resp.status_code})"
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise VisionAPIError("AI Vision API returned invalid JSON") from exc
        responses = data.get("responses") or []
        if len(responses) != len(b64_images):
            raise VisionAPIError("AI Vision API returned a mismatched response")
        return responses

    def _score_safe_search(self, annotation):
        ann = annotation or {}
        fields = ("adult", "racy")
        if self.flag_violence:
            fields = fields + ("violence",)
        best_label, best_score = None, 0.0
        for field in fields:
            likelihood = str(ann.get(field, "UNKNOWN")).upper()
            score = _VISION_LIKELIHOOD_SCORE.get(likelihood, 0.0)
            if score > best_score:
                best_score, best_label = score, f"{field}:{likelihood}"
        detections = []
        if best_label:
            detections.append({"label": best_label, "score": round(best_score, 3)})
        return best_score, detections

    def score_batch(self, frames):
        """Score a chunk of (time_s, frame) pairs in ONE API call.

        Returns a list of (top_score, detections) in the same order.
        """
        b64 = [self._encode_frame(frame) for _, frame in frames]
        responses = self._annotate(b64)
        return [
            self._score_safe_search(resp.get("safeSearchAnnotation") or {})
            for resp in responses
        ]

    def score_frame(self, frame: np.ndarray):
        return self.score_batch([(0.0, frame)])[0]


_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def load_detector(name: str = "auto"):
    """Load (once per mode, cached) the detector for `name`.

    Modes: "vision" (AI Vision API only -- raises when the key is not
    set), "nudenet" / "nsfw-detector" (local models only), "auto"
    (the AI Vision API when AI_VISION_API_KEY is set, otherwise the
    local model -- never an unchecked pass).
    Raises DetectorUnavailable when nothing can be loaded -- the caller
    must refuse the upload instead of passing it through.
    """
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is not None and getattr(_DETECTOR, "_loaded_for", None) == name:
            return _DETECTOR
        if name == "vision":
            try:
                _DETECTOR = VisionBackend()
            except DetectorUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DetectorUnavailable(
                    f"AI Vision backend failed to start: {exc}"
                ) from exc
            _DETECTOR._loaded_for = name
            print("[moderation] detector ready: ai-vision", flush=True)
            return _DETECTOR
        candidates = []
        if name in ("auto", "nudenet"):
            candidates.append(("nudenet", lambda: NudeNetBackend(CONFIG.model_path)))
        if name in ("auto", "nsfw-detector"):
            candidates.append(("nsfw-detector", NsfwDetectorBackend))
        # "auto" prefers the AI Vision API when a key is configured --
        # it is tried first, before the local models.
        if name == "auto" and os.environ.get("AI_VISION_API_KEY"):
            candidates.insert(0, ("ai-vision", VisionBackend))
        last_error = None
        for label, make in candidates:
            try:
                _DETECTOR = make()
                _DETECTOR._loaded_for = name
                print(f"[moderation] detector ready: {label}", flush=True)
                return _DETECTOR
            except Exception as exc:  # noqa: BLE001 -- try the next backend
                last_error = exc
                print(f"[moderation] detector {label} unavailable: {exc}", flush=True)
        raise DetectorUnavailable(
            "No moderation model is installed. "
            "Install it with: pip install --no-deps "
            "'git+https://github.com/notAI-tech/NudeNet.git'"
            + (f" (last error: {last_error})" if last_error else "")
        )


# --------------------------------------------------------------------------
# The moderation decision
# --------------------------------------------------------------------------

def moderate_file(
    path: str,
    *,
    filename: str = "",
    caption: str = "",
    detector=None,
    max_frames: int | None = None,
    min_score: float | None = None,
    interval_s: float | None = None,
) -> dict:
    """Check one video file. Returns a verdict dict; never raises for a
    content decision (only DetectorUnavailable / UnreadableVideo).

    Scoring stops at the FIRST frame above the threshold -- a flagged
    video fails fast instead of paying for the remaining frames."""
    started = time.monotonic()
    max_frames = CONFIG.max_frames if max_frames is None else max_frames
    min_score = CONFIG.min_score if min_score is None else min_score
    interval_s = CONFIG.frame_interval_s if interval_s is None else interval_s

    hit = keyword_hit(filename, caption)
    if hit:
        return {
            "verdict": "unsafe",
            "reason": "keyword",
            "matched": hit,
            "frames_checked": 0,
            "duration_s": 0.0,
            "max_score": 1.0,
            "flagged_at": None,
            "flagged_detections": [],
            "detector": None,
            "ms": int((time.monotonic() - started) * 1000),
        }

    frames, duration, _total = extract_frames(path, max_frames, interval_s)
    if detector is None:
        detector = load_detector(CONFIG.detector_name)

    per_frame = []
    max_score = 0.0

    def _score_all():
        # Batched backends (AI Vision) score a chunk per API call, but
        # the verdict still checks each frame in order -- so a flagged
        # frame in an early chunk means later chunks are never sent.
        if hasattr(detector, "score_batch"):
            chunk_size = getattr(detector, "batch_size", 16) or 16
            for start in range(0, len(frames), chunk_size):
                chunk = frames[start:start + chunk_size]
                for (when, _frame), (top, sensitive) in zip(
                    chunk, detector.score_batch(chunk)
                ):
                    yield when, top, sensitive
        else:
            for when, frame in frames:
                top, sensitive = detector.score_frame(frame)
                yield when, top, sensitive

    for when, top, sensitive in _score_all():
        max_score = max(max_score, top)
        entry = {"t": when, "max_score": round(top, 3), "detections": sensitive}
        per_frame.append(entry)
        # Strict gate: only a score STRICTLY above the threshold flags
        # the video, and the first flagged frame ends the check at once.
        if top > min_score:
            return {
                "verdict": "unsafe",
                "reason": "model",
                "frames_checked": len(per_frame),
                "duration_s": duration,
                "max_score": round(top, 3),
                "flagged_at": when,
                "flagged_detections": sensitive,
                "per_frame": per_frame,
                "detector": getattr(detector, "name", "unknown"),
                "ms": int((time.monotonic() - started) * 1000),
            }

    return {
        "verdict": "safe",
        "reason": None,
        "frames_checked": len(frames),
        "duration_s": duration,
        "max_score": round(max_score, 3),
        "flagged_at": None,
        "flagged_detections": [],
        "per_frame": per_frame,
        "detector": getattr(detector, "name", "unknown"),
        "ms": int((time.monotonic() - started) * 1000),
    }


# --------------------------------------------------------------------------
# Forwarding a clean video to Cloudinary (same unsigned preset as the app)
# --------------------------------------------------------------------------

def _derive_thumbnail_url(secure_url: str, data: dict) -> str:
    eager = data.get("eager") or []
    if eager and eager[0].get("secure_url"):
        return eager[0]["secure_url"]
    if secure_url.endswith(".mp4"):
        return secure_url[:-4] + ".jpg"
    if ".mp4" in secure_url:
        return secure_url.replace(".mp4", ".jpg", 1)
    return secure_url


def forward_to_cloudinary(path: str, *, filename: str = "") -> tuple[str, str]:
    """Upload the already-checked file to Cloudinary (unsigned preset).

    Returns (secure_url, thumbnail_url). Raises CloudinaryError on any
    failure so the frontend never saves a half-published post.
    """
    import httpx

    cfg = CONFIG
    with open(path, "rb") as fh:
        files = {"file": (filename or "video.mp4", fh, "video/mp4")}
        data = {"upload_preset": cfg.upload_preset}
        try:
            # trust_env=False: never inherit stray HTTP(S)_PROXY variables
            # from the host -- the service talks to Cloudinary directly.
            with httpx.Client(trust_env=False, timeout=300.0) as client:
                resp = client.post(cfg.upload_url, files=files, data=data)
        except Exception as exc:  # noqa: BLE001 -- network failure
            raise CloudinaryError(f"Cloudinary upload failed: {exc}") from exc
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = (resp.json().get("error") or {}).get("message", "")
        except Exception:  # noqa: BLE001
            detail = resp.text[:200]
        raise CloudinaryError(
            f"Cloudinary rejected the upload ({resp.status_code}): {detail}"
        )
    data = resp.json()
    secure_url = data.get("secure_url") or ""
    if not secure_url.startswith("https://"):
        raise CloudinaryError("Cloudinary returned no secure_url")
    return secure_url, _derive_thumbnail_url(secure_url, data)


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

def create_app():
    cfg = CONFIG
    app = FastAPI(title="flashgramfo video moderation")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return {
            "service": "flashgramfo video moderation",
            "usage": "POST /moderate-upload with multipart file=<video>&caption=<text>",
            "health": "/health",
        }

    @app.get("/health")
    def health():
        loaded = _DETECTOR is not None
        return {
            "ok": True,
            "detector": getattr(_DETECTOR, "name", None) if loaded else None,
            "detector_loaded": loaded,
            "detector_configured": cfg.detector_name,
            "max_mb": cfg.max_mb,
            "max_frames": cfg.max_frames,
            "frame_interval_s": cfg.frame_interval_s,
            "min_score": cfg.min_score,
        }

    @app.post("/moderate-upload")
    async def moderate_upload(
        file: UploadFile = File(...), caption: str = Form(""),
    ):
        started = time.monotonic()
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix="mod-", delete=False
        )
        try:
            written = 0
            limit = cfg.max_mb * 1024 * 1024
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise UploadTooLarge(cfg.max_mb)
                tmp.write(chunk)
            tmp.close()

            # Blocking work (OpenCV + ONNX) off the event loop.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: moderate_file(
                    tmp.name, filename=file.filename or "", caption=caption
                ),
            )
            result["ms"] = int((time.monotonic() - started) * 1000)

            if result["verdict"] == "unsafe":
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "verdict": "unsafe",
                        "error": REJECT_MESSAGE,
                        "reason": result["reason"],
                        "frames_checked": result["frames_checked"],
                        "max_score": result["max_score"],
                        "detector": result["detector"],
                        "ms": result["ms"],
                    },
                )

            video_url, thumbnail_url = await loop.run_in_executor(
                None,
                lambda: forward_to_cloudinary(
                    tmp.name, filename=file.filename or "video.mp4"
                ),
            )
            return {
                "ok": True,
                "verdict": "safe",
                "video_url": video_url,
                "thumbnail_url": thumbnail_url,
                "frames_checked": result["frames_checked"],
                "duration_s": result["duration_s"],
                "max_score": result["max_score"],
                "detector": result["detector"],
                "ms": int((time.monotonic() - started) * 1000),
            }
        except ModerationError as exc:
            status = getattr(exc, "status", 500)
            return JSONResponse(
                status_code=status,
                content={
                    "ok": False,
                    "verdict": "error",
                    "error": str(exc),
                    "ms": int((time.monotonic() - started) * 1000),
                },
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return app


# --------------------------------------------------------------------------
# Self-test (no real Cloudinary, no real NSFW media needed)
# --------------------------------------------------------------------------

class _FakeCloudinary(BaseHTTPRequestHandler):
    received = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        _FakeCloudinary.received = {
            "path": self.path,
            "bytes": len(body),
            "had_file": b"filename=" in body,
            "had_preset": b"sohel98" in body,
        }
        payload = json.dumps(
            {"secure_url": "https://res.cloudinary.com/pwqyc668/video/upload/"
                           "v1/fake_test.mp4"}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


class _FakeVision(BaseHTTPRequestHandler):
    # mode: "clean" | "flag_second" | "possible" | "likely" | "error"
    mode = "clean"
    requests_seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        _FakeVision.requests_seen.append({
            "headers": dict(self.headers),
            "count": len(data.get("requests", [])),
            "images": [r.get("image", {}).get("content", "")
                       for r in data.get("requests", [])],
            "features": [r.get("features")
                         for r in data.get("requests", [])],
        })
        if self.mode == "error":
            payload = json.dumps({"error": {"message": "boom"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        def _ann(level):
            return {"adult": level, "racy": level, "violence": "VERY_UNLIKELY",
                    "spoof": "VERY_UNLIKELY", "medical": "VERY_UNLIKELY"}

        n = len(data.get("requests", []))
        anns = []
        for i in range(n):
            if self.mode == "flag_second" and i == 1:
                anns.append(_ann("VERY_LIKELY"))
            elif self.mode == "possible":
                anns.append(_ann("POSSIBLE"))
            elif self.mode == "likely":
                anns.append(_ann("LIKELY"))
            else:
                anns.append(_ann("VERY_UNLIKELY"))
        payload = json.dumps(
            {"responses": [{"safeSearchAnnotation": a} for a in anns]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


def _make_test_clip(path: str, seconds: int = 4) -> str:
    """A small real H.264 clip (test pattern) via ffmpeg, or reuse it."""
    import shutil
    import subprocess

    if os.path.exists(path):
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found and no test clip present")
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size=320x240:rate=10:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
        ],
        check=True,
    )
    return path


def self_test() -> int:
    """Run the checks. Returns 0 when everything passes."""
    failures = []

    def check(cond: bool, label: str) -> None:
        print(("ok:   " if cond else "FAIL: ") + label)
        if not cond:
            failures.append(label)

    print("[self-test] 1. keyword gate (no model call needed)")
    clip = _make_test_clip("/tmp/moderation_selftest.mp4")
    r = moderate_file(clip, filename="my porn video.mp4", caption="")
    check(r["verdict"] == "unsafe" and r["reason"] == "keyword",
          "filename keyword rejects before any decoding")
    check(r["frames_checked"] == 0, "keyword reject checks zero frames")

    print("[self-test] 2. injected high-score detector -> unsafe")
    class _FakeDetector:
        name = "fake"

        def score_frame(self, frame):
            return 0.95, [{"label": "FEMALE_GENITALIA_EXPOSED", "score": 0.95}]

    r = moderate_file(clip, filename="ok.mp4", caption="", detector=_FakeDetector())
    check(r["verdict"] == "unsafe" and r["reason"] == "model",
          "flagged frame rejects the upload")
    check(r["max_score"] == 0.95, "max score is reported")
    check(r["frames_checked"] == 1, "scoring stops at the first flagged frame")
    check(r["flagged_at"] == 0.5, "flagged frame timestamp reported (t=0.5 s)")

    print("[self-test] 2b. strict >70% boundary")
    class _BoundaryDetector:
        name = "boundary"

        def __init__(self, score):
            self._score = score

        def score_frame(self, frame):
            return self._score, [
                {"label": "FEMALE_BREAST_EXPOSED", "score": self._score}
            ]

    r = moderate_file(clip, filename="ok.mp4", caption="",
                      detector=_BoundaryDetector(0.70))
    check(r["verdict"] == "safe", "exactly 70% does NOT flag (strictly above only)")
    r = moderate_file(clip, filename="ok.mp4", caption="",
                      detector=_BoundaryDetector(0.71))
    check(r["verdict"] == "unsafe" and r["frames_checked"] == 1,
          "71% flags at once")

    print("[self-test] 3. unreadable file -> UnreadableVideo (fail closed)")
    junk = "/tmp/moderation_selftest.txt"
    Path(junk).write_text("not a video", encoding="utf-8")
    try:
        moderate_file(junk, filename="junk.mp4")
        check(False, "unreadable video must raise")
    except UnreadableVideo:
        check(True, "unreadable video raises UnreadableVideo")

    print("[self-test] 4. real detector on a real safe clip")
    try:
        t0 = time.monotonic()
        r = moderate_file(clip, filename="beach.mp4", caption="beach day")
        dt = time.monotonic() - t0
        check(r["verdict"] == "safe", f"safe clip passes ({r})")
        check(r["frames_checked"] == 4,
              f"4 s clip samples every 1 s: {r['frames_checked']} frames")
        check(r["detector"] == "nudenet", "detector is nudenet")
        print(f"      real check took {dt:.2f}s for {r['frames_checked']} frames")
    except DetectorUnavailable as exc:
        check(False, f"real detector must load in this environment: {exc}")

    print("[self-test] 4b. interval sampling + thinning")
    long_clip = _make_test_clip("/tmp/moderation_selftest_long.mp4", seconds=12)
    frames12, dur12, _ = extract_frames(long_clip, 30, 1.0)
    check(len(frames12) == 12,
          f"12 s clip samples 12 frames at 1 s ({len(frames12)})")
    check(abs(frames12[0][0] - 0.5) < 0.01 and abs(frames12[-1][0] - 11.5) < 0.01,
          "samples are the middles of 1 s slices")
    thin, _, _ = extract_frames(clip, 3, 1.0)
    check([t for t, _ in thin] == [0.5, 2.5, 3.5],
          f"long video thins evenly, first+last kept ({[t for t, _ in thin]})")

    print("[self-test] 5. forward to a fake Cloudinary (no real upload)")
    # The fake server stays up through step 6: the HTTP end-to-end test
    # forwards through it too. (Shutting it down between the two steps
    # leaves a listening-but-dead socket that makes the next upload wait
    # out the full 300 s timeout instead of failing fast.)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCloudinary)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ["CLOUDINARY_UPLOAD_URL"] = f"http://127.0.0.1:{port}/video/upload"
    CONFIG.upload_url = os.environ["CLOUDINARY_UPLOAD_URL"]
    try:
        url, thumb = forward_to_cloudinary(clip, filename="beach.mp4")
        check(url == "https://res.cloudinary.com/pwqyc668/video/upload/v1/fake_test.mp4",
              "forward returns the secure_url")
        check(thumb == "https://res.cloudinary.com/pwqyc668/video/upload/v1/fake_test.jpg",
              "thumbnail derived as .mp4 -> .jpg")
        rec = _FakeCloudinary.received
        check(rec.get("had_file") and rec.get("had_preset"),
              "forward sent the file + the unsigned preset")
    finally:
        pass  # server stays up for step 6; closed at the end

    print("[self-test] 6. HTTP end-to-end (FastAPI TestClient)")
    try:
        from fastapi.testclient import TestClient

        # Spy on temp-file deletion: every upload (pass or reject) must
        # leave nothing on disk afterwards.
        deleted_tmps = []
        _real_unlink = os.unlink

        def _spy_unlink(p):
            deleted_tmps.append(str(p))
            return _real_unlink(p)

        os.unlink = _spy_unlink
        app = create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            check(health.status_code == 200, "GET /health works")
            check(health.json().get("min_score") == 0.7,
                  "health reports the 0.70 threshold")
            check(health.json().get("frame_interval_s") == 1.0,
                  "health reports the 1 s sampling interval")
            with open(clip, "rb") as fh:
                resp = client.post(
                    "/moderate-upload",
                    files={"file": ("beach.mp4", fh, "video/mp4")},
                    data={"caption": "beach day"},
                )
            check(resp.status_code == 200, f"safe upload -> 200 ({resp.status_code})")
            body = resp.json()
            check(body.get("ok") is True and body.get("video_url", "").startswith("https://"),
                  "safe upload returns a video_url")
            with open(clip, "rb") as fh:
                bad = client.post(
                    "/moderate-upload",
                    files={"file": ("nsfw_clip.mp4", fh, "video/mp4")},
                    data={"caption": ""},
                )
            check(bad.status_code == 400, f"keyword upload -> 400 ({bad.status_code})")
            check(bad.json().get("error") == REJECT_MESSAGE,
                  "rejection carries the exact message")
        mp4_deleted = [p for p in deleted_tmps if p.endswith(".mp4")]
        check(len(mp4_deleted) >= 2,
              f"temp upload files deleted after both requests ({len(mp4_deleted)})")
        check(all(not os.path.exists(p) for p in mp4_deleted),
              "deleted temp files are really gone")
    except ImportError as exc:
        print(f"      skipped (fastapi/httpx not installed): {exc}")
    finally:
        try:
            os.unlink = _real_unlink
        except NameError:
            pass
        server.shutdown()
        server.server_close()

    print("[self-test] 7. AI Vision API backend (fake Vision server)")
    vision_server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVision)
    vport = vision_server.server_address[1]
    vthread = threading.Thread(target=vision_server.serve_forever, daemon=True)
    vthread.start()
    _old_key = os.environ.get("AI_VISION_API_KEY")
    os.environ["AI_VISION_API_KEY"] = "test-key-123"
    try:
        vurl = f"http://127.0.0.1:{vport}/v1/images:annotate"
        _FakeVision.requests_seen.clear()
        _FakeVision.mode = "clean"
        backend = VisionBackend(api_url=vurl)
        r = moderate_file(clip, filename="beach.mp4", caption="",
                          detector=backend)
        check(r["verdict"] == "safe" and r["detector"] == "ai-vision",
              "Vision: all-clean frames pass")
        check(len(_FakeVision.requests_seen) == 1,
              f"4 frames sent in ONE batched request ({len(_FakeVision.requests_seen)})")
        req = _FakeVision.requests_seen[0]
        check(req["count"] == 4, "batch carries all 4 frames")
        check(all(f == [{"type": "SAFE_SEARCH_DETECTION"}] for f in req["features"]),
              "each image asks for SafeSearch")
        import base64
        check(all(base64.b64decode(c)[:2] == b"\xff\xd8" for c in req["images"]),
              "frames sent as JPEG")
        check(req["headers"].get("X-Goog-Api-Key") == "test-key-123",
              "API key sent in the header (never in the repo)")
        check(req["headers"].get("Authorization") == "Bearer test-key-123",
              "API key also sent as Bearer")
        _FakeVision.requests_seen.clear()
        _FakeVision.mode = "flag_second"
        r = moderate_file(clip, filename="beach.mp4", caption="",
                          detector=backend)
        check(r["verdict"] == "unsafe" and r["flagged_at"] == 1.5,
              "Vision: flagged frame stops the check")
        check(r["frames_checked"] == 2,
              "only the frames up to the flagged one counted")
        _FakeVision.mode = "possible"
        r = moderate_file(clip, filename="beach.mp4", caption="",
                          detector=backend)
        check(r["verdict"] == "safe", "POSSIBLE (0.5) does not flag")
        _FakeVision.mode = "likely"
        r = moderate_file(clip, filename="beach.mp4", caption="",
                          detector=backend)
        check(r["verdict"] == "unsafe", "LIKELY (0.9) flags")
        long20 = _make_test_clip("/tmp/moderation_selftest_20s.mp4", seconds=20)
        _FakeVision.requests_seen.clear()
        _FakeVision.mode = "clean"
        r = moderate_file(long20, filename="long.mp4", caption="",
                          detector=backend)
        check(r["frames_checked"] == 20,
              f"20 s clip checks 20 frames ({r['frames_checked']})")
        check(len(_FakeVision.requests_seen) == 2,
              f"20 frames chunked into 2 requests ({len(_FakeVision.requests_seen)})")
        check([x["count"] for x in _FakeVision.requests_seen] == [16, 4],
              "chunks are 16 + 4")
        _FakeVision.mode = "error"
        try:
            moderate_file(clip, filename="beach.mp4", caption="",
                          detector=backend)
            check(False, "Vision API error must raise")
        except VisionAPIError as exc:
            check(getattr(exc, "status", None) == 503,
                  "Vision API error fails closed with 503")
    finally:
        if _old_key is None:
            os.environ.pop("AI_VISION_API_KEY", None)
        else:
            os.environ["AI_VISION_API_KEY"] = _old_key
        vision_server.shutdown()
        vision_server.server_close()
    os.environ.pop("AI_VISION_API_KEY", None)
    try:
        load_detector("vision")
        check(False, "vision mode without a key must raise")
    except DetectorUnavailable:
        check(True, "vision mode without AI_VISION_API_KEY fails closed")

    print()
    if failures:
        print(f"[self-test] FAILED ({len(failures)}): {failures}")
        return 1
    print("[self-test] ALL PASSED")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="flashgramfo video moderation")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in checks and exit")
    sub = parser.add_subparsers(dest="command")

    check_p = sub.add_parser("check", help="moderate one file and print JSON")
    check_p.add_argument("video", help="path to the video file")
    check_p.add_argument("--caption", default="", help="caption / description")
    check_p.add_argument("--no-forward", action="store_true",
                         help="only moderate, do not upload to Cloudinary")
    check_p.add_argument("--json", action="store_true",
                         help="print only the JSON verdict")

    serve_p = sub.add_parser("serve", help="run the FastAPI endpoint")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    if args.command == "check":
        try:
            result = moderate_file(
                args.video,
                filename=os.path.basename(args.video),
                caption=args.caption,
            )
        except ModerationError as exc:
            print(json.dumps({"ok": False, "verdict": "error",
                              "error": str(exc)}, indent=2))
            return 1
        if result["verdict"] == "unsafe":
            print(json.dumps({"ok": False, **result,
                              "error": REJECT_MESSAGE}, indent=2))
            return 2
        if args.no_forward:
            print(json.dumps({"ok": True, **result}, indent=2))
            return 0
        try:
            video_url, thumb = forward_to_cloudinary(
                args.video, filename=os.path.basename(args.video)
            )
        except ModerationError as exc:
            print(json.dumps({"ok": False, "verdict": "error",
                              "error": str(exc)}, indent=2))
            return 1
        print(json.dumps({"ok": True, **result, "video_url": video_url,
                          "thumbnail_url": thumb}, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
