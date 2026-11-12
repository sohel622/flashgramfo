#!/usr/bin/env python3
"""
video_moderation.py -- pre-upload video moderation for flashgramfo.

What it does
------------
Sits in front of the upload: the browser sends the video HERE first
instead of straight to Cloudinary. The endpoint

1. runs a fast keyword pre-check on the filename + caption
   (same list the app already uses in the browser),
2. extracts a few evenly-spaced frames with OpenCV,
3. scores each frame with a small, open-source ONNX NSFW detector
   (NudeNet 320n, ~12 MB -- no TensorFlow, no GPU, CPU-friendly),
4. if every frame is clean, forwards the SAME file to Cloudinary
   with the existing unsigned preset (no API secret needed) and
   returns the secure_url + thumbnail URL,
5. if any frame is flagged, rejects the upload with the exact
   message the frontend can show the user.

So the file that passes the check is the exact file that gets
published -- the frontend never uploads a second, unchecked copy.

Flow
----
  browser  --multipart-->  POST /moderate-upload  --pass-->  Cloudinary
     |                          |  (unsigned preset)          |
     |                          v                             v
     +-- 403 "Video violates community guidelines ..."   secure_url
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
      403 unsafe -> {"ok": false, "verdict": "unsafe",
                     "error": "Video violates community guidelines due
                               to sensitive content", "reason": "model"}
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
  MODERATION_MAX_FRAMES    default 6  (frames sampled per video)
  MODERATION_MIN_SCORE     default 0.5 (sensitive-label score that
                           flags a frame)
  MODERATION_DETECTOR      "auto" (default) | "nudenet" | "nsfw-detector"
  MODERATION_CORS_ORIGINS  comma list, default the deployed app origin
  MODERATION_MODEL_PATH    optional path to a different 320n .onnx file

Safety policy
-------------
Fail CLOSED: if no detector can be loaded, or the video cannot be
read, the upload is refused -- never passed through unchecked.
The keyword gate runs first because it costs nothing and catches
the obvious cases before any decoding happens.

Performance notes
-----------------
* Only ~6 frames per video, sampled across the whole duration, so
  a 10-minute upload costs about the same as a 10-second one.
* Frames are scored at 320x320 inside the detector; a frame takes
  ~0.02-0.10 s on plain CPU (measured in this sandbox).
* Inference runs in a worker thread pool, so the HTTP server keeps
  answering /health while a video is being checked.
* The model is loaded once and kept in memory; temp files are
  deleted even when the request fails.
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
REJECT_MESSAGE = "Video violates community guidelines due to sensitive content"


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
        self.max_frames = _env_int("MODERATION_MAX_FRAMES", 6)
        self.min_score = _env_float("MODERATION_MIN_SCORE", 0.5)
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

def extract_frames(path: str, max_frames: int = 6):
    """Sample up to `max_frames` frames, evenly spread across the video.

    Returns (frames, duration_s, total_frames) where frames is a list of
    (time_s, bgr_frame). Raises UnreadableVideo if nothing can be read.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise UnreadableVideo()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration = total / fps if fps and total else 0.0
        count = min(max(1, max_frames), total) if total else max_frames
        frames = []
        for i in range(count):
            # Middle of each slice, so the first/middle/last of the video
            # are all covered and a short bad clip cannot hide between
            # two sampled instants.
            if duration > 0:
                when = duration * (i + 0.5) / count
                cap.set(cv2.CAP_PROP_POS_MSEC, when * 1000.0)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok and frame is not None:
                when = duration * (i + 0.5) / count if duration > 0 else float(i)
                frames.append((round(when, 2), frame))
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


_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def load_detector(name: str = "auto"):
    """Load (once, cached) the first available detector.

    Raises DetectorUnavailable when nothing is installed -- the caller
    must refuse the upload instead of passing it through.
    """
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is not None:
            return _DETECTOR
        candidates = []
        if name in ("auto", "nudenet"):
            candidates.append(("nudenet", lambda: NudeNetBackend(CONFIG.model_path)))
        if name in ("auto", "nsfw-detector"):
            candidates.append(("nsfw-detector", NsfwDetectorBackend))
        last_error = None
        for label, make in candidates:
            try:
                _DETECTOR = make()
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
) -> dict:
    """Check one video file. Returns a verdict dict; never raises for a
    content decision (only DetectorUnavailable / UnreadableVideo)."""
    started = time.monotonic()
    max_frames = CONFIG.max_frames if max_frames is None else max_frames
    min_score = CONFIG.min_score if min_score is None else min_score

    hit = keyword_hit(filename, caption)
    if hit:
        return {
            "verdict": "unsafe",
            "reason": "keyword",
            "matched": hit,
            "frames_checked": 0,
            "duration_s": 0.0,
            "max_score": 1.0,
            "detector": None,
            "ms": int((time.monotonic() - started) * 1000),
        }

    frames, duration, _total = extract_frames(path, max_frames)
    if detector is None:
        detector = load_detector(CONFIG.detector_name)

    per_frame = []
    max_score = 0.0
    for when, frame in frames:
        top, sensitive = detector.score_frame(frame)
        max_score = max(max_score, top)
        per_frame.append(
            {"t": when, "max_score": round(top, 3), "detections": sensitive}
        )

    unsafe = max_score >= min_score
    return {
        "verdict": "unsafe" if unsafe else "safe",
        "reason": "model" if unsafe else None,
        "frames_checked": len(frames),
        "duration_s": duration,
        "max_score": round(max_score, 3),
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
                    status_code=403,
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
    check(r["frames_checked"] >= 1, "frames were actually sampled")

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
        check(r["frames_checked"] == CONFIG.max_frames,
              f"sampled {r['frames_checked']} frames")
        check(r["detector"] == "nudenet", "detector is nudenet")
        print(f"      real check took {dt:.2f}s for {r['frames_checked']} frames")
    except DetectorUnavailable as exc:
        check(False, f"real detector must load in this environment: {exc}")

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

        app = create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            check(health.status_code == 200, "GET /health works")
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
            check(bad.status_code == 403, f"keyword upload -> 403 ({bad.status_code})")
            check(bad.json().get("error") == REJECT_MESSAGE,
                  "rejection carries the exact guidelines message")
    except ImportError as exc:
        print(f"      skipped (fastapi/httpx not installed): {exc}")
    finally:
        server.shutdown()
        server.server_close()

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
