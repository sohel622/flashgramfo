#!/usr/bin/env python3
"""Flashgram / ReelFlow "For You" recommendation algorithm.

A TikTok/Reels-style feed ranker: the videos a person watches, likes,
follows and skips decide what they see next.

How it works
------------
Every watch/like/follow/save/share/skip is an *event*. Events build a
*taste* profile with two hashtag maps:

* ``tags_watch``    -- implicit signals: what the person actually watched.
  A full watch adds +0.5 to each hashtag, a partial watch +0.2, a fast
  skip subtracts 0.4. This is the strongest signal, like TikTok's
  watch-time signal.
* ``tags_explicit`` -- explicit signals: a like adds +1.0, a save or a
  share +0.7, an unlike -1.0.

Creators get an affinity the same way, plus a ``followed`` set.

Each candidate video is then scored (weights are the single source of
truth -- ``index.html`` mirrors these exact numbers in JavaScript, so the
app and this file always rank the same way):

* watch-tags affinity   30 points  (mean of the video's hashtags)
* explicit-tags affinity 20 points (mean of the video's hashtags)
* creator               25 points  (followed creator = 25, creator of a
  liked video = 15, otherwise watch-affinity scaled 0-15)
* engagement            10 points  (log-scaled likes/comments/shares/views)
* recency                5 points  (half-life of 7 days)
* diversity penalty     -8 for repeating a creator from the last 2 picks,
  -4 when hashtag overlap with the previous pick is over 50%.

Finally, every 10th slot is an *exploration* pick: the best video from a
creator the person has never engaged with, so the feed never becomes an
echo chamber. With no history at all (cold start) the feed ranks by
engagement + recency.

Video dicts
-----------
::

    {
        "id": "seed-1",
        "username": "earth_moments",
        "hashtags": ["#nature", "#bloom"],
        "likes": 14200, "comments": 382, "shares": 890, "views": 64200,
        "ts": 1780000000000,          # epoch milliseconds (like index.html)
        # ... or "created_at": "2026-11-01T04:00:00Z" (ISO string)
    }

Event dicts (the same JSON shape ``index.html`` stores in localStorage
under ``reelflow_feed_events``)::

    {"type": "watch", "video_id": "seed-1", "username": "earth_moments",
     "hashtags": ["nature", "bloom"], "completion": 0.95,
     "watched_seconds": 28.0, "duration_seconds": 30.0, "ts": 1780000000}
    {"type": "like" | "follow" | "unfollow" | "unlike" | "save" | "share"
     | "skip", ...}

Usage
-----
::

    # See it work on built-in demo data:
    python3 recommendation.py --demo

    # Run the built-in checks:
    python3 recommendation.py --self-test

    # Rank real data (videos.json: list of video dicts,
    # events.json: list of event dicts):
    python3 recommendation.py --videos videos.json --events events.json \\
        --limit 10 --current seed-1

    # As a library:
    from recommendation import rank_feed, build_taste
    ranked = rank_feed(videos, events, limit=20)  # [(video, score, breakdown)]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

# --- Weights: keep in sync with FEED_RANK_WEIGHTS in index.html ----------
WATCH_TAGS_POINTS = 30.0
EXPLICIT_TAGS_POINTS = 20.0
CREATOR_POINTS = 25.0
CREATOR_LIKED_POINTS = 15.0
ENGAGEMENT_POINTS = 10.0
RECENCY_POINTS = 5.0
RECENCY_HALF_LIFE_DAYS = 7.0
DIVERSITY_CREATOR_PENALTY = -8.0
DIVERSITY_HASHTAG_PENALTY = -4.0
EXPLORATION_EVERY_N = 10  # every 10th slot is an unseen-creator pick

EVENT_LIMIT = 300  # same cap as the app keeps in localStorage


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm_tag(tag: str) -> str:
    """'#Nature' -> 'nature' (tags compare without '#' and case)."""
    return str(tag or "").strip().lstrip("#").lower()


def video_age_days(video: dict, now: float | None = None) -> float:
    """Age of a video in days. Accepts ts (ms), created_ts (s), created_at."""
    if now is None:
        now = datetime.now(timezone.utc).timestamp()
    ts = video.get("ts")
    if ts is not None:
        created = float(ts) / 1000.0
    elif video.get("created_ts") is not None:
        created = float(video["created_ts"])
    elif video.get("created_at"):
        created = datetime.fromisoformat(
            str(video["created_at"]).replace("Z", "+00:00")
        ).timestamp()
    else:
        return 0.0
    return max(0.0, (now - created) / 86400.0)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mapped_affinity(affinity: float) -> float:
    """Affinity in [-1, 1] -> score factor in [0, 1]. Unknown tags score 0.5."""
    return (affinity + 1.0) / 2.0


# --------------------------------------------------------------------------
# Taste: what the person likes, learned from events
# --------------------------------------------------------------------------
def new_taste() -> dict:
    return {
        "tags_watch": {},
        "tags_explicit": {},
        "creators": {},
        "followed": set(),
        "liked_creators": set(),
        "watched_ids": set(),
    }


def _bump(mapping: dict, key: str, delta: float) -> None:
    if not key:
        return
    mapping[key] = _clamp(mapping.get(key, 0.0) + delta)


def build_taste(events: list[dict]) -> dict:
    """Fold a list of events (oldest first) into a taste profile."""
    taste = new_taste()
    for ev in (events or [])[-EVENT_LIMIT:]:
        etype = str(ev.get("type") or "").lower()
        username = str(ev.get("username") or "").lower()
        tags = [norm_tag(t) for t in (ev.get("hashtags") or [])]
        video_id = ev.get("video_id")
        if video_id:
            taste["watched_ids"].add(str(video_id))

        if etype == "like":
            for t in tags:
                _bump(taste["tags_explicit"], t, 1.0)
            _bump(taste["creators"], username, 1.0)
            if username:
                taste["liked_creators"].add(username)
        elif etype == "unlike":
            for t in tags:
                _bump(taste["tags_explicit"], t, -1.0)
            _bump(taste["creators"], username, -1.0)
            taste["liked_creators"].discard(username)
        elif etype == "follow":
            if username:
                taste["followed"].add(username)
            taste["creators"][username] = 1.0
        elif etype == "unfollow":
            taste["followed"].discard(username)
        elif etype in ("save", "share"):
            for t in tags:
                _bump(taste["tags_explicit"], t, 0.7)
            _bump(taste["creators"], username, 0.5)
        elif etype == "watch":
            completion = float(ev.get("completion") or 0.0)
            if completion >= 0.7:
                for t in tags:
                    _bump(taste["tags_watch"], t, 0.5)
                _bump(taste["creators"], username, 0.3)
            elif completion >= 0.3:
                for t in tags:
                    _bump(taste["tags_watch"], t, 0.2)
            # A <30% watch is neutral: the person sampled it, no signal.
        elif etype == "skip":
            for t in tags:
                _bump(taste["tags_watch"], t, -0.4)
            _bump(taste["creators"], username, -0.2)
    return taste


def taste_to_dict(taste: dict) -> dict:
    """JSON-safe copy of a taste profile (sets become sorted lists)."""
    return {
        "tags_watch": taste["tags_watch"],
        "tags_explicit": taste["tags_explicit"],
        "creators": taste["creators"],
        "followed": sorted(taste["followed"]),
        "liked_creators": sorted(taste["liked_creators"]),
        "watched_ids": sorted(taste["watched_ids"]),
    }


def taste_from_dict(data: dict) -> dict:
    taste = new_taste()
    taste.update(
        {
            "tags_watch": dict(data.get("tags_watch") or {}),
            "tags_explicit": dict(data.get("tags_explicit") or {}),
            "creators": dict(data.get("creators") or {}),
            "followed": set(data.get("followed") or []),
            "liked_creators": set(data.get("liked_creators") or []),
            "watched_ids": set(data.get("watched_ids") or []),
        }
    )
    return taste


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _mean_tag_score(tags: list[str], mapping: dict, points: float) -> float:
    if not tags:
        return points * 0.5  # no hashtags: neutral, not punished
    total = sum(_mapped_affinity(mapping.get(norm_tag(t), 0.0)) for t in tags)
    return points * (total / len(tags))


def score_video(video: dict, taste: dict, now: float | None = None) -> dict:
    """Score one video. Returns {'score': float, 'breakdown': {...}}."""
    tags = [norm_tag(t) for t in (video.get("hashtags") or [])]
    username = str(video.get("username") or "").lower()

    watch = _mean_tag_score(tags, taste["tags_watch"], WATCH_TAGS_POINTS)
    explicit = _mean_tag_score(tags, taste["tags_explicit"], EXPLICIT_TAGS_POINTS)

    if username in taste["followed"]:
        creator = CREATOR_POINTS
    elif username in taste["liked_creators"]:
        creator = CREATOR_LIKED_POINTS
    else:
        creator = CREATOR_LIKED_POINTS * max(0.0, taste["creators"].get(username, 0.0))

    engagement_raw = math.log10(
        1.0
        + float(video.get("likes") or 0)
        + 2.0 * float(video.get("comments") or 0)
        + 3.0 * float(video.get("shares") or 0)
        + float(video.get("views") or 0) / 100.0
    )
    engagement = ENGAGEMENT_POINTS * min(engagement_raw / 6.0, 1.0)

    recency = RECENCY_POINTS * (0.5 ** (video_age_days(video, now) / RECENCY_HALF_LIFE_DAYS))

    breakdown = {
        "watch_tags": round(watch, 2),
        "explicit_tags": round(explicit, 2),
        "creator": round(creator, 2),
        "engagement": round(engagement, 2),
        "recency": round(recency, 2),
        "diversity": 0.0,
    }
    total = watch + explicit + creator + engagement + recency
    return {"score": total, "breakdown": breakdown}


def _diversity_penalty(video: dict, picked: list[dict]) -> float:
    """Penalty for repeating the last picks' creator / hashtags."""
    if not picked:
        return 0.0
    penalty = 0.0
    username = str(video.get("username") or "").lower()
    if any(str(p.get("username") or "").lower() == username for p in picked[-2:]):
        penalty += DIVERSITY_CREATOR_PENALTY
    tags = {norm_tag(t) for t in (video.get("hashtags") or [])}
    last_tags = {norm_tag(t) for t in (picked[-1].get("hashtags") or [])}
    if tags and last_tags:
        overlap = len(tags & last_tags) / len(tags | last_tags)
        if overlap > 0.5:
            penalty += DIVERSITY_HASHTAG_PENALTY
    return penalty


def _is_unseen(video: dict, taste: dict) -> bool:
    username = str(video.get("username") or "").lower()
    if username in taste["followed"] or username in taste["liked_creators"]:
        return False
    if taste["creators"].get(username, 0.0) != 0.0:
        return False
    tags = [norm_tag(t) for t in (video.get("hashtags") or [])]
    return all(
        taste["tags_watch"].get(t, 0.0) == 0.0
        and taste["tags_explicit"].get(t, 0.0) == 0.0
        for t in tags
    )


def rank_feed(
    videos: list[dict],
    events: list[dict] | None = None,
    limit: int | None = None,
    now: float | None = None,
    exclude_ids: set[str] | None = None,
) -> list[tuple[dict, float, dict]]:
    """Rank videos best-first.

    Returns a list of ``(video, score, breakdown)`` tuples. Greedy pick:
    each round takes the highest effective score (base score + diversity
    penalty against what was already picked), with every 10th slot
    reserved for the best unseen-creator video (exploration).
    """
    taste = build_taste(events or [])
    excluded = {str(x) for x in (exclude_ids or set())}
    candidates = [v for v in videos if str(v.get("id")) not in excluded]
    if not candidates:
        return []

    scored = []
    for v in candidates:
        result = score_video(v, taste, now)
        scored.append((v, result["score"], result["breakdown"]))
    # Highest base score first; video id breaks ties deterministically.
    scored.sort(key=lambda item: (-item[1], str(item[0].get("id"))))

    picked: list[tuple[dict, float, dict]] = []
    remaining = list(scored)
    while remaining and (limit is None or len(picked) < limit):
        slot = len(picked)
        choice = None
        if slot % EXPLORATION_EVERY_N == EXPLORATION_EVERY_N - 1:
            unseen = [item for item in remaining if _is_unseen(item[0], taste)]
            if unseen:
                choice = unseen[0]
        if choice is None:
            best, best_effective = None, None
            picked_videos = [item[0] for item in picked]
            for item in remaining:
                effective = item[1] + _diversity_penalty(item[0], picked_videos)
                if best_effective is None or effective > best_effective:
                    best, best_effective = item, effective
            choice = best
        video, base, breakdown = choice
        picked_videos = [item[0] for item in picked]
        penalty = _diversity_penalty(video, picked_videos)
        final_breakdown = dict(breakdown)
        final_breakdown["diversity"] = round(penalty, 2)
        picked.append((video, round(base + penalty, 2), final_breakdown))
        remaining.remove(choice)
    return picked


def next_videos(
    videos: list[dict],
    events: list[dict] | None = None,
    current_id: str | None = None,
    count: int = 5,
    now: float | None = None,
) -> list[tuple[dict, float, dict]]:
    """The next ``count`` videos after ``current_id`` in ranked order."""
    ranked = rank_feed(videos, events, now=now)
    if current_id is None:
        return ranked[:count]
    for i, (video, _score, _bd) in enumerate(ranked):
        if str(video.get("id")) == str(current_id):
            return ranked[i + 1 : i + 1 + count]
    return ranked[:count]


# --------------------------------------------------------------------------
# Demo + self-test
# --------------------------------------------------------------------------
def _demo_data() -> tuple[list[dict], list[dict]]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    day = 86400_000
    videos = [
        {"id": "v-cat-1", "username": "kitty_club", "hashtags": ["#cat", "#cute"],
         "likes": 1200, "comments": 90, "shares": 40, "views": 30000, "ts": now_ms - day},
        {"id": "v-cat-2", "username": "kitty_club", "hashtags": ["#cat", "#funny"],
         "likes": 900, "comments": 60, "shares": 30, "views": 22000, "ts": now_ms - 2 * day},
        {"id": "v-dog-1", "username": "doggo_daily", "hashtags": ["#dog", "#cute"],
         "likes": 5000, "comments": 300, "shares": 200, "views": 150000, "ts": now_ms - day},
        {"id": "v-football", "username": "sports_hub", "hashtags": ["#football", "#sports"],
         "likes": 20000, "comments": 900, "shares": 800, "views": 900000, "ts": now_ms - day},
        {"id": "v-cook", "username": "chef_life", "hashtags": ["#food", "#recipe"],
         "likes": 800, "comments": 40, "shares": 25, "views": 18000, "ts": now_ms - 3 * day},
        {"id": "v-dance", "username": "dance_crew", "hashtags": ["#dance", "#music"],
         "likes": 3000, "comments": 150, "shares": 120, "views": 80000, "ts": now_ms - 5 * day},
    ]
    # A person who watched cat videos fully, liked a dog video, follows the chef.
    events = [
        {"type": "watch", "video_id": "v-cat-1", "username": "kitty_club",
         "hashtags": ["cat", "cute"], "completion": 0.98,
         "watched_seconds": 29.0, "duration_seconds": 30.0, "ts": 1780000000},
        {"type": "watch", "video_id": "v-cat-2", "username": "kitty_club",
         "hashtags": ["cat", "funny"], "completion": 0.92,
         "watched_seconds": 27.0, "duration_seconds": 30.0, "ts": 1780000000},
        {"type": "like", "video_id": "v-dog-1", "username": "doggo_daily",
         "hashtags": ["dog", "cute"], "ts": 1780000000},
        {"type": "follow", "video_id": "", "username": "chef_life",
         "hashtags": [], "ts": 1780000000},
        {"type": "skip", "video_id": "v-football", "username": "sports_hub",
         "hashtags": ["football", "sports"], "ts": 1780000000},
    ]
    return videos, events


def run_demo() -> int:
    videos, events = _demo_data()
    ranked = rank_feed(videos, events)
    print("For You feed for a cat-lover who follows the chef (demo):\n")
    for i, (video, score, bd) in enumerate(ranked, 1):
        print(f"{i}. {video['id']} (@{video['username']})  score={score}")
        print(f"   watch={bd['watch_tags']} explicit={bd['explicit_tags']} "
              f"creator={bd['creator']} engagement={bd['engagement']} "
              f"recency={bd['recency']} diversity={bd['diversity']}")
    print("\nNext after v-cat-1:", [v["id"] for v, _, _ in next_videos(videos, events, "v-cat-1", 3)])
    return 0


def run_self_test() -> int:
    videos, events = _demo_data()
    order = [v["id"] for v, _, _ in rank_feed(videos, events)]
    # 1. Cat videos the person watched fully outrank the far more popular
    #    football video they skipped.
    assert order.index("v-cat-1") < order.index("v-football"), order
    assert order.index("v-cat-2") < order.index("v-football"), order
    # 2. The followed chef's video is boosted above the dance video.
    assert order.index("v-cook") < order.index("v-dance"), order
    # 3. Cold start (no events): the most engaging recent video leads.
    cold = [v["id"] for v, _, _ in rank_feed(videos, [])]
    assert cold[0] == "v-football", cold
    # 4. A skip pushes that topic down: a fresh football video scores lower
    #    for this person than for a cold-start user.
    fresh = {"id": "v-football-2", "username": "sports_hub",
             "hashtags": ["#football"], "likes": 20000, "comments": 900,
             "shares": 800, "views": 900000,
             "ts": videos[3]["ts"]}
    taste = build_taste(events)
    assert score_video(fresh, taste)["score"] < score_video(fresh, build_taste([]))["score"]
    # 5. Deterministic: same input, same order.
    assert order == [v["id"] for v, _, _ in rank_feed(videos, events)]
    # 6. next_videos returns the videos after the current one.
    nxt = [v["id"] for v, _, _ in next_videos(videos, events, "v-cat-1", 2)]
    assert nxt == order[order.index("v-cat-1") + 1 : order.index("v-cat-1") + 3], nxt
    # 7. Taste survives a JSON round-trip.
    assert taste_from_dict(taste_to_dict(taste))["followed"] == taste["followed"]
    print("All self-tests passed.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flashgram For You ranking")
    parser.add_argument("--videos", help="JSON file: list of video dicts")
    parser.add_argument("--events", help="JSON file: list of event dicts")
    parser.add_argument("--limit", type=int, default=10, help="videos to show")
    parser.add_argument("--current", help="video id to show next videos after")
    parser.add_argument("--demo", action="store_true", help="rank built-in demo data")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.demo or (not args.videos and not args.events):
        return run_demo()

    videos = json.load(open(args.videos)) if args.videos else []
    events = json.load(open(args.events)) if args.events else []
    if args.current:
        ranked = next_videos(videos, events, args.current, args.limit)
    else:
        ranked = rank_feed(videos, events, limit=args.limit)
    print(json.dumps(
        [{"id": v["id"], "username": v.get("username"), "score": s, "breakdown": b}
         for v, s, b in ranked],
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
