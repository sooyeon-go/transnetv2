#!/usr/bin/env python3
"""Show progress for enrich_json_transitions jobs."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from enrich_json_transitions import DEFAULT_JSON_FILES, progress_path_for


def format_eta(seconds):
    if seconds is None:
        return "unknown"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_progress(json_path: str):
    progress_path = progress_path_for(json_path)
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            return json.load(f), progress_path

    if not os.path.isfile(json_path):
        return None, progress_path

    with open(json_path) as f:
        data = json.load(f)

    total = len(data.get("videos", {}))
    scored = data.get("num_videos_transition_scored")
    errors = data.get("num_videos_transition_errors", 0)
    if scored is None:
        return None, progress_path

    finished = scored + errors
    return {
        "status": "done" if finished >= total else "unknown",
        "json_path": json_path,
        "total_videos": total,
        "finished_total": finished,
        "remaining_total": max(total - finished, 0),
        "percent_total": round(finished / total * 100, 2) if total else 100.0,
        "processed_this_run": scored,
        "failed_this_run": errors,
        "note": "derived from JSON aggregate fields (no .progress.json found)",
    }, progress_path


def print_progress(info: dict, progress_path: str) -> None:
    print(f"JSON:     {info.get('json_path')}")
    print(f"Progress: {progress_path} ({'found' if os.path.isfile(progress_path) else 'missing'})")
    print(f"Status:   {info.get('status')}")
    if info.get("updated_at"):
        print(f"Updated:  {info['updated_at']}")
    print(
        f"Total:    {info.get('finished_total', '?')}/{info.get('total_videos', '?')} "
        f"({info.get('percent_total', '?')}%)"
    )
    if info.get("remaining_total") is not None:
        print(f"Remaining:{info['remaining_total']}")
    if info.get("done_this_run") is not None:
        print(
            f"This run: {info.get('done_this_run')}/{info.get('pending_this_run')} "
            f"({info.get('percent_this_run', '?')}%)"
        )
    print(
        f"OK/Fail:  processed={info.get('processed_this_run', info.get('processed', '?'))}, "
        f"failed={info.get('failed_this_run', info.get('failed', '?'))}, "
        f"skipped={info.get('skipped_existing', info.get('skipped', 0))}"
    )
    if info.get("videos_per_second"):
        print(f"Speed:    {info['videos_per_second']} videos/sec")
    if info.get("elapsed_seconds") is not None:
        print(f"Elapsed:  {format_eta(info['elapsed_seconds'])}")
    if info.get("eta_seconds") is not None and info.get("status") == "running":
        print(f"ETA:      {format_eta(info['eta_seconds'])}")
    if info.get("note"):
        print(f"Note:     {info['note']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Check enrich_json_transitions progress.")
    parser.add_argument(
        "json_files",
        nargs="*",
        help="result JSON paths (default: built-in OpenVid result files)",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="refresh every N seconds (0 = print once)",
    )
    args = parser.parse_args()

    json_files = args.json_files or list(DEFAULT_JSON_FILES)

    while True:
        if args.watch:
            print(f"=== {datetime.now(timezone.utc).isoformat()} ===")
        for json_path in json_files:
            info, progress_path = load_progress(json_path)
            if info is None:
                print(f"JSON:     {json_path}")
                print(f"Progress: {progress_path} (no progress data yet)\n")
                continue
            print_progress(info, progress_path)

        if not args.watch:
            break
        import time
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
