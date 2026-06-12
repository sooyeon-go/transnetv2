#!/usr/bin/env python3
"""Add scene transition counts to optical-flow-style JSON result files."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

from transnetv2_infer import TransNetV2Predictor


def resolve_max_frames(entry: Dict[str, Any], cli_default: Optional[int]) -> Optional[int]:
    if "max_frames_limit" in entry:
        return int(entry["max_frames_limit"])
    if cli_default is not None:
        return cli_default
    return None


def score_video(
    predictor: TransNetV2Predictor,
    video_path: str,
    max_frames: Optional[int],
    threshold: float,
) -> Dict[str, Any]:
    video, single_frame_predictions, _ = predictor.predict_video(
        video_path, max_frames=max_frames, verbose=False
    )
    scenes = predictor.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    transition_count = predictor.count_transitions(single_frame_predictions, threshold=threshold)

    return {
        "transition_count": int(transition_count),
        "transition_details": {
            "num_scenes": int(len(scenes)),
            "used_frame_count": int(len(video)),
            "max_frames_limit": max_frames,
            "threshold": threshold,
            "scene_ranges": scenes.tolist(),
        },
    }


def update_aggregate(data: Dict[str, Any]) -> None:
    counts = []
    errors = 0
    for entry in data.get("videos", {}).values():
        if "transition_count" in entry:
            counts.append(entry["transition_count"])
        elif entry.get("transition_error"):
            errors += 1

    data["num_videos_transition_scored"] = len(counts)
    data["num_videos_transition_errors"] = errors
    if counts:
        data["aggregate_mean_transition_count"] = float(np.mean(counts))
    else:
        data.pop("aggregate_mean_transition_count", None)


def enrich_json(
    input_path: str,
    output_path: str,
    predictor: TransNetV2Predictor,
    threshold: float,
    default_max_frames: Optional[int],
    resume: bool,
    save_every: int,
) -> None:
    source_path = input_path
    if resume and os.path.isfile(output_path) and not os.path.samefile(input_path, output_path):
        source_path = output_path
        print(f"[enrich] resuming from existing output: {output_path}")

    with open(source_path) as f:
        data = json.load(f)

    videos = data.get("videos", {})
    total = len(videos)
    processed = 0
    skipped = 0
    failed = 0

    print(f"[enrich] {input_path}: {total} videos")

    for idx, (video_key, entry) in enumerate(videos.items(), start=1):
        if resume and "transition_count" in entry and not entry.get("transition_error"):
            skipped += 1
            continue

        video_path = entry.get("video_path")
        if not video_path:
            entry["transition_error"] = "missing video_path"
            failed += 1
            continue

        if not os.path.isfile(video_path):
            entry["transition_error"] = f"file not found: {video_path}"
            failed += 1
            continue

        max_frames = resolve_max_frames(entry, default_max_frames)

        try:
            result = score_video(predictor, video_path, max_frames, threshold)
            entry.update(result)
            entry.pop("transition_error", None)
            processed += 1
        except Exception as exc:
            entry["transition_error"] = str(exc)
            entry.pop("transition_count", None)
            entry.pop("transition_details", None)
            failed += 1

        if save_every > 0 and (processed + failed) % save_every == 0:
            update_aggregate(data)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(
                f"[enrich] checkpoint {idx}/{total} "
                f"(processed={processed}, skipped={skipped}, failed={failed})"
            )

        if idx % 100 == 0 or idx == total:
            print(
                f"[enrich] progress {idx}/{total} "
                f"(processed={processed}, skipped={skipped}, failed={failed})",
                flush=True,
            )

    update_aggregate(data)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(
        f"[enrich] done -> {output_path} "
        f"(processed={processed}, skipped={skipped}, failed={failed})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add transition_count to each video entry in a result JSON file."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="input JSON files (e.g. result50k.json result_50k_150k.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory (default: same directory as each input file)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_with_transitions",
        help="suffix for output filename before .json (default: _with_transitions)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="overwrite input JSON in place (use with care)",
    )
    parser.add_argument("--weights", type=str, default=None, help="path to .pth weights")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="default max frames if entry has no max_frames_limit (omit = full video)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip entries that already have transition_count",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="write checkpoint every N newly processed videos (0 = only at end)",
    )
    args = parser.parse_args()

    predictor = TransNetV2Predictor(weights_path=args.weights, device=args.device)

    for input_path in args.json_files:
        if not os.path.isfile(input_path):
            print(f"[enrich] skip missing file: {input_path}", file=sys.stderr)
            continue

        if args.inplace:
            output_path = input_path
        else:
            base, ext = os.path.splitext(os.path.basename(input_path))
            out_dir = args.output_dir or os.path.dirname(input_path)
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"{base}{args.suffix}{ext}")

        started = time.time()
        enrich_json(
            input_path=input_path,
            output_path=output_path,
            predictor=predictor,
            threshold=args.threshold,
            default_max_frames=args.max_frames,
            resume=args.resume,
            save_every=args.save_every,
        )
        print(f"[enrich] elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
