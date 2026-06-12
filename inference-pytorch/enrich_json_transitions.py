#!/usr/bin/env python3
"""Add scene transition counts to optical-flow-style JSON result files."""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from transnetv2_infer import DEFAULT_WEIGHTS_PATH, TransNetV2Predictor

DEFAULT_JSON_FILES = [
    "/data/project-vilab/sy/optical_flow_metric/output/result50k.json",
    "/data/project-vilab/sy/optical_flow_metric/output/result_50k_150k.json",
]
DEFAULT_GPU_IDS = [5, 6, 7]


def parse_gpu_ids(gpus_arg: Optional[str]) -> Optional[List[int]]:
    if gpus_arg is None:
        return None
    gpu_ids = [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("empty --gpus value")
    return gpu_ids


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


def atomic_write_json(data: Dict[str, Any], output_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, output_path)


def progress_path_for(output_path: str) -> str:
    return output_path + ".progress.json"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_previous_elapsed(progress_path: str, resume: bool) -> float:
    if not resume or not os.path.isfile(progress_path):
        return 0.0
    with open(progress_path) as f:
        prev = json.load(f)
    if prev.get("status") == "done":
        return float(prev.get("accumulated_elapsed_seconds", prev.get("elapsed_seconds", 0)))
    return float(prev.get("accumulated_elapsed_seconds", 0))


def build_timing_info(
    started_at: float,
    started_at_iso: str,
    previous_elapsed: float,
    done_in_batch: int,
    status: str,
) -> Dict[str, Any]:
    now = time.time()
    elapsed_this_run = max(now - started_at, 0.0)
    accumulated = previous_elapsed + elapsed_this_run
    rate = done_in_batch / elapsed_this_run if elapsed_this_run > 0 else 0.0

    info = {
        "status": status,
        "started_at": started_at_iso,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": round(elapsed_this_run, 1),
        "accumulated_elapsed_seconds": round(accumulated, 1),
        "elapsed_human": format_duration(accumulated),
        "elapsed_this_run_human": format_duration(elapsed_this_run),
        "videos_per_second_this_run": round(rate, 3),
    }
    if status == "done":
        info["finished_at"] = datetime.now(timezone.utc).isoformat()
    return info


def write_progress_file(
    progress_path: str,
    output_path: str,
    total: int,
    pending_total: int,
    processed: int,
    failed: int,
    skipped: int,
    done_in_batch: int,
    started_at: float,
    started_at_iso: str,
    previous_elapsed: float,
    status: str = "running",
) -> Dict[str, Any]:
    finished = skipped + processed + failed
    percent = (finished / total * 100.0) if total else 100.0
    batch_percent = (done_in_batch / pending_total * 100.0) if pending_total else 100.0
    elapsed_this_run = max(time.time() - started_at, 1e-6)
    accumulated = previous_elapsed + elapsed_this_run
    rate = done_in_batch / elapsed_this_run
    remaining = pending_total - done_in_batch
    eta_seconds = int(remaining / rate) if rate > 0 else None

    timing = build_timing_info(
        started_at, started_at_iso, previous_elapsed, done_in_batch, status
    )
    payload = {
        **timing,
        "json_path": output_path,
        "total_videos": total,
        "pending_this_run": pending_total,
        "done_this_run": done_in_batch,
        "processed_this_run": processed,
        "failed_this_run": failed,
        "skipped_existing": skipped,
        "finished_total": finished,
        "remaining_total": max(total - finished, 0),
        "percent_total": round(percent, 2),
        "percent_this_run": round(batch_percent, 2),
        "eta_seconds": eta_seconds,
        "eta_human": format_duration(eta_seconds),
    }
    atomic_write_json(payload, progress_path)
    return payload


def report_progress(
    progress_path: str,
    output_path: str,
    data: Dict[str, Any],
    total: int,
    pending_total: int,
    processed: int,
    failed: int,
    skipped: int,
    done_in_batch: int,
    started_at: float,
    started_at_iso: str,
    previous_elapsed: float,
    save_every: int,
    label: str,
    status: str = "running",
) -> None:
    progress = write_progress_file(
        progress_path, output_path, total, pending_total,
        processed, failed, skipped, done_in_batch,
        started_at, started_at_iso, previous_elapsed, status=status,
    )
    data["transition_scoring_timing"] = {
        k: progress[k] for k in (
            "status", "started_at", "updated_at", "finished_at",
            "elapsed_seconds_this_run", "accumulated_elapsed_seconds",
            "elapsed_human", "elapsed_this_run_human",
            "videos_per_second_this_run", "eta_seconds", "eta_human",
        ) if k in progress
    }

    if done_in_batch % 100 != 0 and done_in_batch != pending_total:
        return

    print(
        f"[enrich] progress {label} "
        f"run={done_in_batch}/{pending_total}, "
        f"total={skipped + processed + failed}/{total}, "
        f"processed={processed}, skipped={skipped}, failed={failed}, "
        f"elapsed={progress['elapsed_human']}, eta={progress.get('eta_human', 'unknown')}",
        flush=True,
    )

    if save_every > 0 and (processed + failed) % save_every == 0:
        update_aggregate(data)
        atomic_write_json(data, output_path)
        print(f"[enrich] checkpoint saved -> {output_path}", flush=True)


def _gpu_worker(
    gpu_id: int,
    weights_path: str,
    threshold: float,
    task_queue,
    result_queue,
) -> None:
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)

    predictor = TransNetV2Predictor(
        weights_path=weights_path,
        device=f"cuda:{gpu_id}",
    )
    print(f"[enrich] worker started on GPU {gpu_id}", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break

        video_key, video_path, max_frames = task
        try:
            result = score_video(predictor, video_path, max_frames, threshold)
            result_queue.put((video_key, result, None))
        except Exception as exc:
            result_queue.put((video_key, None, str(exc)))


def apply_result(entry: Dict[str, Any], result: Optional[Dict[str, Any]], error: Optional[str]) -> bool:
    if error:
        entry["transition_error"] = error
        entry.pop("transition_count", None)
        entry.pop("transition_details", None)
        return False

    entry.update(result)
    entry.pop("transition_error", None)
    return True


def collect_pending_tasks(
    videos: Dict[str, Any],
    resume: bool,
    default_max_frames: Optional[int],
) -> Tuple[List[Tuple[str, str, Optional[int]]], int, int]:
    pending = []
    skipped = 0
    failed = 0

    for video_key, entry in videos.items():
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
        pending.append((video_key, video_path, max_frames))

    return pending, skipped, failed


def enrich_json_single_gpu(
    data: Dict[str, Any],
    output_path: str,
    progress_path: str,
    started_at: float,
    started_at_iso: str,
    previous_elapsed: float,
    predictor: TransNetV2Predictor,
    threshold: float,
    default_max_frames: Optional[int],
    resume: bool,
    save_every: int,
) -> None:
    videos = data.get("videos", {})
    total = len(videos)
    pending, skipped, failed = collect_pending_tasks(videos, resume, default_max_frames)
    processed = 0
    pending_total = len(pending)

    print(f"[enrich] single-GPU mode ({pending_total} videos to score)", flush=True)
    write_progress_file(
        progress_path, output_path, total, pending_total,
        processed, failed, skipped, 0,
        started_at, started_at_iso, previous_elapsed, status="running",
    )

    for idx, (video_key, video_path, max_frames) in enumerate(pending, start=1):
        entry = videos[video_key]
        try:
            result = score_video(predictor, video_path, max_frames, threshold)
            if apply_result(entry, result, None):
                processed += 1
            else:
                failed += 1
        except Exception as exc:
            apply_result(entry, None, str(exc))
            failed += 1

        report_progress(
            progress_path, output_path, data, total, pending_total,
            processed, failed, skipped, idx,
            started_at, started_at_iso, previous_elapsed, save_every,
            f"{idx}/{pending_total}",
        )

    final_progress = write_progress_file(
        progress_path, output_path, total, pending_total,
        processed, failed, skipped, pending_total,
        started_at, started_at_iso, previous_elapsed, status="done",
    )
    data["transition_scoring_timing"] = {
        k: final_progress[k] for k in (
            "status", "started_at", "updated_at", "finished_at",
            "elapsed_seconds_this_run", "accumulated_elapsed_seconds",
            "elapsed_human", "elapsed_this_run_human",
            "videos_per_second_this_run",
        ) if k in final_progress
    }
    print(
        f"[enrich] done -> {output_path} "
        f"(processed={processed}, skipped={skipped}, failed={failed}, "
        f"elapsed={final_progress['elapsed_human']})"
    )


def enrich_json_multi_gpu(
    data: Dict[str, Any],
    output_path: str,
    progress_path: str,
    started_at: float,
    started_at_iso: str,
    previous_elapsed: float,
    weights_path: str,
    gpu_ids: List[int],
    threshold: float,
    default_max_frames: Optional[int],
    resume: bool,
    save_every: int,
) -> None:
    videos = data.get("videos", {})
    total = len(videos)
    pending, skipped, failed = collect_pending_tasks(videos, resume, default_max_frames)
    processed = 0
    pending_total = len(pending)

    print(
        f"[enrich] multi-GPU mode gpus={gpu_ids} ({pending_total} videos to score)",
        flush=True,
    )

    if not pending:
        write_progress_file(
            progress_path, output_path, total, 0,
            processed, failed, skipped, 0,
            started_at, started_at_iso, previous_elapsed, status="done",
        )
        print(f"[enrich] nothing to do (skipped={skipped}, failed={failed})", flush=True)
        return

    write_progress_file(
        progress_path, output_path, total, pending_total,
        processed, failed, skipped, 0,
        started_at, started_at_iso, previous_elapsed, status="running",
    )

    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in pending:
        task_queue.put(task)
    for _ in gpu_ids:
        task_queue.put(None)

    workers = []
    for gpu_id in gpu_ids:
        proc = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, weights_path, threshold, task_queue, result_queue),
        )
        proc.start()
        workers.append(proc)

    for idx in range(1, pending_total + 1):
        video_key, result, error = result_queue.get()
        entry = videos[video_key]
        if apply_result(entry, result, error):
            processed += 1
        else:
            failed += 1

        report_progress(
            progress_path, output_path, data, total, pending_total,
            processed, failed, skipped, idx,
            started_at, started_at_iso, previous_elapsed, save_every,
            f"{idx}/{pending_total}",
        )

    for proc in workers:
        proc.join()

    final_progress = write_progress_file(
        progress_path, output_path, total, pending_total,
        processed, failed, skipped, pending_total,
        started_at, started_at_iso, previous_elapsed, status="done",
    )
    data["transition_scoring_timing"] = {
        k: final_progress[k] for k in (
            "status", "started_at", "updated_at", "finished_at",
            "elapsed_seconds_this_run", "accumulated_elapsed_seconds",
            "elapsed_human", "elapsed_this_run_human",
            "videos_per_second_this_run",
        ) if k in final_progress
    }
    print(
        f"[enrich] done -> {output_path} "
        f"(processed={processed}, skipped={skipped}, failed={failed}, "
        f"elapsed={final_progress['elapsed_human']})"
    )


def enrich_json(
    input_path: str,
    output_path: str,
    weights_path: str,
    threshold: float,
    default_max_frames: Optional[int],
    resume: bool,
    save_every: int,
    gpu_ids: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> None:
    source_path = input_path
    if resume and os.path.isfile(output_path):
        try:
            same_file = os.path.samefile(input_path, output_path)
        except FileNotFoundError:
            same_file = False
        if not same_file:
            source_path = output_path
            print(f"[enrich] resuming from existing output: {output_path}")

    with open(source_path) as f:
        data = json.load(f)

    data["transition_weights_path"] = weights_path
    data["transition_threshold"] = threshold
    if gpu_ids:
        data["transition_gpu_ids"] = gpu_ids

    total = len(data.get("videos", {}))
    progress_path = progress_path_for(output_path)
    previous_elapsed = load_previous_elapsed(progress_path, resume)
    started_at = time.time()
    started_at_iso = datetime.now(timezone.utc).isoformat()
    print(f"[enrich] {source_path} -> {output_path} ({total} videos)")
    print(f"[enrich] progress file: {progress_path}")
    if previous_elapsed > 0:
        print(f"[enrich] previous elapsed: {format_duration(previous_elapsed)}")

    if gpu_ids and len(gpu_ids) > 1:
        enrich_json_multi_gpu(
            data=data,
            output_path=output_path,
            progress_path=progress_path,
            started_at=started_at,
            started_at_iso=started_at_iso,
            previous_elapsed=previous_elapsed,
            weights_path=weights_path,
            gpu_ids=gpu_ids,
            threshold=threshold,
            default_max_frames=default_max_frames,
            resume=resume,
            save_every=save_every,
        )
    else:
        if gpu_ids:
            device = f"cuda:{gpu_ids[0]}"
        predictor = TransNetV2Predictor(weights_path=weights_path, device=device)
        enrich_json_single_gpu(
            data=data,
            output_path=output_path,
            progress_path=progress_path,
            started_at=started_at,
            started_at_iso=started_at_iso,
            previous_elapsed=previous_elapsed,
            predictor=predictor,
            threshold=threshold,
            default_max_frames=default_max_frames,
            resume=resume,
            save_every=save_every,
        )

    update_aggregate(data)
    atomic_write_json(data, output_path)


def resolve_json_files(json_files: Optional[List[str]], use_defaults: bool) -> List[str]:
    if json_files:
        return json_files
    if use_defaults:
        return list(DEFAULT_JSON_FILES)
    raise SystemExit("No JSON files given. Pass paths or use --use-default-jsons.")


def main():
    parser = argparse.ArgumentParser(
        description="Add transition_count to each video entry in a result JSON file."
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        help="input JSON files (omit with --use-default-jsons for built-in paths)",
    )
    parser.add_argument(
        "--use-default-jsons",
        action="store_true",
        help="use default OpenVid result JSON paths under /data/project-vilab/sy/optical_flow_metric/output/",
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
        help="overwrite input JSON in place (atomic write)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"path to .pth weights (default: {DEFAULT_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=f"comma-separated GPU ids for multi-GPU inference (e.g. 5,6,7; default in run script: {','.join(map(str, DEFAULT_GPU_IDS))})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="single device for inference (e.g. cpu, cuda:0). ignored when --gpus has 2+ ids",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="default max frames if entry has no max_frames_limit (omit = use per-entry value)",
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

    if args.gpus and args.device:
        print("[enrich] warning: --device is ignored when --gpus is set", file=sys.stderr)

    json_files = resolve_json_files(args.json_files, args.use_default_jsons)
    weights_path = os.path.abspath(args.weights)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    gpu_ids = parse_gpu_ids(args.gpus)
    if gpu_ids:
        print(f"[enrich] weights: {weights_path}, gpus: {gpu_ids}")
    else:
        print(f"[enrich] weights: {weights_path}, device: {args.device or 'auto'}")

    job_started = time.time()
    job_started_iso = datetime.now(timezone.utc).isoformat()

    for input_path in json_files:
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

        file_started = time.time()
        enrich_json(
            input_path=input_path,
            output_path=output_path,
            weights_path=weights_path,
            threshold=args.threshold,
            default_max_frames=args.max_frames,
            resume=args.resume,
            save_every=args.save_every,
            gpu_ids=gpu_ids,
            device=args.device,
        )
        file_elapsed = time.time() - file_started
        print(
            f"[enrich] file elapsed: {format_duration(file_elapsed)} "
            f"({round(file_elapsed, 1)}s) -> {output_path}"
        )

    total_elapsed = time.time() - job_started
    timing_summary_path = os.path.join(
        _SCRIPT_DIR, "logs", "enrich_timing_summary.json"
    )
    os.makedirs(os.path.dirname(timing_summary_path), exist_ok=True)
    timing_summary = {
        "job_started_at": job_started_iso,
        "job_finished_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_elapsed_human": format_duration(total_elapsed),
        "json_files": json_files,
        "gpu_ids": gpu_ids,
    }
    atomic_write_json(timing_summary, timing_summary_path)
    print(
        f"[enrich] all files done in {format_duration(total_elapsed)} "
        f"({round(total_elapsed, 1)}s)"
    )
    print(f"[enrich] timing summary: {timing_summary_path}")


if __name__ == "__main__":
    main()
