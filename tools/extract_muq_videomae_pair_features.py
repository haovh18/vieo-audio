#!/usr/bin/env python3
"""Extract paired pretrained VideoMAE and MuQ features for a bounded sample set."""

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import torchaudio
from muq import MuQ
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor


ROOT = Path(__file__).resolve().parents[1]
VIDEO_FIELDS = ("video_path", "path", "video", "mp4_path", "video_file")
AUDIO_FIELDS = ("audio_path", "wav_path", "mp3_path", "music_path", "audio_file")
ID_FIELDS = ("sample_id", "id", "item_id", "video_id")
MODEL_SR = 24000
MODEL_AUDIO_SAMPLES = 10 * MODEL_SR


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def first_value(row, fields):
    for field in fields:
        if row.get(field):
            return field, row[field]
    return None, None


def parse_layers(value):
    layers = sorted(set(int(item) for item in value.split(",")))
    if not layers or min(layers) < 0:
        raise argparse.ArgumentTypeError("layers must be non-negative integers")
    return layers


def decode_video(video_path, processor):
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name",
                "-show_entries", "format=duration", "-of", "json", str(video_path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    )
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path), "-an",
            "-vf", f"fps=16/{duration}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        stderr=subprocess.STDOUT,
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(
        -1, int(stream["height"]), int(stream["width"]), 3
    )
    if len(frames) < 16:
        raise RuntimeError(f"decoded {len(frames)} frames, expected at least 16")
    inputs = processor([frame.copy() for frame in frames[:16]], return_tensors="pt")
    return inputs["pixel_values"], {
        "codec": stream["codec_name"],
        "duration_seconds": duration,
        "decoded_frames_used": 16,
    }


def load_audio(audio_path, duration_mode):
    waveform, sample_rate = torchaudio.load(str(audio_path))
    duration = waveform.shape[1] / sample_rate
    channels = waveform.shape[0]
    if channels > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != MODEL_SR:
        waveform = torchaudio.functional.resample(waveform, sample_rate, MODEL_SR)
    if duration_mode == "full":
        action = "keep_full_audio"
    else:
        if waveform.shape[1] < MODEL_AUDIO_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, MODEL_AUDIO_SAMPLES - waveform.shape[1])
            )
            action = "zero_pad_to_10_seconds"
        elif waveform.shape[1] > MODEL_AUDIO_SAMPLES:
            waveform = waveform[:, :MODEL_AUDIO_SAMPLES]
            action = "truncate_to_first_10_seconds"
        else:
            action = "none"
    return waveform, {
        "original_sample_rate": sample_rate,
        "original_channels": channels,
        "duration_seconds": duration,
        "length_action": action,
        "model_input_num_samples": waveform.shape[1],
        "model_input_duration_seconds": waveform.shape[1] / MODEL_SR,
    }


def stats(arrays):
    if not arrays:
        return {"all_finite": False, "has_nan": False, "has_inf": False,
                "all_l2_norms_positive": False, "zero_vector_count": 0}
    array = np.stack(arrays)
    norms = np.linalg.norm(array, axis=1)
    return {
        "all_finite": bool(np.isfinite(array).all()),
        "has_nan": bool(np.isnan(array).any()),
        "has_inf": bool(np.isinf(array).any()),
        "all_l2_norms_positive": bool((norms > 0).all()),
        "zero_vector_count": int((norms == 0).sum()),
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
    }


def markdown(report, output_dir):
    label = "SMOKE" if report["requested_max_samples"] <= 10 else "FIRST100"
    vshape = report["video_final_shape"]
    ashape = report["audio_final_shape"]
    can_continue = report["num_success"] > 0 and report["finite_check"]["all"]
    text = f"""# B Method {label} Feature Extraction

- Status: **{"PASS" if can_continue else "FAIL"}**
- Rows in input index: {report["total_rows_in_sample_index"]}
- Requested samples: {report["requested_max_samples"]}
- Valid pairs found: {report["num_valid_pairs_found"]}
- Successful pairs: {report["num_success"]}
- Failed rows/samples: {report["num_failed"]}
- Video final shape: `{vshape}`
- Audio final shape: `{ashape}`
- Selected layers: `{report["selected_layers"]}`
- Video layer shapes: `{report["video_layer_shapes"]}`
- Audio layer shapes: `{report["audio_layer_shapes"]}`
- All values finite: {report["finite_check"]["all"]}
- NaN present: {report["finite_check"]["has_nan"]}
- Inf present: {report["finite_check"]["has_inf"]}
- Zero vectors present: {not report["l2_norm_check"]["all_positive"]}
- Missing/invalid path rows: {report["path_failure_count"]}
- Device: `{report["device"]}`
- Elapsed seconds: {report["elapsed_seconds"]}

The input index contains only {report["total_rows_in_sample_index"]} rows, so this run cannot
produce more than that number of pairs. Feature row order exactly matches `sample_ids.json`
and `pair_paths.json`.
"""
    name = "FEATURE_SMOKE_RESULT.md" if label == "SMOKE" else "FEATURE_FIRST100_RESULT.md"
    (output_dir / name).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", default="outputs/muq_videomae/sample_index.csv")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,4,8,12"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--audio-duration",
        choices=("first10", "full"),
        default="first10",
        help="Use a 10-second input or the complete mono resampled waveform.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    index_path, output_dir = resolve(args.sample_index), resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / "extract_report.json"
    if sentinel.exists() and not args.overwrite:
        raise SystemExit(f"{sentinel} exists; use --overwrite to replace generated features")
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    started = time.monotonic()

    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    processor = VideoMAEImageProcessor.from_pretrained(
        str(ROOT / "weights/videomae/hf_videomae_base"), local_files_only=True
    )
    video_model = VideoMAEForPreTraining.from_pretrained(
        str(ROOT / "weights/videomae/hf_videomae_base"),
        local_files_only=True,
        use_safetensors=True,
    ).to(device).eval()
    audio_model = MuQ.from_pretrained(str(ROOT / "weights/muq/MuQ-large-msd-iter"))
    audio_model = audio_model.to(device).eval()

    sample_ids, pair_paths, statuses, failures = [], [], [], []
    video_features = {"final": []}
    audio_features = {"final": []}
    for layer in args.layers:
        video_features[layer] = []
        audio_features[layer] = []
    valid_found = 0

    for row_number, row in enumerate(rows, start=2):
        if len(sample_ids) >= args.max_samples:
            break
        id_field, sample_id = first_value(row, ID_FIELDS)
        video_field, video_value = first_value(row, VIDEO_FIELDS)
        audio_field, audio_value = first_value(row, AUDIO_FIELDS)
        sample_id = sample_id or f"row_{row_number}"
        base_status = {"row_number": row_number, "sample_id": sample_id}
        if not video_value or not audio_value:
            failure = {**base_status, "status": "failed_path",
                       "error": "missing video or audio path field"}
            failures.append(failure)
            statuses.append(failure)
            continue
        video_path, audio_path = resolve(video_value), resolve(audio_value)
        missing = [str(path) for path in (video_path, audio_path) if not path.is_file()]
        if missing:
            failure = {**base_status, "status": "failed_path",
                       "error": f"files not found: {missing}"}
            failures.append(failure)
            statuses.append(failure)
            continue
        valid_found += 1
        sample_started = time.monotonic()
        try:
            pixel_values, video_meta = decode_video(video_path, processor)
            waveform, audio_meta = load_audio(audio_path, args.audio_duration)
            pixel_values = pixel_values.to(device)
            waveform = waveform.to(device)
            with torch.inference_mode():
                video_output = video_model.videomae(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                audio_output = audio_model(waveform, output_hidden_states=True)
            if max(args.layers) >= len(video_output.hidden_states):
                raise IndexError("selected VideoMAE layer exceeds hidden_states count")
            if max(args.layers) >= len(audio_output.hidden_states):
                raise IndexError("selected MuQ layer exceeds hidden_states count")
            video_features["final"].append(
                video_output.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
            )
            audio_features["final"].append(
                audio_output.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
            )
            for layer in args.layers:
                video_features[layer].append(
                    video_output.hidden_states[layer].mean(dim=1)[0].float().cpu().numpy()
                )
                audio_features[layer].append(
                    audio_output.hidden_states[layer].mean(dim=1)[0].float().cpu().numpy()
                )
            sample_ids.append(sample_id)
            pair_paths.append({
                "sample_id": sample_id,
                "video_path": video_value,
                "audio_path": audio_value,
            })
            statuses.append({
                **base_status, "status": "success",
                "video": video_meta, "audio": audio_meta,
                "elapsed_seconds": round(time.monotonic() - sample_started, 3),
            })
        except Exception:
            failure = {
                **base_status, "status": "failed_forward",
                "video_path": video_value, "audio_path": audio_value,
                "error": traceback.format_exc(),
            }
            failures.append(failure)
            statuses.append(failure)
        finally:
            for item in ("pixel_values", "waveform", "video_output", "audio_output"):
                if item in locals():
                    del locals()[item]
            torch.cuda.empty_cache()

    if sample_ids:
        np.save(output_dir / "video_final.npy", np.stack(video_features["final"]))
        np.save(output_dir / "audio_final.npy", np.stack(audio_features["final"]))
        for layer in args.layers:
            np.save(output_dir / f"video_layer_{layer}.npy", np.stack(video_features[layer]))
            np.save(output_dir / f"audio_layer_{layer}.npy", np.stack(audio_features[layer]))
    else:
        np.save(output_dir / "video_final.npy", np.empty((0, 768), dtype=np.float32))
        np.save(output_dir / "audio_final.npy", np.empty((0, 1024), dtype=np.float32))
        for layer in args.layers:
            np.save(output_dir / f"video_layer_{layer}.npy", np.empty((0, 768), dtype=np.float32))
            np.save(output_dir / f"audio_layer_{layer}.npy", np.empty((0, 1024), dtype=np.float32))

    (output_dir / "sample_ids.json").write_text(
        json.dumps(sample_ids, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "pair_paths.json").write_text(
        json.dumps(pair_paths, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    all_groups = list(video_features.values()) + list(audio_features.values())
    group_stats = [stats(group) for group in all_groups]
    num_success = len(sample_ids)
    report = {
        "sample_index": str(index_path.relative_to(ROOT)),
        "total_rows_in_sample_index": len(rows),
        "requested_max_samples": args.max_samples,
        "num_valid_pairs_found": valid_found,
        "num_success": num_success,
        "num_failed": len(failures),
        "selected_layers": args.layers,
        "audio_duration_mode": args.audio_duration,
        "video_sampling": "16 frames uniformly over the complete video duration",
        "video_final_shape": [num_success, 768],
        "audio_final_shape": [num_success, 1024],
        "video_layer_shapes": {str(x): [num_success, 768] for x in args.layers},
        "audio_layer_shapes": {str(x): [num_success, 1024] for x in args.layers},
        "per_sample_status": statuses,
        "failed_samples": failures,
        "path_failure_count": sum(x["status"] == "failed_path" for x in failures),
        "finite_check": {
            "all": all(x["all_finite"] for x in group_stats),
            "has_nan": any(x["has_nan"] for x in group_stats),
            "has_inf": any(x["has_inf"] for x in group_stats),
        },
        "l2_norm_check": {
            "all_positive": all(x["all_l2_norms_positive"] for x in group_stats),
            "total_zero_vectors": sum(x["zero_vector_count"] for x in group_stats),
        },
        "device": str(device),
        "dtype": "float32",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    sentinel.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown(report, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
