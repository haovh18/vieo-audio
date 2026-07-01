#!/usr/bin/env python3
"""Run zero-shot ImageBind audio-video retrieval on paired local media."""

import argparse
import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# The host may expose OMP_NUM_THREADS=0, which libgomp rejects.
if int(os.environ.get("OMP_NUM_THREADS", "1") or "1") < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "weights/imagebind/imagebind_huge.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ImageBind embeddings and evaluate paired retrieval."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("datasets/HarmonySet/processed/audio_video_index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/imagebind/baseline",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def retrieval_metrics(similarity: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-similarity, axis=1, kind="stable")
    targets = np.arange(similarity.shape[0])[:, None]
    ranks = np.argmax(order == targets, axis=1) + 1
    return {
        "R@1": float(np.mean(ranks <= 1) * 100.0),
        "R@5": float(np.mean(ranks <= 5) * 100.0),
        "R@10": float(np.mean(ranks <= 10) * 100.0),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def load_pairs(index_csv: Path) -> Tuple[pd.DataFrame, int]:
    frame = pd.read_csv(index_csv)
    required_columns = {"video_path", "audio_path", "status"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Index CSV is missing columns: {sorted(missing)}")

    successful = frame.loc[
        frame["status"].astype(str).str.lower() == "success",
        ["video_path", "audio_path"],
    ].copy()
    ignored = len(frame) - len(successful)
    successful.reset_index(drop=True, inplace=True)
    return successful, ignored


def chunks(frame: pd.DataFrame, size: int) -> Sequence[pd.DataFrame]:
    return [frame.iloc[start : start + size] for start in range(0, len(frame), size)]


def create_compatible_video(video_path: str, cache_dir: Path) -> Path:
    source = Path(video_path)
    path_hash = hashlib.sha1(
        source.resolve().as_posix().encode("utf-8")
    ).hexdigest()[:10]
    proxy = cache_dir / f"{source.stem}_{path_hash}.mp4"
    if proxy.is_file():
        return proxy

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = proxy.with_name(f".{proxy.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "fps=2,scale=if(gt(iw\\,ih)\\,-2\\,256):if(gt(iw\\,ih)\\,256\\,-2)",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg proxy conversion failed with code {result.returncode}")
    temporary.replace(proxy)
    return proxy


def load_video(video_path: str, device: str, cache_dir: Path) -> torch.Tensor:
    try:
        return data.load_and_transform_video_data([video_path], device)
    except Exception as original_error:
        proxy = create_compatible_video(video_path, cache_dir)
        warnings.warn(
            f"Direct video decode failed for {video_path!r}; using codec-compatible "
            f"proxy {proxy}: {original_error}",
            stacklevel=1,
        )
        return data.load_and_transform_video_data([str(proxy)], device)


def prepare_batch(
    batch: pd.DataFrame, device: str, video_cache_dir: Path
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, str]], int]:
    video_inputs: List[torch.Tensor] = []
    audio_inputs: List[torch.Tensor] = []
    valid_rows: List[Dict[str, str]] = []
    skipped = 0

    for row in batch.itertuples(index=False):
        video_path = str(row.video_path)
        audio_path = str(row.audio_path)
        try:
            video = load_video(video_path, device, video_cache_dir)
            audio = data.load_and_transform_audio_data([audio_path], device)
            video_inputs.append(video)
            audio_inputs.append(audio)
            valid_rows.append(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                }
            )
        except Exception as error:
            skipped += 1
            warnings.warn(
                f"Skipping pair video={video_path!r}, audio={audio_path!r}: {error}",
                stacklevel=1,
            )

    return video_inputs, audio_inputs, valid_rows, skipped


def infer_batch(
    model: torch.nn.Module,
    video_inputs: List[torch.Tensor],
    audio_inputs: List[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = {
        ModalityType.VISION: torch.cat(video_inputs, dim=0),
        ModalityType.AUDIO: torch.cat(audio_inputs, dim=0),
    }
    with torch.inference_mode():
        embeddings = model(inputs)
    video = embeddings[ModalityType.VISION].detach().float().cpu().numpy()
    audio = embeddings[ModalityType.AUDIO].detach().float().cpu().numpy()
    return video, audio


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.index_csv.is_file():
        raise SystemExit(f"Index CSV does not exist: {args.index_csv}")

    pairs, ignored_rows = load_pairs(args.index_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Eligible pairs: {len(pairs)}")
    print(f"Rows ignored because status != success: {ignored_rows}")
    print("Loading ImageBind model...")

    if not args.checkpoint.is_file():
        raise SystemExit(f"ImageBind checkpoint does not exist: {args.checkpoint}")
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    model.to(device)

    video_batches: List[np.ndarray] = []
    audio_batches: List[np.ndarray] = []
    processed_rows: List[Dict[str, str]] = []
    skipped = 0
    video_cache_dir = args.output_dir / "video_cache"

    for batch in tqdm(chunks(pairs, args.batch_size), desc="ImageBind batches"):
        video_inputs, audio_inputs, valid_rows, load_skipped = prepare_batch(
            batch, device, video_cache_dir
        )
        skipped += load_skipped
        if not valid_rows:
            continue

        try:
            video_embedding, audio_embedding = infer_batch(
                model, video_inputs, audio_inputs
            )
        except Exception as error:
            skipped += len(valid_rows)
            warnings.warn(
                f"Skipping inference batch of {len(valid_rows)} pair(s): {error}",
                stacklevel=1,
            )
            continue

        video_batches.append(video_embedding)
        audio_batches.append(audio_embedding)
        processed_rows.extend(valid_rows)

    if not processed_rows:
        raise RuntimeError("No sample was processed successfully")

    video_embeddings = l2_normalize(np.concatenate(video_batches, axis=0))
    audio_embeddings = l2_normalize(np.concatenate(audio_batches, axis=0))
    similarity = video_embeddings @ audio_embeddings.T

    metrics = {
        "num_index_rows": int(len(pairs) + ignored_rows),
        "num_eligible_pairs": int(len(pairs)),
        "num_processed_pairs": int(len(processed_rows)),
        "num_skipped_pairs": int(skipped),
        "num_ignored_index_rows": int(ignored_rows),
        "video_to_audio": retrieval_metrics(similarity),
        "audio_to_video": retrieval_metrics(similarity.T),
    }

    np.save(args.output_dir / "video_embeddings.npy", video_embeddings)
    np.save(args.output_dir / "audio_embeddings.npy", audio_embeddings)
    np.save(args.output_dir / "similarity.npy", similarity)
    pd.DataFrame(processed_rows).to_csv(
        args.output_dir / "sample_index.csv", index=False
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
