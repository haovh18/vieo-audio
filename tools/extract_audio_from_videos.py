#!/usr/bin/env python3
"""Extract mono 24 kHz WAV audio from local video files."""

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively extract audio tracks from local videos."
    )
    parser.add_argument("--input-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        default=Path("datasets/HarmonySet/processed/audio_wav"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("datasets/HarmonySet/processed/audio_video_index.csv"),
    )
    return parser.parse_args()


def find_videos(input_root: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.as_posix(),
    )


def sanitized_part(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return cleaned.strip("_") or "video"


def candidate_names(video: Path, input_root: Path) -> Iterable[str]:
    stem = sanitized_part(video.stem)
    yield f"{stem}.wav"

    try:
        relative_parent = video.relative_to(input_root).parent
        parent_parts = relative_parent.parts
    except ValueError:
        parent_parts = video.parent.parts

    for depth in range(1, len(parent_parts) + 1):
        prefix = "_".join(sanitized_part(part) for part in parent_parts[-depth:])
        yield f"{prefix}_{stem}.wav"


def assign_output_paths(
    videos: List[Path], input_root: Path, output_dir: Path
) -> Dict[Path, Path]:
    stem_counts = Counter(video.stem for video in videos)
    used_names = set()
    assignments: Dict[Path, Path] = {}

    for video in videos:
        candidates = candidate_names(video, input_root)
        name = next(candidates)

        if stem_counts[video.stem] > 1 or name in used_names:
            for candidate in candidates:
                if candidate not in used_names:
                    name = candidate
                    break
            else:
                index = 2
                base = sanitized_part(video.stem)
                while f"{base}_{index}.wav" in used_names:
                    index += 1
                name = f"{base}_{index}.wav"

        used_names.add(name)
        assignments[video] = output_dir / name

    return assignments


def extract_audio(video: Path, audio: Path) -> bool:
    temporary_audio = audio.with_name(f".{audio.stem}.tmp.wav")
    temporary_audio.unlink(missing_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        str(temporary_audio),
    ]

    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not temporary_audio.is_file():
            temporary_audio.unlink(missing_ok=True)
            return False
        temporary_audio.replace(audio)
        return True
    except OSError as error:
        print(f"ffmpeg execution failed for {video}: {error}")
        temporary_audio.unlink(missing_ok=True)
        return False


def write_index(index_csv: Path, rows: List[Tuple[str, str, str]]) -> None:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "audio_path", "status"])
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    output_dir = args.output_audio_dir

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_videos(input_root)
    output_paths = assign_output_paths(videos, input_root, output_dir)

    rows: List[Tuple[str, str, str]] = []
    counts = Counter()

    for position, video in enumerate(videos, start=1):
        audio = output_paths[video]
        if audio.exists():
            status = "skipped"
        elif extract_audio(video, audio):
            status = "success"
        else:
            status = "failed"

        counts[status] += 1
        rows.append((video.as_posix(), audio.as_posix(), status))
        print(f"[{position}/{len(videos)}] {status}: {video}")

    write_index(args.index_csv, rows)

    print()
    print(f"Videos found: {len(videos)}")
    print(f"Successfully extracted: {counts['success']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Failed: {counts['failed']}")
    print(f"Audio output directory: {output_dir}")
    print(f"Index CSV: {args.index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
