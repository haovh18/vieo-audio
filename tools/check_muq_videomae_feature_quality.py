#!/usr/bin/env python3
"""Read-only quality checks for MuQ+VideoMAE NumPy feature matrices."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--layers", default="0,4,8,12")
    args = parser.parse_args()
    feature_dir = Path(args.feature_dir)
    if not feature_dir.is_absolute():
        feature_dir = ROOT / feature_dir
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    expected = {"video_final.npy": 768, "audio_final.npy": 1024}
    for layer in layers:
        expected[f"video_layer_{layer}.npy"] = 768
        expected[f"audio_layer_{layer}.npy"] = 1024

    files, blockers, first_dims = {}, [], []
    for name, width in expected.items():
        path = feature_dir / name
        entry = {"exists": path.is_file(), "expected_width": width}
        if not path.is_file():
            entry["status"] = "missing"
            blockers.append(f"missing {name}")
            files[name] = entry
            continue
        try:
            value = np.load(path, mmap_mode="r")
            norms = np.linalg.norm(value, axis=1) if value.ndim == 2 else np.array([])
            entry.update({
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "shape_valid": value.ndim == 2 and value.shape[1] == width,
                "all_finite": bool(np.isfinite(value).all()),
                "has_nan": bool(np.isnan(value).any()),
                "has_inf": bool(np.isinf(value).any()),
                "zero_vector_count": int((norms == 0).sum()),
                "all_l2_norms_positive": bool(len(norms) > 0 and (norms > 0).all()),
                "l2_norm_min": float(norms.min()) if len(norms) else None,
                "l2_norm_mean": float(norms.mean()) if len(norms) else None,
                "l2_norm_max": float(norms.max()) if len(norms) else None,
            })
            if value.ndim == 2:
                first_dims.append(value.shape[0])
            for check, message in (
                (entry["shape_valid"], f"invalid shape for {name}"),
                (entry["all_finite"], f"non-finite values in {name}"),
                (entry["all_l2_norms_positive"], f"zero/empty vectors in {name}"),
            ):
                if not check:
                    blockers.append(message)
        except Exception as error:
            entry.update(status="unreadable", error=repr(error))
            blockers.append(f"cannot read {name}: {error}")
        files[name] = entry

    metadata = {}
    n = first_dims[0] if first_dims else None
    dimensions_consistent = bool(first_dims) and len(set(first_dims)) == 1
    if not dimensions_consistent:
        blockers.append("feature matrices have inconsistent first dimensions")
    for name in ("sample_ids.json", "pair_paths.json"):
        path = feature_dir / name
        if not path.is_file():
            metadata[name] = {"exists": False}
            blockers.append(f"missing {name}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata[name] = {
                "exists": True, "count": len(data), "matches_feature_count": len(data) == n
            }
            if len(data) != n:
                blockers.append(f"{name} count does not match features")
        except Exception as error:
            metadata[name] = {"exists": True, "error": repr(error)}
            blockers.append(f"cannot read {name}")

    report = {
        "status": "pass" if not blockers else "fail",
        "feature_dir": str(feature_dir.relative_to(ROOT)),
        "selected_layers": layers,
        "feature_count": n,
        "all_first_dimensions_consistent": dimensions_consistent,
        "files": files,
        "metadata": metadata,
        "blockers": blockers,
        "can_proceed": not blockers,
    }
    (feature_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# B Method Feature Quality",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Feature directory: `{report['feature_dir']}`",
        f"- Samples: {n}",
        f"- First dimensions consistent: {dimensions_consistent}",
        f"- sample_ids/pair_paths consistent: {all(x.get('matches_feature_count') for x in metadata.values())}",
        f"- Can proceed: {report['can_proceed']}",
        "",
        "## Files",
        "",
    ]
    for name, item in files.items():
        lines.append(
            f"- `{name}`: shape `{item.get('shape')}`, finite={item.get('all_finite')}, "
            f"zero_vectors={item.get('zero_vector_count')}, "
            f"norm(min/mean/max)="
            f"{item.get('l2_norm_min')}/{item.get('l2_norm_mean')}/{item.get('l2_norm_max')}"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {x}" for x in blockers] or ["- None"])
    (feature_dir / "QUALITY_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first100_result = feature_dir / "FEATURE_FIRST100_RESULT.md"
    if first100_result.is_file():
        original = first100_result.read_text(encoding="utf-8")
        marker = "\n## Independent quality check\n"
        original = original.split(marker, 1)[0].rstrip()
        first100_result.write_text(
            original + marker
            + f"\n- Quality status: **{report['status'].upper()}**\n"
            + f"- Can enter retrieval baseline: {report['can_proceed']}\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
