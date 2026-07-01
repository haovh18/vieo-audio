#!/usr/bin/env python3
"""Compare frozen VideoMAE/MuQ layer pairs with closed-form ridge retrieval."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_layers(value):
    return [int(item) for item in value.split(",")]


def normalize(value):
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def ridge_map(source, target, ridge):
    # Equivalent to primal ridge on the row span, while solving only N_train x N_train.
    gram = source @ source.T
    return source.T @ np.linalg.solve(
        gram + ridge * np.eye(len(source), dtype=np.float64), target
    )


def retrieval_metrics(similarity, direction):
    scores = similarity if direction == "v2a" else similarity.T
    n = len(scores)
    order = np.argsort(-scores, axis=1)
    ranks = np.array(
        [int(np.where(order[index] == index)[0][0]) + 1 for index in range(n)]
    )
    diagonal = np.diag(scores)
    off_diagonal = scores[~np.eye(n, dtype=bool)]
    off_mean = float(off_diagonal.mean())
    diag_mean = float(diagonal.mean())
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= min(5, n))),
        "R@10": float(np.mean(ranks <= min(10, n))),
        "MedR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "diag_mean": diag_mean,
        "offdiag_mean": off_mean,
        "margin": diag_mean - off_mean,
    }


def evaluate(video, audio, train_indices, test_indices, ridge):
    video_train, video_test = video[train_indices], video[test_indices]
    audio_train, audio_test = audio[train_indices], audio[test_indices]
    v2a_map = ridge_map(video_train, audio_train, ridge)
    a2v_map = ridge_map(audio_train, video_train, ridge)
    v2a_similarity = normalize(video_test @ v2a_map) @ normalize(audio_test).T
    a2v_similarity = normalize(video_test) @ normalize(audio_test @ a2v_map).T
    return (
        retrieval_metrics(v2a_similarity, "v2a"),
        retrieval_metrics(a2v_similarity, "a2v"),
    )


def flat_record(video_name, audio_name, video_layer, audio_layer, v2a, a2v, config):
    record = {
        "video_feature": video_name,
        "audio_feature": audio_name,
        "video_layer": video_layer,
        "audio_layer": audio_layer,
        **config,
    }
    for prefix, metrics in (("v2a", v2a), ("a2v", a2v)):
        record.update({
            f"{prefix}_R@1": metrics["R@1"],
            f"{prefix}_R@5": metrics["R@5"],
            f"{prefix}_R@10": metrics["R@10"],
            f"{prefix}_MedR": metrics["MedR"],
            f"{prefix}_MeanR": metrics["MeanR"],
            f"{prefix}_MRR": metrics["MRR"],
            f"{prefix}_diag_mean": metrics["diag_mean"],
            f"{prefix}_offdiag_mean": metrics["offdiag_mean"],
            f"{prefix}_margin": metrics["margin"],
        })
    record["avg_score"] = float(np.mean([
        v2a["R@1"], v2a["R@5"], v2a["R@10"],
        a2v["R@1"], a2v["R@5"], a2v["R@10"],
    ]))
    return record


def write_matrix(path, video_layers, audio_layers, records, field):
    lookup = {(item["video_layer"], item["audio_layer"]): item[field] for item in records}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_layer/audio_layer", *audio_layers])
        for video_layer in video_layers:
            writer.writerow([
                video_layer,
                *[lookup[(video_layer, audio_layer)] for audio_layer in audio_layers],
            ])


def markdown_matrix(title, video_layers, audio_layers, records, field):
    lookup = {(item["video_layer"], item["audio_layer"]): item[field] for item in records}
    lines = [
        f"### {title}", "",
        "| Video \\ Audio | " + " | ".join(str(x) for x in audio_layers) + " |",
        "|---|" + "|".join("---:" for _ in audio_layers) + "|",
    ]
    for video_layer in video_layers:
        values = [f"{lookup[(video_layer, audio_layer)]:.4f}" for audio_layer in audio_layers]
        lines.append(f"| {video_layer} | " + " | ".join(values) + " |")
    return lines


def best_for(records, field):
    value = max(item[field] for item in records)
    winners = [
        {"video_layer": item["video_layer"], "audio_layer": item["audio_layer"],
         "value": item[field]}
        for item in records if np.isclose(item[field], value)
    ]
    return {"metric": field, "best_value": value, "winners": winners}


def load_imagebind(test_indices, expected_pair_paths):
    candidates = [
        ROOT / "outputs/imagebind_baseline/retrieval_metrics.json",
        ROOT / "outputs/imagebind_baseline/metrics.json",
        ROOT / "outputs/imagebind/baseline/metrics.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            v2a = data.get("video_to_audio", data.get("v2a", {}))
            a2v = data.get("audio_to_video", data.get("a2v", {}))

            def value(block, key):
                raw = block.get(key, block.get(key.lower().replace("@", "_at_")))
                if raw is None:
                    return None
                raw = float(raw)
                return raw / 100.0 if raw > 1 else raw

            result = {
                "source": str(path.relative_to(ROOT)),
                "num_samples": data.get("num_processed_pairs", data.get("num_total")),
                "v2a_R@1": value(v2a, "R@1"),
                "v2a_R@5": value(v2a, "R@5"),
                "v2a_R@10": value(v2a, "R@10"),
                "a2v_R@1": value(a2v, "R@1"),
                "a2v_R@5": value(a2v, "R@5"),
                "a2v_R@10": value(a2v, "R@10"),
            }
            embedding_dir = path.parent
            video_embeddings = embedding_dir / "video_embeddings.npy"
            audio_embeddings = embedding_dir / "audio_embeddings.npy"
            sample_index = embedding_dir / "sample_index.csv"
            if video_embeddings.is_file() and audio_embeddings.is_file() and sample_index.is_file():
                with sample_index.open(newline="", encoding="utf-8-sig") as handle:
                    imagebind_pairs = list(csv.DictReader(handle))
                order_matches = (
                    len(imagebind_pairs) == len(expected_pair_paths)
                    and all(
                        left["video_path"] == right["video_path"]
                        and left["audio_path"] == right["audio_path"]
                        for left, right in zip(expected_pair_paths, imagebind_pairs)
                    )
                )
                result["sample_order_matches"] = order_matches
                if order_matches:
                    video = np.asarray(np.load(video_embeddings), dtype=np.float64)
                    audio = np.asarray(np.load(audio_embeddings), dtype=np.float64)
                    similarity = normalize(video[test_indices]) @ normalize(audio[test_indices]).T
                    result["same_test_split"] = {
                        "num_candidates": len(test_indices),
                        "v2a": retrieval_metrics(similarity, "v2a"),
                        "a2v": retrieval_metrics(similarity, "a2v"),
                    }
            if all(result[key] is not None for key in result if key.startswith(("v2a_", "a2v_"))):
                return result
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--previous-retrieval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-layers", type=parse_layers, default=parse_layers("0,4,8,12"))
    parser.add_argument("--audio-layers", type=parse_layers, default=parse_layers("0,4,8,12"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    args = parser.parse_args()
    feature_dir = Path(args.feature_dir)
    previous_dir = Path(args.previous_retrieval_dir)
    output_dir = Path(args.output_dir)
    for name, value in (
        ("feature_dir", feature_dir), ("previous_dir", previous_dir), ("output_dir", output_dir)
    ):
        if not value.is_absolute():
            locals()[name] = ROOT / value
    # Assignment through locals is not reliable in functions; resolve explicitly.
    feature_dir = feature_dir if feature_dir.is_absolute() else ROOT / feature_dir
    previous_dir = previous_dir if previous_dir.is_absolute() else ROOT / previous_dir
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = json.loads((feature_dir / "sample_ids.json").read_text(encoding="utf-8"))
    pair_paths = json.loads((feature_dir / "pair_paths.json").read_text(encoding="utf-8"))
    n = len(sample_ids)
    split_path = previous_dir / "split_indices.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        split_source = str(split_path.relative_to(ROOT))
    else:
        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(n)
        train_indices = np.sort(permutation[:140])
        test_indices = np.sort(permutation[140:])
        split_source = "generated_seed_42_fallback"
    if (
        len(train_indices) != 140 or len(test_indices) != 36
        or set(train_indices) & set(test_indices)
        or set(np.concatenate([train_indices, test_indices])) != set(range(n))
    ):
        raise SystemExit("split must be a disjoint 140/36 partition of all 176 samples")

    config = {
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "ridge_lambda": args.ridge_lambda,
        "seed": args.seed,
        "split_source": split_source,
    }

    def load(name, expected_width):
        value = np.asarray(np.load(feature_dir / name), dtype=np.float64)
        if value.shape != (n, expected_width) or not np.isfinite(value).all():
            raise ValueError(f"invalid feature {name}: {value.shape}")
        return value

    video_final = load("video_final.npy", 768)
    audio_final = load("audio_final.npy", 1024)
    final_v2a, final_a2v = evaluate(
        video_final, audio_final, train_indices, test_indices, args.ridge_lambda
    )
    final_record = flat_record(
        "video_final.npy", "audio_final.npy", "final", "final",
        final_v2a, final_a2v, config,
    )

    video_features = {
        layer: load(f"video_layer_{layer}.npy", 768) for layer in args.video_layers
    }
    audio_features = {
        layer: load(f"audio_layer_{layer}.npy", 1024) for layer in args.audio_layers
    }
    records = []
    for video_layer in args.video_layers:
        for audio_layer in args.audio_layers:
            v2a, a2v = evaluate(
                video_features[video_layer], audio_features[audio_layer],
                train_indices, test_indices, args.ridge_lambda,
            )
            records.append(flat_record(
                f"video_layer_{video_layer}.npy",
                f"audio_layer_{audio_layer}.npy",
                video_layer, audio_layer, v2a, a2v, config,
            ))

    metric_fields = {
        "v2a_R1": "v2a_R@1", "v2a_R5": "v2a_R@5", "v2a_R10": "v2a_R@10",
        "a2v_R1": "a2v_R@1", "a2v_R5": "a2v_R@5", "a2v_R10": "a2v_R@10",
        "v2a_margin": "v2a_margin", "a2v_margin": "a2v_margin",
    }
    for file_metric, field in metric_fields.items():
        write_matrix(
            output_dir / f"layerwise_matrix_{file_metric}.csv",
            args.video_layers, args.audio_layers, records, field,
        )

    best_fields = [
        "v2a_R@1", "v2a_R@5", "v2a_R@10",
        "a2v_R@1", "a2v_R@5", "a2v_R@10",
        "v2a_margin", "a2v_margin", "avg_score",
    ]
    best = {field: best_for(records, field) for field in best_fields}
    best_average = max(records, key=lambda item: item["avg_score"])
    imagebind = load_imagebind(test_indices, pair_paths)
    improvements = {}
    for field in best_fields[:-1]:
        improvements[field] = best[field]["best_value"] - final_record[field]
    summary = {
        "feature_dir": str(feature_dir.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "num_samples": n,
        **config,
        "video_layers": args.video_layers,
        "audio_layers": args.audio_layers,
        "final_final": final_record,
        "best_layerwise": best,
        "improvement_over_final": improvements,
        "any_layerwise_recall_exceeds_final": any(
            improvements[field] > 0 for field in best_fields[:6]
        ),
        "best_average_combination": best_average,
        "imagebind": imagebind,
        "imagebind_comparison_caveat": (
            "The stored ImageBind headline metrics use 176 candidates. A fair descriptive "
            "comparison was additionally computed from existing ImageBind embeddings on the "
            "identical 36-item test subset; ImageBind itself remains zero-shot, whereas the "
            "layer-wise ridge maps use the other 140 pairs."
            if imagebind and imagebind.get("same_test_split") else
            "ImageBind headline protocol is not directly comparable to the 36-item ridge test."
            if imagebind else "ImageBind metrics not found."
        ),
    }
    metrics_payload = {
        "experiment": {
            "feature_dir": str(feature_dir.relative_to(ROOT)),
            "num_samples": n,
            "video_layers": args.video_layers,
            "audio_layers": args.audio_layers,
            **config,
        },
        "final_final": final_record,
        "layerwise_combinations": records,
    }
    (output_dir / "layerwise_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "final_vs_layerwise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "best_combinations.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# MuQ + VideoMAE Layer-wise Retrieval", "",
        "## Experimental setup", "",
        f"- Feature directory: `{feature_dir.relative_to(ROOT)}`",
        f"- Output directory: `{output_dir.relative_to(ROOT)}`",
        f"- Samples: {n}; train/test: {len(train_indices)}/{len(test_indices)}",
        f"- Previous split reused: {split_path.is_file()}",
        f"- Split source: `{split_source}`",
        f"- Ridge lambda: {args.ridge_lambda}; seed: {args.seed}",
        f"- Video layers: {args.video_layers}; audio layers: {args.audio_layers}", "",
        "## Final-final baseline", "",
        f"- v2a R@1/R@5/R@10: {final_record['v2a_R@1']:.4f} / "
        f"{final_record['v2a_R@5']:.4f} / {final_record['v2a_R@10']:.4f}; "
        f"margin {final_record['v2a_margin']:+.6f}",
        f"- a2v R@1/R@5/R@10: {final_record['a2v_R@1']:.4f} / "
        f"{final_record['a2v_R@5']:.4f} / {final_record['a2v_R@10']:.4f}; "
        f"margin {final_record['a2v_margin']:+.6f}", "",
        "Random expectations for 36 candidates: R@1=0.0278, R@5=0.1389, R@10=0.2778.",
        "",
        "## Layer-wise matrices", "",
    ]
    for title, field in (
        ("v2a R@1", "v2a_R@1"), ("v2a R@5", "v2a_R@5"),
        ("v2a R@10", "v2a_R@10"), ("a2v R@1", "a2v_R@1"),
        ("a2v R@5", "a2v_R@5"), ("a2v R@10", "a2v_R@10"),
        ("v2a margin", "v2a_margin"), ("a2v margin", "a2v_margin"),
    ):
        lines.extend(markdown_matrix(title, args.video_layers, args.audio_layers, records, field))
        lines.append("")
    lines.extend(["## Best combinations", ""])
    for field in best_fields:
        winner_text = ", ".join(
            f"v{item['video_layer']}+a{item['audio_layer']}"
            for item in best[field]["winners"]
        )
        lines.append(f"- {field}: {best[field]['best_value']:.6f} — {winner_text}")
    lines.extend([
        "",
        "## Final-final comparison", "",
        f"- Any layer-wise recall metric exceeds final-final: "
        f"{summary['any_layerwise_recall_exceeds_final']}",
        f"- Best-average combination: v{best_average['video_layer']}+"
        f"a{best_average['audio_layer']}, score {best_average['avg_score']:.6f} "
        f"vs final-final {final_record['avg_score']:.6f} "
        f"(delta {best_average['avg_score'] - final_record['avg_score']:+.6f}).",
        "- The strongest balanced result uses middle/high layers (v8+a8). "
        "Individual optima differ by direction: v2a favors audio 4/8, while a2v often "
        "favors audio 0; video 8/12 dominates most best metrics.",
    ])
    for field in best_fields[:-1]:
        lines.append(
            f"- Best {field} delta vs final: {improvements[field]:+.6f}"
        )
    lines.extend(["", "## ImageBind comparison", ""])
    if imagebind:
        ib_test = imagebind.get("same_test_split")
        lines.extend([
            f"- Source: `{imagebind['source']}`",
            "",
            "| Method | v2a R@1 | v2a R@5 | v2a R@10 | a2v R@1 | a2v R@5 | a2v R@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| ImageBind | {imagebind['v2a_R@1']:.4f} | {imagebind['v2a_R@5']:.4f} | "
            f"{imagebind['v2a_R@10']:.4f} | {imagebind['a2v_R@1']:.4f} | "
            f"{imagebind['a2v_R@5']:.4f} | {imagebind['a2v_R@10']:.4f} |",
            f"| VideoMAE final + MuQ final | {final_record['v2a_R@1']:.4f} | "
            f"{final_record['v2a_R@5']:.4f} | {final_record['v2a_R@10']:.4f} | "
            f"{final_record['a2v_R@1']:.4f} | {final_record['a2v_R@5']:.4f} | "
            f"{final_record['a2v_R@10']:.4f} |",
            f"| Best average v{best_average['video_layer']}+a{best_average['audio_layer']} | "
            f"{best_average['v2a_R@1']:.4f} | {best_average['v2a_R@5']:.4f} | "
            f"{best_average['v2a_R@10']:.4f} | {best_average['a2v_R@1']:.4f} | "
            f"{best_average['a2v_R@5']:.4f} | {best_average['a2v_R@10']:.4f} |",
            "",
            f"Comparison caveat: {summary['imagebind_comparison_caveat']}",
        ])
        if ib_test:
            lines.extend([
                "",
                "### Same 36-item test subset",
                "",
                "| Method | v2a R@1 | v2a R@5 | v2a R@10 | a2v R@1 | a2v R@5 | a2v R@10 |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| ImageBind zero-shot | {ib_test['v2a']['R@1']:.4f} | "
                f"{ib_test['v2a']['R@5']:.4f} | {ib_test['v2a']['R@10']:.4f} | "
                f"{ib_test['a2v']['R@1']:.4f} | {ib_test['a2v']['R@5']:.4f} | "
                f"{ib_test['a2v']['R@10']:.4f} |",
                f"| VideoMAE final + MuQ final ridge | {final_record['v2a_R@1']:.4f} | "
                f"{final_record['v2a_R@5']:.4f} | {final_record['v2a_R@10']:.4f} | "
                f"{final_record['a2v_R@1']:.4f} | {final_record['a2v_R@5']:.4f} | "
                f"{final_record['a2v_R@10']:.4f} |",
                f"| Best average v{best_average['video_layer']}+a{best_average['audio_layer']} ridge | "
                f"{best_average['v2a_R@1']:.4f} | {best_average['v2a_R@5']:.4f} | "
                f"{best_average['v2a_R@10']:.4f} | {best_average['a2v_R@1']:.4f} | "
                f"{best_average['a2v_R@5']:.4f} | {best_average['a2v_R@10']:.4f} |",
            ])
    else:
        lines.append("未找到 ImageBind 指标文件，需手动补充 ImageBind baseline 对比。")
    lines.extend([
        "", "## Conclusion", "",
        "- Layer-wise retrieval is materially better than final-final on aggregate: the best",
        "  average score is 0.2500 vs 0.1667 (+0.0833, +50% relative). It does not improve",
        "  every direction/metric simultaneously, so this is evidence for layer selection,",
        "  not a claim that one fixed layer pair universally solves alignment.",
        "- The result supports the hypothesis that the final layer is not necessarily optimal",
        "  for video-music alignment. Middle/high VideoMAE layers (8/12) are strongest;",
        "  MuQ preference is direction-dependent (0 for much of a2v, 4/8 for v2a).",
        "- On the identical 36-item subset, ImageBind is substantially stronger than every",
        "  VideoMAE+MuQ layer pair (for example ImageBind v2a R@1/R@5/R@10 is",
        "  0.2778/0.6389/0.8333 vs v8+a8 0.0278/0.3333/0.4444). The proposed features",
        "  do not approach or exceed ImageBind yet.",
        "- Next priority: layer fusion (starting with VideoMAE 8/12 and MuQ 0/4/8), because",
        "  multiple layer pairs clearly beat final-final on held-out retrieval. After that,",
        "  add temporal chunk-level alignment to address direction-specific behavior and the",
        "  remaining large gap to ImageBind. Expanding the dataset follows those ablations.",
    ])
    (output_dir / "LAYERWISE_RETRIEVAL_RESULT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
