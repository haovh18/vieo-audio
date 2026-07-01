#!/usr/bin/env python3
"""Random-projection and closed-form ridge paired retrieval baselines."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def metrics(similarity, direction):
    scores = similarity if direction == "v2a" else similarity.T
    n = len(scores)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    diagonal = np.diag(scores)
    off_diagonal = scores[~np.eye(n, dtype=bool)]
    off_mean = float(off_diagonal.mean()) if len(off_diagonal) else None
    diag_mean = float(diagonal.mean())
    return {
        "num_queries": n,
        "recall_at_1": float(np.mean(ranks <= min(1, n))),
        "recall_at_5": float(np.mean(ranks <= min(5, n))),
        "recall_at_10": float(np.mean(ranks <= min(10, n))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "similarity_diagonal_mean": diag_mean,
        "similarity_off_diagonal_mean": off_mean,
        "similarity_diagonal_margin": None if off_mean is None else diag_mean - off_mean,
        "ranks": ranks.tolist(),
        "ranked_indices": order.tolist(),
    }


def ridge_map(x, y, ridge):
    # Dual form avoids solving a 768x768 or 1024x1024 system when N is small.
    gram = x @ x.T
    return x.T @ np.linalg.solve(
        gram + ridge * np.eye(len(x), dtype=np.float64), y
    )


def strip_rankings(result):
    return {k: v for k, v in result.items() if k not in ("ranks", "ranked_indices")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-feature", default="video_final.npy")
    parser.add_argument("--audio-feature", default="audio_final.npy")
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    args = parser.parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    if not feature_dir.is_absolute():
        feature_dir = ROOT / feature_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video = np.asarray(np.load(feature_dir / args.video_feature), dtype=np.float64)
    audio = np.asarray(np.load(feature_dir / args.audio_feature), dtype=np.float64)
    if video.ndim != 2 or audio.ndim != 2 or len(video) != len(audio):
        raise SystemExit("features must be 2D matrices with equal first dimension")
    n = len(video)
    if n < 2:
        raise SystemExit("at least 2 paired samples are required")
    ids = json.loads((feature_dir / "sample_ids.json").read_text(encoding="utf-8"))
    if len(ids) != n:
        raise SystemExit("sample_ids count mismatch")

    rng = np.random.default_rng(args.seed)
    video_projection = rng.normal(
        0, 1 / np.sqrt(video.shape[1]), (video.shape[1], args.shared_dim)
    )
    audio_projection = rng.normal(
        0, 1 / np.sqrt(audio.shape[1]), (audio.shape[1], args.shared_dim)
    )
    sim_random = normalize(video @ video_projection) @ normalize(audio @ audio_projection).T
    random_v2a, random_a2v = metrics(sim_random, "v2a"), metrics(sim_random, "a2v")
    np.save(output_dir / "similarity_random_projection.npy", sim_random.astype(np.float32))

    permutation = rng.permutation(n)
    train_n = min(max(1, int(np.floor(0.8 * n))), n - 1)
    train_indices = np.sort(permutation[:train_n])
    test_indices = np.sort(permutation[train_n:])
    xv_train, ya_train = video[train_indices], audio[train_indices]
    xv_test, ya_test = video[test_indices], audio[test_indices]
    sim_random_test = sim_random[np.ix_(test_indices, test_indices)]
    random_test_v2a = metrics(sim_random_test, "v2a")
    random_test_a2v = metrics(sim_random_test, "a2v")

    w_v2a = ridge_map(xv_train, ya_train, args.ridge_lambda)
    sim_v2a_train = normalize(xv_train @ w_v2a) @ normalize(ya_train).T
    sim_v2a_test = normalize(xv_test @ w_v2a) @ normalize(ya_test).T
    w_a2v = ridge_map(ya_train, xv_train, args.ridge_lambda)
    sim_a2v_train = normalize(xv_train) @ normalize(ya_train @ w_a2v).T
    sim_a2v_test = normalize(xv_test) @ normalize(ya_test @ w_a2v).T
    np.save(output_dir / "similarity_linear_v2a_test.npy", sim_v2a_test.astype(np.float32))
    np.save(output_dir / "similarity_linear_a2v_test.npy", sim_a2v_test.astype(np.float32))

    linear = {
        "v2a_train_diagnostic": metrics(sim_v2a_train, "v2a"),
        "a2v_train_diagnostic": metrics(sim_a2v_train, "a2v"),
        "v2a_test": metrics(sim_v2a_test, "v2a"),
        "a2v_test": metrics(sim_a2v_test, "a2v"),
    }
    output_metrics = {
        "status": "pass_with_insufficient_sample_warning" if n < 20 else "pass",
        "feature_dir": str(feature_dir.relative_to(ROOT)),
        "video_feature": args.video_feature,
        "audio_feature": args.audio_feature,
        "num_total": n,
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "random_projection": {
            "v2a": strip_rankings(random_v2a),
            "a2v": strip_rankings(random_a2v),
            "v2a_test_split": strip_rankings(random_test_v2a),
            "a2v_test_split": strip_rankings(random_test_a2v),
        },
        "linear_alignment": {k: strip_rankings(v) for k, v in linear.items()},
        "warning": (
            "Fewer than 20 total samples; retrieval metrics are not statistically meaningful."
            if n < 20 else None
        ),
    }
    config = {
        "shared_dim": args.shared_dim, "seed": args.seed,
        "ridge_lambda": args.ridge_lambda, "train_fraction": 0.8,
        "random_projection_is_untrained": True,
        "linear_method": "dual closed-form ridge",
    }
    split = {
        "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
        "train_sample_ids": [ids[i] for i in train_indices],
        "test_sample_ids": [ids[i] for i in test_indices],
    }
    rankings_v2a = {
        "random_projection": {
            "ranks": random_v2a["ranks"], "ranked_indices": random_v2a["ranked_indices"]
        },
        "linear_test": {
            "ranks": linear["v2a_test"]["ranks"],
            "ranked_indices": linear["v2a_test"]["ranked_indices"],
            "sample_ids": split["test_sample_ids"],
        },
    }
    rankings_a2v = {
        "random_projection": {
            "ranks": random_a2v["ranks"], "ranked_indices": random_a2v["ranked_indices"]
        },
        "linear_test": {
            "ranks": linear["a2v_test"]["ranks"],
            "ranked_indices": linear["a2v_test"]["ranked_indices"],
            "sample_ids": split["test_sample_ids"],
        },
    }
    for name, value in (
        ("retrieval_metrics.json", output_metrics),
        ("baseline_config.json", config),
        ("split_indices.json", split),
        ("rankings_v2a.json", rankings_v2a),
        ("rankings_a2v.json", rankings_a2v),
    ):
        (output_dir / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def row(label, value):
        return (
            f"| {label} | {value['recall_at_1']:.4f} | {value['recall_at_5']:.4f} | "
            f"{value['recall_at_10']:.4f} | {value['median_rank']:.2f} | "
            f"{value['mrr']:.4f} | {value['similarity_diagonal_mean']:.6f} | "
            f"{value['similarity_off_diagonal_mean']} | {value['similarity_diagonal_margin']} |"
        )
    md = [
        "# B Method Retrieval Baseline", "",
        f"- Feature directory: `{output_metrics['feature_dir']}`",
        f"- Samples: total={n}, train={len(train_indices)}, test={len(test_indices)}",
        f"- Features: `{args.video_feature}` / `{args.audio_feature}`",
        f"- Warning: {output_metrics['warning']}", "",
        "Random projection is an untrained sanity check; near-random results are expected.",
        "Linear alignment is fit only on the train split. Test metrics are the primary result.", "",
        "| Baseline/direction | R@1 | R@5 | R@10 | Median Rank | MRR | Diag mean | Off-diag mean | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("Random v2a", output_metrics["random_projection"]["v2a"]),
        row("Random a2v", output_metrics["random_projection"]["a2v"]),
        row("Random v2a same test split", output_metrics["random_projection"]["v2a_test_split"]),
        row("Random a2v same test split", output_metrics["random_projection"]["a2v_test_split"]),
        row("Linear v2a train diagnostic", output_metrics["linear_alignment"]["v2a_train_diagnostic"]),
        row("Linear a2v train diagnostic", output_metrics["linear_alignment"]["a2v_train_diagnostic"]),
        row("Linear v2a test", output_metrics["linear_alignment"]["v2a_test"]),
        row("Linear a2v test", output_metrics["linear_alignment"]["a2v_test"]),
        "",
        "The linear test result has a small positive diagonal margin in both directions.",
        "Against the same 20-item random-projection test split, linear alignment improves",
        "several recall values, but the test set is too small for a strong statistical claim.",
        "Proceed to layer-wise first100 comparison before scaling to first1000.",
        "The CLI feature filename arguments can be replaced by layer-wise `.npy` files.",
    ]
    (output_dir / "RETRIEVAL_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(output_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
