#!/usr/bin/env python
"""
Aggregate CLIP embeddings per title and export title-level stats + relations.

Earth Mover's Distance (Wasserstein-2) is approximated by treating each title's
embeddings as a diagonal Gaussian distribution. Pairwise distances between
titles are reported using this metric.

Outputs (default path: TK/clustering):
  - title_embeddings.npy                : stacked mean embedding per title
  - title_embedding_variances.npy       : stacked diagonal variances per title
  - title_embeddings_meta.json          : order of titles + sample counts
  - title_similarity_topk.csv           : Wasserstein-based neighbor list
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build title-level embedding vectors and similarity outputs."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("TK/clustering/clip_embeddings.npy"),
        help="Path to sample-level CLIP embeddings (.npy).",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("TK/clustering/cluster_assignments.csv"),
        help="CSV containing file-level metadata (must align with embeddings order).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TK/clustering"),
    )
    parser.add_argument(
        "--similarity-topk",
        type=int,
        default=10,
        help="Number of nearest titles to report per title.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize sample-level embeddings before computing stats.",
    )
    return parser.parse_args()


def load_titles(assignments_csv: Path):
    titles = []
    with assignments_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(row["title"])
    return titles


def normalize_rows(matrix: np.ndarray):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_title_stats(embeddings, titles):
    groups = defaultdict(list)
    for idx, title in enumerate(titles):
        groups[title].append(idx)

    ordered = []
    counts = []
    mean_vectors = []
    var_vectors = []
    for title in sorted(groups.keys()):
        idxs = groups[title]
        subset = embeddings[idxs]
        mean_vec = subset.mean(axis=0)
        if len(idxs) > 1:
            var_vec = subset.var(axis=0, ddof=1)
        else:
            var_vec = np.zeros_like(mean_vec)
        ordered.append(title)
        counts.append(len(idxs))
        mean_vectors.append(mean_vec)
        var_vectors.append(var_vec)
    mean_matrix = np.vstack(mean_vectors)
    var_matrix = np.vstack(var_vectors)
    return ordered, counts, mean_matrix, var_matrix


def wasserstein2_distance(mean_a, var_a, mean_b, var_b):
    mean_term = np.sum((mean_a - mean_b) ** 2)
    std_diff = np.sqrt(np.clip(var_a, 0, None)) - np.sqrt(np.clip(var_b, 0, None))
    cov_term = np.sum(std_diff ** 2)
    dist_sq = mean_term + cov_term
    return float(np.sqrt(max(dist_sq, 0.0)))


def build_distance_matrix(mean_matrix, var_matrix):
    n = mean_matrix.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein2_distance(mean_matrix[i], var_matrix[i], mean_matrix[j], var_matrix[j])
            dist[i, j] = dist[j, i] = d
    return dist


def write_topk_csv(path, titles, distance_matrix, topk):
    n = len(titles)
    topk = min(max(topk, 1), n - 1)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "neighbor_title", "rank", "wasserstein2_distance"])
        for idx, title in enumerate(titles):
            distances = distance_matrix[idx].copy()
            distances[idx] = np.inf
            best = np.argpartition(distances, topk)[:topk]
            ranked = sorted(best, key=lambda j: distances[j])
            for rank, neighbor_idx in enumerate(ranked, start=1):
                writer.writerow(
                    [
                        title,
                        titles[neighbor_idx],
                        rank,
                        float(distances[neighbor_idx]),
                    ]
                )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    if args.normalize:
        embeddings = normalize_rows(embeddings)
    titles = load_titles(args.assignments)
    if len(embeddings) != len(titles):
        raise SystemExit(
            f"Embeddings count ({len(embeddings)}) and CSV rows ({len(titles)}) mismatch."
        )

    ordered_titles, counts, mean_matrix, var_matrix = build_title_stats(embeddings, titles)

    title_embeddings_path = args.output_dir / "title_embeddings.npy"
    np.save(title_embeddings_path, mean_matrix)
    title_variances_path = args.output_dir / "title_embedding_variances.npy"
    np.save(title_variances_path, var_matrix)

    meta = {
        "titles": ordered_titles,
        "counts": counts,
        "normalize": bool(args.normalize),
        "distance_metric": "wasserstein-2 (diagonal gaussian approximation)",
    }
    meta_path = args.output_dir / "title_embeddings_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    distance_matrix = build_distance_matrix(mean_matrix, var_matrix)
    topk_csv = args.output_dir / "title_similarity_topk.csv"
    write_topk_csv(topk_csv, ordered_titles, distance_matrix, args.similarity_topk)

    print(
        f"Wrote {title_embeddings_path} ({mean_matrix.shape[0]} titles, dim={mean_matrix.shape[1]})"
    )
    print(f"Wrote {title_variances_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {topk_csv}")


if __name__ == "__main__":
    main()
