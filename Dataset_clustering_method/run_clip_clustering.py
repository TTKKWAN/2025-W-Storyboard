#!/usr/bin/env python
"""
Generate CLIP vision embeddings for TK/final_dataset2_trigger images and run clustering.
Outputs (inside TK/clustering):
  - clip_embeddings.npy : numpy array of shape (N, D)
  - cluster_assignments.csv : file-wise metadata + clustering labels
  - cluster_summary.txt : simple aggregate counts per algorithm
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel


SHOT_PATTERN = re.compile(r"<([^>]+)>")


def build_title_lookup(label_root: Path) -> dict:
    """Scan labeling JSONs and build source filename -> title mapping."""
    lookup = {}
    json_paths = list(label_root.rglob("*.json"))
    for path in tqdm(json_paths, desc="Loading label titles"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        source_path = (
            data.get("meta", {})
            .get("dataset", {})
            .get("source_path")
        )
        title = (
            data.get("meta", {})
            .get("product", {})
            .get("title")
        )
        if not source_path or not title:
            continue
        lookup[Path(source_path).name] = title.strip()
    return lookup


def extract_shot_tag(text: str) -> str:
    match = SHOT_PATTERN.search(text or "")
    return match.group(1) if match else "unknown"


def load_samples(meta_path: Path, image_root: Path, title_lookup: dict):
    """Return metadata entries that have both title + existing image file."""
    samples = []
    missing_title = []
    missing_image = []
    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fname = record["file_name"]
            title = title_lookup.get(fname)
            image_path = image_root / fname
            if not title and "_1024sq_hint" in fname:
                base_name = fname.replace("_1024sq_hint", "")
                title = title_lookup.get(base_name)
                if not image_path.exists():
                    image_path = image_root / base_name
            if not title:
                missing_title.append(fname)
                continue
            if not image_path.exists():
                missing_image.append(fname)
                continue
            samples.append(
                {
                    "file_name": fname,
                    "image_path": image_path,
                    "title": title,
                    "text": record.get("text", ""),
                    "shot": extract_shot_tag(record.get("text", "")),
                }
            )
    if missing_title:
        print(f"[warn] Skipped {len(missing_title)} files without title match.")
    if missing_image:
        print(f"[warn] Skipped {len(missing_image)} files without image file.")
    return samples


def preprocess_inputs(batch, processor):
    images = []
    for sample in batch:
        img = Image.open(sample["image_path"]).convert("RGB")
        images.append(img)
    return processor(images=images, return_tensors="pt")


def resolve_model_path(model_name_or_path: str) -> str:
    """Return a local snapshot path if available, otherwise the original string."""
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    hf_key = model_name_or_path.replace("/", "--")
    snapshot_root = cache_root / f"models--{hf_key}" / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted(snapshot_root.iterdir())
        if snapshots:
            return str(snapshots[-1])
    return model_name_or_path


def compute_embeddings(samples, model_name, batch_size, device):
    """Return np.ndarray embeddings from a CLIP vision tower."""
    model_path = resolve_model_path(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(
        model_path, local_files_only=True
    )
    model = CLIPVisionModel.from_pretrained(
        model_path, local_files_only=True
    )
    model.to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Encoding images"):
        batch = samples[i: i + batch_size]
        inputs = preprocess_inputs(batch, image_processor)
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        emb = outputs.pooler_output.detach().cpu().numpy()
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


def run_clustering(embeddings, n_clusters, hdbscan_min_cluster_size):
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, n_init="auto"
    )
    kmeans_labels = kmeans.fit_predict(embeddings)

    hdbscan = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=5,
        metric="euclidean",
        allow_single_cluster=False,
    )
    hdb_labels = hdbscan.fit_predict(embeddings)
    hdb_probs = getattr(hdbscan, "probabilities_", None)

    return kmeans_labels, hdb_labels, hdb_probs


def write_assignment_csv(out_csv, samples, kmeans_labels, hdb_labels, hdb_probs):
    header = [
        "file_name",
        "title",
        "shot",
        "kmeans_cluster",
        "hdbscan_cluster",
        "hdbscan_membership_prob",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, sample in enumerate(samples):
            writer.writerow(
                [
                    sample["file_name"],
                    sample["title"],
                    sample["shot"],
                    int(kmeans_labels[idx]),
                    int(hdb_labels[idx]),
                    float(hdb_probs[idx]) if hdb_probs is not None else "",
                ]
            )


def write_summary(out_txt, samples, kmeans_labels, hdb_labels):
    title_counts = Counter(sample["title"] for sample in samples)
    shot_counts = Counter(sample["shot"] for sample in samples)
    kmeans_counts = Counter(kmeans_labels)
    hdb_counts = Counter(hdb_labels)

    lines = []
    lines.append(f"Total samples: {len(samples)}")
    lines.append("\nTop 10 titles:")
    for title, count in title_counts.most_common(10):
        lines.append(f"  {title}: {count}")

    lines.append("\nShot distribution:")
    for shot, count in shot_counts.items():
        lines.append(f"  {shot}: {count}")

    lines.append("\nKMeans cluster sizes:")
    for cluster_id, count in sorted(kmeans_counts.items()):
        lines.append(f"  {cluster_id}: {count}")

    lines.append("\nHDBSCAN cluster sizes (includes -1 noise):")
    for cluster_id, count in sorted(hdb_counts.items()):
        lines.append(f"  {cluster_id}: {count}")

    out_txt.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP-based clustering for TK/final_dataset2_trigger"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("TK/final_dataset2_trigger/metadata.jsonl"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("TK/final_dataset2_trigger"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("dataset/Training/02.라벨링데이터"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TK/clustering/clip-vit-base-patch32",
        help="Path or HF repo id for a CLIP vision model (local snapshot recommended).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--kmeans-clusters", type=int, default=20)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TK/clustering"),
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    title_lookup = build_title_lookup(args.label_root)
    samples = load_samples(args.metadata, args.image_root, title_lookup)
    if not samples:
        raise SystemExit("No samples available for embedding.")

    embeddings = compute_embeddings(
        samples, args.model_name, args.batch_size, device
    )
    np.save(args.output_dir / "clip_embeddings.npy", embeddings)
    print(f"Saved embeddings to {args.output_dir / 'clip_embeddings.npy'}")

    kmeans_labels, hdb_labels, hdb_probs = run_clustering(
        embeddings,
        args.kmeans_clusters,
        args.hdbscan_min_cluster_size,
    )

    write_assignment_csv(
        args.output_dir / "cluster_assignments.csv",
        samples,
        kmeans_labels,
        hdb_labels,
        hdb_probs,
    )
    write_summary(
        args.output_dir / "cluster_summary.txt",
        samples,
        kmeans_labels,
        hdb_labels,
    )
    print("Clustering complete. Results saved under:", args.output_dir)


if __name__ == "__main__":
    main()
