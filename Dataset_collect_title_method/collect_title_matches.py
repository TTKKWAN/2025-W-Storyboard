#!/usr/bin/env python3
"""
Collect title-matched images and intersect them with an existing metadata list.

The script scans JSON label files under dataset/Training/02.라벨링데이터, copies the
corresponding source images into TK/Dataset_close_shot, and writes a metadata.jsonl
file containing only the entries that also exist in TK/Dataset/matched.jsonl.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set

try:
    from PIL import Image
except ImportError:
    Image = None


TARGET_TITLES: Sequence[str] = (
    "40억 vs 1",
    "너에게 하고 싶은 말",
    "환생소녀",
    "편지창조",
    "강도하 청춘 시리즈",
    "다정종합병원",
)

REPO_ROOT = Path("/home/aikusrv01/storyboard")
LABEL_ROOT = REPO_ROOT / "dataset" / "Training" / "02.라벨링데이터"
IMAGE_ROOT = REPO_ROOT / "dataset" / "Training" / "01.원천데이터"
MATCHED_JSONL = Path("/home/aikusrv01/storyboard/TK/Dataset_fin/metadata.jsonl")  # 이게 SY 마지막 ver dataset이랑 똑같은 데이터셋

OUTPUT_DIR = Path("/home/aikusrv01/storyboard/TK/Dataset_fin/cluster")
OUTPUT_METADATA = Path("/home/aikusrv01/storyboard/TK/Dataset_fin/cluster/metadata.jsonl")



@dataclass
class Match:
    file_name: str
    caption: str


def find_image_path(json_path: Path, source_name: str) -> Path | None:
    """Return the best-guess path to the source image for a label JSON."""
    category_dir = json_path.parent.name
    candidate = IMAGE_ROOT / category_dir / source_name
    if candidate.exists():
        return candidate

    # Fall back to searching across the whole image tree once per miss.
    matches = list(IMAGE_ROOT.rglob(source_name))
    return matches[0] if matches else None


def iter_label_files(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*.json")) if path.is_file())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def has_target_title(title: str) -> bool:
    normalized = title.strip()
    return any(keyword in normalized for keyword in TARGET_TITLES)


def base_file_name(file_name: str) -> str:
    stem, ext = file_name.rsplit(".", 1)
    if stem.endswith("_flip"):
        stem = stem[:-5]
    return f"{stem}.{ext}"


def person_file_name_candidates(file_name: str) -> Set[str]:
    """Return possible person-only file names for a composite background/person image."""
    base = base_file_name(file_name)
    try:
        stem, ext = base.rsplit(".", 1)
    except ValueError:
        return set()

    if "_" not in stem:
        return set()

    person_id = stem.rsplit("_", 1)[1]
    candidates = {f"{person_id}.{ext}"}

    original_stem = file_name.rsplit(".", 1)[0]
    if original_stem.endswith("_flip"):
        candidates.add(f"{person_id}_flip.{ext}")

    return candidates


def collect_matches() -> tuple[List[Match], Set[str]]:
    matches: List[Match] = []
    file_names: Set[str] = set()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for json_path in iter_label_files(LABEL_ROOT):
        data = load_json(json_path)
        title = data.get("meta", {}).get("product", {}).get("title")
        if not title or not has_target_title(title):
            continue

        source_path = data.get("meta", {}).get("dataset", {}).get("source_path", "")
        source_name = Path(source_path).name if source_path else ""
        if not source_name:
            # Default to matching the label's numeric id (LA -> SA variant).
            label_name = json_path.stem
            source_name = label_name.replace("LA", "SA", 1) + ".JPEG"

        image_path = find_image_path(json_path, source_name)
        if image_path is None:
            print(f"[WARN] Image not found for {json_path} (expected {source_name})")
            continue

        destination = OUTPUT_DIR / image_path.name
        if not destination.exists():
            shutil.copy2(image_path, destination)

        text = data.get("caption") or data.get("label", {}).get("prompt") or ""
        matches.append(Match(file_name=destination.name, caption=text))
        file_names.add(destination.name)

    return matches, file_names


def ensure_flip_image(target_name: str) -> None:
    stem, ext = target_name.rsplit(".", 1)
    if not stem.endswith("_flip"):
        return

    base_name = f"{stem[:-5]}.{ext}"
    base_path = OUTPUT_DIR / base_name
    flip_path = OUTPUT_DIR / target_name
    if not base_path.exists() or flip_path.exists():
        return

    if Image is None:
        shutil.copy2(base_path, flip_path)
        print(f"[WARN] Pillow not installed; copied {base_name} to {target_name} without flipping.")
        return

    with Image.open(base_path) as img:
        img.transpose(Image.FLIP_LEFT_RIGHT).save(flip_path)


def load_matched_intersection(file_names: Set[str]) -> List[Match]:
    if not MATCHED_JSONL.exists():
        print(f"[WARN] Matched file not found: {MATCHED_JSONL}")
        return []

    matches: List[Match] = []
    with MATCHED_JSONL.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            fname = data.get("file_name")
            if not fname:
                continue

            has_exact = fname in file_names
            base_name = base_file_name(fname)
            has_base = base_name in file_names
            person_candidates = person_file_name_candidates(fname)
            has_person = any(candidate in file_names for candidate in person_candidates)

            if not (has_exact or has_base or has_person):
                continue

            if has_base and not has_exact and fname != base_name:
                ensure_flip_image(fname)
                file_names.add(fname)
            elif has_person and fname not in file_names:
                file_names.add(fname)

            matches.append(Match(file_name=fname, caption=data.get("text", "")))

    return matches


def write_metadata(matches: Sequence[Match]) -> None:
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w", encoding="utf-8") as fp:
        for match in matches:
            json.dump(
                {"file_name": match.file_name, "text": match.caption},
                fp,
                ensure_ascii=False,
            )
            fp.write("\n")


def main() -> None:
    matches, file_names = collect_matches()
    intersected = load_matched_intersection(file_names)
    write_metadata(intersected)
    print(
        f"Collected {len(matches)} matching images into {OUTPUT_DIR}. "
        f"{len(intersected)} entries matched {MATCHED_JSONL} "
        f"and were written to {OUTPUT_METADATA}"
    )


if __name__ == "__main__":
    main()
