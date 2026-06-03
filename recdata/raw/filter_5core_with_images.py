import argparse
import csv
from collections import Counter
from pathlib import Path


DATASET_CONFIGS = {
    "Beauty": {
        "data_dir": "Beauty",
        "review_files": ["All_Beauty.jsonl"],
    },
    "Baby": {
        "data_dir": "Baby",
        "review_files": ["All_Baby.jsonl", "Baby_Products.jsonl"],
    },
    "Clothing": {
        "data_dir": "Clothing",
        "review_files": ["All_Clothing.jsonl", "Clothing_Shoes_and_Jewelry.jsonl"],
    },
}


def first_existing_path(data_dir: Path, filenames: list[str], label: str) -> Path:
    for filename in filenames:
        path = data_dir / filename
        if path.exists():
            return path
    tried = ", ".join(str(data_dir / filename) for filename in filenames)
    raise FileNotFoundError(f"{label} file not found. Tried: {tried}")


def collect_nonempty_image_item_ids(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        return set()
    return {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0
    }


def load_interactions(path: Path) -> list[tuple[str, str]]:
    interactions = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get("user_id")
            parent_asin = row.get("parent_asin")
            if user_id and parent_asin:
                interactions.append((user_id, parent_asin))
    return interactions


def load_items(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or ["parent_asin", "title", "image"]
    return rows, fieldnames


def filter_k_core(
    interactions: list[tuple[str, str]],
    min_interactions: int,
) -> list[tuple[str, str]]:
    filtered = interactions
    while True:
        user_counts = Counter(user_id for user_id, _ in filtered)
        item_counts = Counter(parent_asin for _, parent_asin in filtered)
        next_filtered = [
            (user_id, parent_asin)
            for user_id, parent_asin in filtered
            if user_counts[user_id] >= min_interactions
            and item_counts[parent_asin] >= min_interactions
        ]
        if len(next_filtered) == len(filtered):
            return filtered
        filtered = next_filtered


def write_interactions(path: Path, interactions: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "parent_asin"])
        writer.writerows(interactions)


def write_users(path: Path, users: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id"])
        writer.writerows((user_id,) for user_id in users)


def write_items(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_orphan_images(image_dir: Path, final_items: set[str]) -> list[Path]:
    if not image_dir.exists():
        return []
    return [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.stem not in final_items
    ]


def process_dataset(
    dataset_name: str,
    raw_dir: Path,
    min_interactions: int,
    dry_run: bool,
    delete_images: bool,
) -> None:
    if dataset_name not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset: {dataset_name}. Choices: {choices}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    data_dir = raw_dir / dataset_config["data_dir"]
    review_path = first_existing_path(data_dir, dataset_config["review_files"], "Review")
    output_prefix = review_path.stem

    interaction_path = data_dir / f"{output_prefix}_5core_interactions.csv"
    user_path = data_dir / f"{output_prefix}_5core_users.csv"
    item_path = data_dir / f"{output_prefix}_5core_items.csv"
    image_dir = data_dir / "image"

    interactions = load_interactions(interaction_path)
    item_rows, item_fieldnames = load_items(item_path)
    item_by_id = {
        row["parent_asin"]: row
        for row in item_rows
        if row.get("parent_asin")
    }

    nonempty_image_item_ids = collect_nonempty_image_item_ids(image_dir)
    items_with_images = set(item_by_id) & nonempty_image_item_ids
    image_filtered_interactions = [
        (user_id, parent_asin)
        for user_id, parent_asin in interactions
        if parent_asin in items_with_images
    ]
    final_interactions = filter_k_core(image_filtered_interactions, min_interactions)
    final_users = sorted({user_id for user_id, _ in final_interactions})
    final_items = sorted({parent_asin for _, parent_asin in final_interactions})
    final_item_set = set(final_items)
    final_item_rows = [
        row
        for row in item_rows
        if row.get("parent_asin") in final_item_set
    ]
    orphan_images = collect_orphan_images(image_dir, final_item_set)

    print(f"Dataset: {dataset_name}")
    print(f"Original interactions: {len(interactions)}")
    print(f"Original items: {len(item_rows)}")
    print(f"Items with local nonempty images: {len(items_with_images)}")
    print(f"Interactions after image filter: {len(image_filtered_interactions)}")
    print(f"Final 5-core interactions: {len(final_interactions)}")
    print(f"Final users: {len(final_users)}")
    print(f"Final items: {len(final_items)}")
    print(f"Images to delete: {len(orphan_images)}")
    print(f"Dry run: {dry_run}")

    if dry_run:
        return

    write_interactions(interaction_path, final_interactions)
    write_users(user_path, final_users)
    write_items(item_path, final_item_rows, item_fieldnames)

    if delete_images:
        for path in orphan_images:
            path.unlink()

    print(f"Updated interactions: {interaction_path}")
    print(f"Updated users: {user_path}")
    print(f"Updated items: {item_path}")
    if delete_images:
        print(f"Deleted images: {len(orphan_images)}")
    else:
        print("Skipped image deletion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter 5-core recommendation CSVs to items with local nonempty images "
            "and remove images for filtered-out items."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="Baby",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing dataset folders such as Baby/ and Beauty/.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum interactions for both users and items in the final k-core.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts; do not overwrite CSVs or delete images.",
    )
    parser.add_argument(
        "--no-delete-images",
        action="store_true",
        help="Overwrite CSVs but keep image files for filtered-out items.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        dataset_name=args.dataset,
        raw_dir=args.raw_dir,
        min_interactions=args.min_interactions,
        dry_run=args.dry_run,
        delete_images=not args.no_delete_images,
    )
