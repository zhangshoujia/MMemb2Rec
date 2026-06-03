import argparse
import csv
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


def image_exists(image_dir: Path, parent_asin: str | None) -> bool:
    if not parent_asin:
        return False
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in image_dir.glob(f"{parent_asin}.*")
    )


def write_missing_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["parent_asin", "title", "image", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_missing_images(dataset_name: str, raw_dir: Path) -> None:
    if dataset_name not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset: {dataset_name}. Choices: {choices}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    data_dir = raw_dir / dataset_config["data_dir"]
    review_path = first_existing_path(
        data_dir,
        dataset_config["review_files"],
        "Review",
    )

    output_prefix = review_path.stem
    items_path = data_dir / f"{output_prefix}_5core_items.csv"
    image_dir = data_dir / "image"
    missing_log_path = data_dir / f"{output_prefix}_missing_images.csv"
    summary_log_path = data_dir / f"{output_prefix}_missing_images_summary.log"

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")

    with items_path.open("r", encoding="utf-8", newline="") as f:
        item_rows = list(csv.DictReader(f))

    total_items = len(item_rows)
    missing_rows = []
    for index, row in enumerate(item_rows, start=1):
        parent_asin = row.get("parent_asin")
        if not image_exists(image_dir, parent_asin):
            missing_rows.append(
                {
                    "parent_asin": parent_asin or "",
                    "title": row.get("title", ""),
                    "image": row.get("image", ""),
                    "reason": "image file missing or empty",
                }
            )
        if index % 100 == 0 or index == total_items:
            print(
                f"{index}/{total_items} checked parent_asin={parent_asin} "
                f"missing={len(missing_rows)}",
                flush=True,
            )

    write_missing_csv(missing_log_path, missing_rows)

    existing_count = len(item_rows) - len(missing_rows)
    summary_lines = [
        f"dataset={dataset_name}",
        f"items_path={items_path}",
        f"image_dir={image_dir}",
        f"total_items={len(item_rows)}",
        f"existing_images={existing_count}",
        f"missing_images={len(missing_rows)}",
        f"missing_log={missing_log_path}",
    ]
    summary_log_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for line in summary_lines:
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count missing Amazon item images and write missing-image logs."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="Baby",
        help="Dataset to check.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing dataset folders such as Baby/ and Beauty/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_missing_images(args.dataset, args.raw_dir)
