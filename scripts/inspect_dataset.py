import json
from pathlib import Path
import sys


def inspect_split(split_directory):
    split_directory = Path(split_directory)

    annotation_file = split_directory / "_annotations.coco.json"

    if not annotation_file.exists():
        print(f"\nERROR: Annotation file not found:")
        print(annotation_file)
        return

    print("\n" + "=" * 70)
    print(f"DATASET SPLIT: {split_directory.name.upper()}")
    print("=" * 70)

    with open(annotation_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print(f"Images       : {len(images)}")
    print(f"Annotations  : {len(annotations)}")
    print(f"Categories   : {len(categories)}")

    # ---------------------------------------------------------
    # Categories
    # ---------------------------------------------------------

    print("\nCategories:")
    category_names = {}

    for category in categories:
        category_id = category["id"]
        category_name = category["name"]

        category_names[category_id] = category_name

        print(f"  ID {category_id}: {category_name}")

    # ---------------------------------------------------------
    # Annotation distribution
    # ---------------------------------------------------------

    print("\nAnnotation distribution:")

    counts = {}

    for annotation in annotations:
        category_id = annotation["category_id"]
        category_name = category_names.get(
            category_id,
            f"Unknown ({category_id})"
        )

        counts[category_name] = counts.get(category_name, 0) + 1

    for category_name, count in counts.items():
        print(f"  {category_name}: {count}")

    # ---------------------------------------------------------
    # Sample annotation
    # ---------------------------------------------------------

    if annotations:
        print("\nExample annotation:")

        sample = annotations[0]

        print(f"  Image ID    : {sample.get('image_id')}")
        print(f"  Category ID : {sample.get('category_id')}")
        print(f"  Bounding Box: {sample.get('bbox')}")
        print(f"  Area        : {sample.get('area')}")

    # ---------------------------------------------------------
    # Sample image
    # ---------------------------------------------------------

    if images:
        print("\nExample image:")

        sample_image = images[0]

        print(f"  ID     : {sample_image.get('id')}")
        print(f"  File   : {sample_image.get('file_name')}")
        print(f"  Width  : {sample_image.get('width')}")
        print(f"  Height : {sample_image.get('height')}")

    print()


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/inspect_dataset.py data/raw/parking_dataset")
        return

    dataset_directory = Path(sys.argv[1])

    if not dataset_directory.exists():
        print(f"ERROR: Directory not found: {dataset_directory}")
        return

    for split in ["train", "valid", "test"]:

        split_directory = dataset_directory / split

        if split_directory.exists():
            inspect_split(split_directory)
        else:
            print(f"\nWARNING: {split} directory not found.")


if __name__ == "__main__":
    main()