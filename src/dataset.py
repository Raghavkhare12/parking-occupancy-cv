import json
import random
from pathlib import Path

import cv2
import pandas as pd


# COCO category IDs in your dataset
EMPTY_CATEGORY_ID = 1
OCCUPIED_CATEGORY_ID = 2


def load_coco_annotations(split_directory):
    """
    Load COCO annotations from a dataset split.

    Parameters
    ----------
    split_directory : str or Path
        Path to train, valid, or test directory.

    Returns
    -------
    dict
        Loaded COCO annotation data.
    """

    split_directory = Path(split_directory)

    annotation_file = split_directory / "_annotations.coco.json"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def get_category_names(coco_data):
    """
    Return category ID -> category name mapping.
    """

    return {
        category["id"]: category["name"]
        for category in coco_data.get("categories", [])
    }


def create_image_mapping(coco_data):
    """
    Create a mapping from image ID to image information.
    """

    return {
        image["id"]: image
        for image in coco_data.get("images", [])
    }


def get_valid_annotations(coco_data):
    """
    Return only annotations belonging to the two occupancy classes.

    Category 1 = space-empty
    Category 2 = space-occupied
    """

    valid_annotations = []

    for annotation in coco_data.get("annotations", []):

        category_id = annotation.get("category_id")

        if category_id in {
            EMPTY_CATEGORY_ID,
            OCCUPIED_CATEGORY_ID,
        }:
            valid_annotations.append(annotation)

    return valid_annotations


def crop_bounding_box(image, bounding_box):
    """
    Crop an image using a COCO bounding box.

    COCO format:
        [x, y, width, height]
    """

    x, y, width, height = bounding_box

    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    width = int(round(width))
    height = int(round(height))

    image_height, image_width = image.shape[:2]

    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    if x >= x2 or y >= y2:
        return None

    crop = image[y:y2, x:x2]

    if crop.size == 0:
        return None

    return crop


def collect_samples(
    split_directory,
    max_samples_per_class=None,
    random_seed=42,
):
    """
    Collect parking-space samples from one dataset split.

    Parameters
    ----------
    split_directory : str or Path
        Path to train/valid/test.
    max_samples_per_class : int or None
        Maximum number of samples from each class.
        None means use all samples.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    list
        List of sample dictionaries.
    """

    split_directory = Path(split_directory)

    coco_data = load_coco_annotations(split_directory)

    image_mapping = create_image_mapping(coco_data)

    annotations = get_valid_annotations(coco_data)

    random.seed(random_seed)

    # Separate annotations by class
    empty_annotations = []
    occupied_annotations = []

    for annotation in annotations:

        if annotation["category_id"] == EMPTY_CATEGORY_ID:
            empty_annotations.append(annotation)

        elif annotation["category_id"] == OCCUPIED_CATEGORY_ID:
            occupied_annotations.append(annotation)

    # Shuffle before selecting samples
    random.shuffle(empty_annotations)
    random.shuffle(occupied_annotations)

    if max_samples_per_class is not None:

        empty_annotations = empty_annotations[
            :max_samples_per_class
        ]

        occupied_annotations = occupied_annotations[
            :max_samples_per_class
        ]

    selected_annotations = (
        empty_annotations +
        occupied_annotations
    )

    samples = []

    for annotation in selected_annotations:

        image_id = annotation["image_id"]

        if image_id not in image_mapping:
            continue

        image_info = image_mapping[image_id]

        image_filename = image_info["file_name"]

        image_path = split_directory / image_filename

        if not image_path.exists():
            continue

        category_id = annotation["category_id"]

        if category_id == EMPTY_CATEGORY_ID:
            label = "empty"

        else:
            label = "occupied"

        samples.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "bbox": annotation["bbox"],
                "category_id": category_id,
                "label": label,
            }
        )

    random.shuffle(samples)

    return samples


def save_processed_dataset(
    dataset_root,
    output_root,
    train_samples=3000,
    valid_samples=1000,
    test_samples=1000,
):
    """
    Extract a manageable subset of parking-space crops.

    The number specified is the maximum number of samples
    PER CLASS for each split.
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    splits = {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples,
    }

    output_root.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for split, samples_per_class in splits.items():

        split_directory = dataset_root / split

        if not split_directory.exists():
            print(f"Skipping missing split: {split}")
            continue

        print()
        print("=" * 60)
        print(f"Processing {split.upper()} dataset")
        print("=" * 60)

        samples = collect_samples(
            split_directory=split_directory,
            max_samples_per_class=samples_per_class,
        )

        print(f"Selected samples: {len(samples)}")

        split_output = output_root / split

        empty_output = split_output / "empty"
        occupied_output = split_output / "occupied"

        empty_output.mkdir(parents=True, exist_ok=True)
        occupied_output.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(samples):

            image = cv2.imread(sample["image_path"])

            if image is None:
                print(
                    f"Warning: Could not read "
                    f"{sample['image_path']}"
                )
                continue

            crop = crop_bounding_box(
                image,
                sample["bbox"]
            )

            if crop is None:
                continue

            label = sample["label"]

            output_directory = (
                empty_output
                if label == "empty"
                else occupied_output
            )

            output_filename = (
                f"{sample['image_id']}_{index}.jpg"
            )

            output_path = (
                output_directory / output_filename
            )

            success = cv2.imwrite(
                str(output_path),
                crop
            )

            if not success:
                continue

            metadata_rows.append(
                {
                    "split": split,
                    "image_id": sample["image_id"],
                    "label": label,
                    "category_id": sample["category_id"],
                    "original_image": sample["image_path"],
                    "crop_path": str(output_path),
                }
            )

            if (index + 1) % 500 == 0:
                print(
                    f"Processed {index + 1}/"
                    f"{len(samples)}"
                )

        print(f"Completed {split}")

    metadata = pd.DataFrame(metadata_rows)

    metadata_path = output_root / "metadata.csv"

    metadata.to_csv(
        metadata_path,
        index=False
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Total crops: {len(metadata)}")
    print(f"Metadata: {metadata_path}")

    if not metadata.empty:
        print()
        print("Class distribution:")
        print(
            metadata.groupby(
                ["split", "label"]
            ).size()
        )

    return metadata


if __name__ == "__main__":

    # Allows this file to be run directly for testing.

    project_root = Path(__file__).resolve().parent.parent

    dataset_root = (
        project_root
        / "data"
        / "raw"
        / "parking_dataset"
    )

    output_root = (
        project_root
        / "data"
        / "processed"
    )

    save_processed_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        train_samples=3000,
        valid_samples=1000,
        test_samples=1000,
    )