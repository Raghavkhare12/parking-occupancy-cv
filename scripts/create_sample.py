import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "parking_dataset" / "test"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

ANNOTATION_FILE = TEST_DIR / "_annotations.coco.json"


def create_sample():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    # Find an image that has parking-space annotations
    selected_image = None
    selected_annotations = []

    for image in coco_data["images"]:
        image_id = image["id"]

        annotations = [
            annotation
            for annotation in coco_data["annotations"]
            if annotation["image_id"] == image_id
            and annotation["category_id"] in [1, 2]
        ]

        if len(annotations) >= 5:
            selected_image = image
            selected_annotations = annotations
            break

    if selected_image is None:
        raise RuntimeError("Could not find a suitable sample image.")

    image_name = selected_image["file_name"]
    source_image = TEST_DIR / image_name

    if not source_image.exists():
        raise FileNotFoundError(
            f"Image not found: {source_image}"
        )

    # Copy image
    destination_image = SAMPLE_DIR / image_name
    shutil.copy2(source_image, destination_image)

    # Keep only the selected image and its annotations
    sample_data = {
        "images": [selected_image],
        "annotations": selected_annotations,
        "categories": coco_data["categories"]
    }

    destination_annotations = SAMPLE_DIR / "annotations.json"

    with open(
        destination_annotations,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            sample_data,
            file,
            indent=2
        )

    print("=" * 60)
    print("SAMPLE CREATED")
    print("=" * 60)

    print(f"Image       : {destination_image}")
    print(f"Annotations : {destination_annotations}")
    print(f"Parking spaces: {len(selected_annotations)}")


if __name__ == "__main__":
    create_sample()