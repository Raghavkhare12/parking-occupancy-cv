import json
from pathlib import Path

import cv2
import numpy as np

from src.dataset import crop_bounding_box
from src.preprocessing import preprocess_image, extract_hog_features


def test_crop_bounding_box():
    """
    Test that a COCO bounding box produces a valid crop.
    """

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    bbox = [10, 20, 30, 40]

    crop = crop_bounding_box(image, bbox)

    assert crop is not None
    assert crop.shape[0] == 40
    assert crop.shape[1] == 30


def test_preprocess_image():
    """
    Test image resizing, grayscale conversion,
    and normalization.
    """

    image = np.zeros((100, 80, 3), dtype=np.uint8)

    processed = preprocess_image(image)

    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32

    assert processed.min() >= 0
    assert processed.max() <= 1


def test_hog_feature_extraction():
    """
    Test that HOG produces the expected feature size.
    """

    # HOG extraction expects a BGR image because
    # preprocessing performs BGR-to-grayscale conversion.
    image = np.zeros(
        (64, 64, 3),
        dtype=np.uint8
    )

    features = extract_hog_features(image)

    assert features.shape == (1764,)


def test_sample_annotations():
    """
    Test that the sample annotation file is valid
    and contains parking-space annotations.
    """

    annotation_file = Path(
        "data/sample/annotations.json"
    )

    assert annotation_file.exists()

    with open(
        annotation_file,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    assert "images" in data
    assert "annotations" in data
    assert "categories" in data

    assert len(data["images"]) == 1
    assert len(data["annotations"]) > 0


def test_sample_image_exists():
    """
    Test that the sample parking image exists.
    """

    sample_dir = Path("data/sample")

    images = list(sample_dir.glob("*.jpg"))

    assert len(images) >= 1


def test_output_image_exists():
    """
    Test that the application generated an
    annotated parking-lot image.
    """

    output_file = Path(
        "outputs/parking_analysis.jpg"
    )

    assert output_file.exists()

    image = cv2.imread(str(output_file))

    assert image is not None
    assert image.size > 0


def test_output_csv_exists():
    """
    Test that the application generated a CSV report.
    """

    output_file = Path(
        "outputs/parking_analysis.csv"
    )

    assert output_file.exists()

    import pandas as pd

    df = pd.read_csv(output_file)

    assert len(df) > 0

    required_columns = {
        "space_id",
        "x",
        "y",
        "width",
        "height",
        "prediction",
        "confidence"
    }

    assert required_columns.issubset(df.columns)