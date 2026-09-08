from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


# Standard image size used by the classifier
IMAGE_SIZE = (64, 64)


def load_image(image_path):
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str or Path
        Path to the image.

    Returns
    -------
    numpy.ndarray
        Loaded image.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.
    ValueError
        If OpenCV cannot read the image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    return image


def preprocess_image(image):
    """
    Preprocess a parking-space image.

    Steps:
        1. Resize
        2. Convert to grayscale
        3. Normalize pixel values

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale image.
    """

    # Resize to a consistent size
    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert BGR to grayscale
    grayscale = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize pixel values to [0, 1]
    normalized = grayscale.astype(
        np.float32
    ) / 255.0

    return normalized


def extract_hog_features(image):
    """
    Extract HOG features from a parking-space image.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    Returns
    -------
    numpy.ndarray
        HOG feature vector.
    """

    preprocessed = preprocess_image(image)

    features = hog(
        preprocessed,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    return features.astype(np.float32)


def extract_features_from_file(image_path):
    """
    Load an image and extract its HOG features.

    Parameters
    ----------
    image_path : str or Path
        Path to image.

    Returns
    -------
    numpy.ndarray
        HOG feature vector.
    """

    image = load_image(image_path)

    return extract_hog_features(image)


def extract_features_from_directory(directory):
    """
    Extract HOG features from all JPG/PNG images
    inside a directory.

    Parameters
    ----------
    directory : str or Path
        Image directory.

    Returns
    -------
    tuple
        (features, image_paths)
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    image_paths = sorted(
        [
            path
            for path in directory.iterdir()
            if path.suffix.lower() in image_extensions
        ]
    )

    features = []
    valid_paths = []

    for image_path in image_paths:

        try:
            feature_vector = extract_features_from_file(
                image_path
            )

            features.append(feature_vector)
            valid_paths.append(image_path)

        except (FileNotFoundError, ValueError) as error:

            print(
                f"Warning: {error}"
            )

    if not features:
        raise ValueError(
            f"No valid images found in {directory}"
        )

    return (
        np.array(features),
        valid_paths
    )


if __name__ == "__main__":

    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)

    print(f"Image size: {IMAGE_SIZE}")

    # Test using the first image in the training set
    project_root = Path(__file__).resolve().parent.parent

    train_empty_directory = (
        project_root
        / "data"
        / "processed"
        / "train"
        / "empty"
    )

    image_files = list(
        train_empty_directory.glob("*.jpg")
    )

    if not image_files:

        print(
            "No test images found."
        )

    else:

        test_image = image_files[0]

        print(f"Test image: {test_image.name}")

        features = extract_features_from_file(
            test_image
        )

        print(
            f"HOG feature vector size: "
            f"{len(features)}"
        )

        print(
            "Preprocessing test successful."
        )