from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from src.preprocessing import extract_features_from_file


class ParkingClassifier:
    """
    SVM-based classifier for parking-space occupancy.

    Classes:
        0 -> empty
        1 -> occupied
    """

    def __init__(
        self,
        kernel="rbf",
        C=10.0,
        gamma="scale",
        random_state=42,
    ):
        base_svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            random_state=42
        )

        self.model = CalibratedClassifierCV(
            base_svm,
            method="sigmoid",
            cv=5
        )

    def train(self, features, labels):
        """
        Train the SVM classifier.

        Parameters
        ----------
        features : numpy.ndarray
            HOG feature matrix.

        labels : numpy.ndarray
            Class labels.
        """

        if len(features) == 0:
            raise ValueError(
                "Training features are empty."
            )

        if len(features) != len(labels):
            raise ValueError(
                "Number of features and labels must match."
            )

        print("Training SVM classifier...")
        print(f"Training samples: {len(features)}")
        print(f"Feature dimensions: {features.shape[1]}")

        self.model.fit(
            features,
            labels
        )

        print("Training completed.")

    def predict(self, features):
        """
        Predict parking-space classes.

        Returns
        -------
        numpy.ndarray
            Predicted class labels.
        """

        return self.model.predict(features)

    def predict_proba(self, features):
        """
        Predict class probabilities.
        """

        return self.model.predict_proba(features)

    def predict_image(self, image_path):
        """
        Predict occupancy for a single image.

        Returns
        -------
        tuple
            (label, confidence)
        """

        features = extract_features_from_file(
            image_path
        )

        features = features.reshape(1, -1)

        prediction = self.model.predict(
            features
        )[0]

        probabilities = self.model.predict_proba(
            features
        )[0]

        confidence = float(
            np.max(probabilities)
        )

        if prediction == 0:
            label = "empty"
        else:
            label = "occupied"

        return label, confidence

    def save(self, model_path):
        """
        Save trained model to disk.
        """

        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.model,
            model_path
        )

        print(
            f"Model saved to: {model_path}"
        )

    def load(self, model_path):
        """
        Load a trained model from disk.
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model = joblib.load(
            model_path
        )

        print(
            f"Model loaded from: {model_path}"
        )


def load_dataset_features(
    dataset_directory,
):
    """
    Extract HOG features from an entire
    processed dataset split.

    Expected structure:

        dataset_directory/
            empty/
            occupied/

    Returns
    -------
    tuple
        features, labels
    """

    dataset_directory = Path(
        dataset_directory
    )

    empty_directory = (
        dataset_directory / "empty"
    )

    occupied_directory = (
        dataset_directory / "occupied"
    )

    if not empty_directory.exists():
        raise FileNotFoundError(
            f"Missing directory: {empty_directory}"
        )

    if not occupied_directory.exists():
        raise FileNotFoundError(
            f"Missing directory: {occupied_directory}"
        )

    features = []
    labels = []

    # ---------------------------------------------------------
    # Empty parking spaces
    # ---------------------------------------------------------

    empty_images = sorted(
        empty_directory.glob("*.jpg")
    )

    print()
    print(
        f"Extracting features from "
        f"{len(empty_images)} empty images..."
    )

    for index, image_path in enumerate(
        empty_images,
        start=1
    ):

        try:

            feature_vector = (
                extract_features_from_file(
                    image_path
                )
            )

            features.append(
                feature_vector
            )

            labels.append(0)

        except (FileNotFoundError, ValueError) as error:

            print(
                f"Warning: {error}"
            )

        if index % 500 == 0:
            print(
                f"Processed {index}/"
                f"{len(empty_images)} empty images"
            )

    # ---------------------------------------------------------
    # Occupied parking spaces
    # ---------------------------------------------------------

    occupied_images = sorted(
        occupied_directory.glob("*.jpg")
    )

    print()
    print(
        f"Extracting features from "
        f"{len(occupied_images)} occupied images..."
    )

    for index, image_path in enumerate(
        occupied_images,
        start=1
    ):

        try:

            feature_vector = (
                extract_features_from_file(
                    image_path
                )
            )

            features.append(
                feature_vector
            )

            labels.append(1)

        except (FileNotFoundError, ValueError) as error:

            print(
                f"Warning: {error}"
            )

        if index % 500 == 0:
            print(
                f"Processed {index}/"
                f"{len(occupied_images)} "
                f"occupied images"
            )

    if not features:
        raise ValueError(
            f"No valid images found in "
            f"{dataset_directory}"
        )

    return (
        np.array(features),
        np.array(labels)
    )


def main():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    model_directory = (
        project_root
        / "models"
    )

    model_path = (
        model_directory
        / "parking_classifier.joblib"
    )

    # ---------------------------------------------------------
    # Load training features
    # ---------------------------------------------------------

    train_directory = (
        processed_directory
        / "train"
    )

    print("=" * 60)
    print("PARKING OCCUPANCY CLASSIFIER")
    print("=" * 60)

    X_train, y_train = load_dataset_features(
        train_directory
    )

    print()
    print(
        f"Training feature matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Training labels: "
        f"{y_train.shape}"
    )

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    classifier = ParkingClassifier()

    classifier.train(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    classifier.save(
        model_path
    )

    print()
    print("=" * 60)
    print("CLASSIFIER CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()