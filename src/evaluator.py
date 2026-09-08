from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from classifier import ParkingClassifier, load_dataset_features


def evaluate_model(
    classifier,
    features,
    labels,
    dataset_name="Dataset",
):
    """
    Evaluate a trained parking occupancy classifier.

    Parameters
    ----------
    classifier : ParkingClassifier
        Trained classifier.

    features : numpy.ndarray
        Feature matrix.

    labels : numpy.ndarray
        True labels.

    dataset_name : str
        Name of the dataset being evaluated.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    print()
    print("=" * 60)
    print(f"{dataset_name.upper()} EVALUATION")
    print("=" * 60)

    predictions = classifier.predict(features)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    print()
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "empty",
                "occupied"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions,
    }


def save_confusion_matrix(
    labels,
    predictions,
    output_path,
):
    """
    Generate and save a confusion matrix.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "empty",
            "occupied"
        ]
    )

    display.plot()

    plt.title(
        "Parking Occupancy Classification"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{output_path}"
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

    model_path = (
        project_root
        / "models"
        / "parking_classifier.joblib"
    )

    output_directory = (
        project_root
        / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load trained classifier
    # ---------------------------------------------------------

    classifier = ParkingClassifier()

    classifier.load(
        model_path
    )

    # ---------------------------------------------------------
    # Evaluate validation dataset
    # ---------------------------------------------------------

    validation_directory = (
        processed_directory
        / "valid"
    )

    print()
    print(
        "Loading validation dataset..."
    )

    X_valid, y_valid = load_dataset_features(
        validation_directory
    )

    print()
    print(
        f"Validation feature matrix: "
        f"{X_valid.shape}"
    )

    validation_results = evaluate_model(
        classifier,
        X_valid,
        y_valid,
        dataset_name="Validation"
    )

    save_confusion_matrix(
        y_valid,
        validation_results["predictions"],
        output_directory
        / "validation_confusion_matrix.png"
    )

    # ---------------------------------------------------------
    # Evaluate test dataset
    # ---------------------------------------------------------

    test_directory = (
        processed_directory
        / "test"
    )

    print()
    print(
        "Loading test dataset..."
    )

    X_test, y_test = load_dataset_features(
        test_directory
    )

    print()
    print(
        f"Test feature matrix: "
        f"{X_test.shape}"
    )

    test_results = evaluate_model(
        classifier,
        X_test,
        y_test,
        dataset_name="Final Test"
    )

    save_confusion_matrix(
        y_test,
        test_results["predictions"],
        output_directory
        / "test_confusion_matrix.png"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL MODEL EVALUATION SUMMARY")
    print("=" * 60)

    print()
    print("Validation:")
    print(
        f"  Accuracy  : "
        f"{validation_results['accuracy']:.4f}"
    )
    print(
        f"  Precision : "
        f"{validation_results['precision']:.4f}"
    )
    print(
        f"  Recall    : "
        f"{validation_results['recall']:.4f}"
    )
    print(
        f"  F1 Score  : "
        f"{validation_results['f1_score']:.4f}"
    )

    print()
    print("Final Test:")
    print(
        f"  Accuracy  : "
        f"{test_results['accuracy']:.4f}"
    )
    print(
        f"  Precision : "
        f"{test_results['precision']:.4f}"
    )
    print(
        f"  Recall    : "
        f"{test_results['recall']:.4f}"
    )
    print(
        f"  F1 Score  : "
        f"{test_results['f1_score']:.4f}"
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()