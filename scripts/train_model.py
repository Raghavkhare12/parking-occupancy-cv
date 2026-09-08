from pathlib import Path

from src.classifier import ParkingClassifier, load_dataset_features


def main():
    project_root = Path(__file__).resolve().parent.parent

    train_directory = (
        project_root
        / "data"
        / "processed"
        / "train"
    )

    model_path = (
        project_root
        / "models"
        / "parking_classifier.joblib"
    )

    print("=" * 60)
    print("PARKING OCCUPANCY MODEL TRAINING")
    print("=" * 60)

    print("\nLoading training dataset...")

    X_train, y_train = load_dataset_features(
        train_directory
    )

    print(
        f"\nTraining feature matrix: {X_train.shape}"
    )

    print(
        f"Training labels: {y_train.shape}"
    )

    classifier = ParkingClassifier()

    classifier.train(
        X_train,
        y_train
    )

    classifier.save(
        model_path
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()