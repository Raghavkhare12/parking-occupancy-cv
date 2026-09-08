import argparse
from pathlib import Path

import cv2
from src.reporter import ParkingReportGenerator
from config import MODEL_FILE, OUTPUT_DIR, create_directories
from src.classifier import ParkingClassifier
from src.analyzer import ParkingLotAnalyzer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Computer Vision Based Parking Lot Occupancy Analyzer"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the parking-lot image"
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to the COCO annotation JSON file"
    )

    parser.add_argument(
        "--output",
        default="outputs/parking_analysis.jpg",
        help="Path for the annotated output image"
    )

    parser.add_argument(
        "--csv",
        default="outputs/parking_analysis.csv",
        help="Path for the CSV report"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    create_directories()

    image_path = Path(args.image)
    annotation_path = Path(args.annotations)
    output_path = Path(args.output)
    csv_path = Path(args.csv)

    print("=" * 60)
    print("PARKING LOT OCCUPANCY ANALYSIS SYSTEM")
    print("=" * 60)

    print(f"\nInput image  : {image_path}")
    print(f"Annotations  : {annotation_path}")
    print(f"Model        : {MODEL_FILE}")

    # Check required files
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    # Load trained classifier
    print("\nLoading trained classifier...")

    classifier = ParkingClassifier()
    classifier.load(MODEL_FILE)

    # Create analyzer
    analyzer = ParkingLotAnalyzer(classifier)

    print("\nAnalyzing parking spaces...")

    # Analyze image
    annotated_image, results_df, summary = analyzer.analyze(
        image_path,
        annotation_path
    )

    # Create output directories
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save annotated image
    success = cv2.imwrite(
        str(output_path),
        annotated_image
    )

    if not success:
        raise RuntimeError(
            f"Could not save output image: {output_path}"
        )

    # Save CSV report
    # Generate reports
    reporter = ParkingReportGenerator(
        output_path.parent
    )

    reporter.save_csv(
        results_df,
        csv_path.name
    )

    reporter.save_summary(
        summary
    )

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(
        f"Total parking spaces : {summary['total_spaces']}"
    )

    print(
        f"Occupied spaces      : {summary['occupied_spaces']}"
    )

    print(
        f"Empty spaces         : {summary['empty_spaces']}"
    )

    print(
        f"Occupancy percentage : "
        f"{summary['occupancy_percentage']:.2f}%"
    )

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)

    summary_path = reporter.output_directory / "parking_summary.csv"

    print(f"Annotated image : {output_path}")
    print(f"CSV report      : {csv_path}")
    print(f"Summary report  : {summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()