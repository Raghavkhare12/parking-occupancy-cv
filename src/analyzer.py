from pathlib import Path
import json
import cv2
import pandas as pd

from src.preprocessing import extract_features_from_file


class ParkingLotAnalyzer:
    """
    Analyzes a parking-lot image using parking-space annotations
    and a trained occupancy classifier.
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def load_annotations(self, annotation_file):
        """
        Load COCO-format annotations.
        """
        with open(annotation_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    def find_image_annotations(self, coco_data, image_path):
        """
        Find the image ID and parking-space annotations
        corresponding to the input image.
        """

        image_name = Path(image_path).name

        image_info = None

        for image in coco_data["images"]:
            if image["file_name"] == image_name:
                image_info = image
                break

        if image_info is None:
            raise ValueError(
                f"Image '{image_name}' was not found in the annotation file."
            )

        image_id = image_info["id"]

        annotations = [
            annotation
            for annotation in coco_data["annotations"]
            if annotation["image_id"] == image_id
            and annotation["category_id"] in [1, 2]
        ]

        if not annotations:
            raise ValueError(
                f"No parking-space annotations found for '{image_name}'."
            )

        return annotations

    def crop_from_bbox(self, image, bbox):
        """
        Crop a parking-space region using a COCO bounding box.

        COCO format:
        [x, y, width, height]
        """

        x, y, width, height = bbox

        x = max(0, int(x))
        y = max(0, int(y))
        width = int(width)
        height = int(height)

        image_height, image_width = image.shape[:2]

        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        if x >= x2 or y >= y2:
            raise ValueError("Invalid bounding box.")

        return image[y:y2, x:x2]

    def predict_space(self, crop):
        """
        Predict whether a parking space is empty or occupied.
        """

        if crop is None or crop.size == 0:
            raise ValueError("Empty parking-space crop.")

        # extract_features_from_file() normally handles preprocessing
        # internally, but here we already have the image in memory.
        from src.preprocessing import preprocess_image, extract_hog_features

        # Preprocess exactly once.
        processed = preprocess_image(crop)

        # HOG extraction receives the already-preprocessed image.
        # We call HOG directly here instead of extract_features_from_file().
        from skimage.feature import hog

        features = hog(
            processed,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = features.reshape(1, -1)

        prediction = self.classifier.predict(features)[0]

        confidence = None

        if hasattr(self.classifier.model, "predict_proba"):
            probabilities = self.classifier.model.predict_proba(features)[0]
            confidence = float(max(probabilities))

        return prediction, confidence

    def analyze(self, image_path, annotation_file):
        """
        Analyze every parking space in the input image.
        """

        image_path = Path(image_path)
        annotation_file = Path(annotation_file)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        coco_data = self.load_annotations(annotation_file)

        annotations = self.find_image_annotations(
            coco_data,
            image_path
        )

        results = []

        for index, annotation in enumerate(annotations, start=1):

            bbox = annotation["bbox"]

            crop = self.crop_from_bbox(
                image,
                bbox
            )

            prediction, confidence = self.predict_space(crop)

            x, y, width, height = bbox

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + width)
            y2 = int(y + height)

            # The classifier uses:
            # 0 = empty
            # 1 = occupied
            if int(prediction) == 1:
                label = "Occupied"
                box_color = (0, 0, 255)      # Red
                prediction_name = "occupied"
            else:
                label = "Empty"
                box_color = (0, 255, 0)      # Green
                prediction_name = "empty"

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # Prepare label
            if confidence is not None:
                text = f"{index}: {label} ({confidence:.2f})"
            else:
                text = f"{index}: {label}"

            cv2.putText(
                image,
                text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            results.append({
                "space_id": index,
                "x": x1,
                "y": y1,
                "width": int(width),
                "height": int(height),
                "prediction": prediction_name,
                "confidence": confidence
            })

        total_spaces = len(results)

        occupied_spaces = sum(
            result["prediction"] == "occupied"
            for result in results
        )

        empty_spaces = total_spaces - occupied_spaces

        occupancy_percentage = (
            occupied_spaces / total_spaces * 100
            if total_spaces > 0
            else 0
        )

        summary = {
            "total_spaces": total_spaces,
            "occupied_spaces": occupied_spaces,
            "empty_spaces": empty_spaces,
            "occupancy_percentage": occupancy_percentage
        }

        return image, pd.DataFrame(results), summary