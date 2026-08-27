from ultralytics import YOLO
import numpy as np


class Model:

    def __init__(
        self,
        model_path="assets/models/best.pt",
        input_size=(960, 960),
        conf=0.001
    ):
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.conf = conf

    def run_on_batch(self, tensor):

        results = self.model.predict(
            source=tensor,
            imgsz=self.input_size,
            conf=self.conf,
            verbose=False
        )

        predictions = []

        for result in results:

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                predictions.append(
                    np.empty((0, 5), dtype=np.float32)
                )
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            detections = np.column_stack([
                xyxy,
                conf
            ])

            predictions.append(detections)

        return predictions