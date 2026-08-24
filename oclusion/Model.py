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
                    np.empty((0, 6), dtype=np.float32)
                )
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            # Para uma única classe:
            #
            # [x1, y1, x2, y2, objectness/confidence, class_probability]
            #
            # A Ultralytics fornece a confiança final da detecção.
            #
            class_probability = conf.copy()

            detections = np.column_stack([
                xyxy,
                conf,
                class_probability
            ])

            predictions.append(detections)

        return predictions