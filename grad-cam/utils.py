import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def calculate_box_iou(box1, box2):
    """
    Calcula IoU entre duas bounding boxes.

    Formato:
        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0]) *
        max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0]) *
        max(0.0, box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def get_target_detection(
    model,
    image_tensor,
    detection_index=0,
    imgsz=(960, 960),
    conf=0.001
):
    """
    Obtém a detecção original utilizada como alvo.

    Retorna:

        target = [x1, y1, x2, y2, confidence]
    """

    predictions = model.predict(
        source=image_tensor,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    boxes = predictions[0].boxes

    if boxes is None or len(boxes) == 0:
        raise ValueError(
            "Nenhuma detecção encontrada."
        )

    if detection_index >= len(boxes):
        raise ValueError(
            f"detection_index={detection_index}, "
            f"mas existem apenas "
            f"{len(boxes)} detecções."
        )

    xyxy = (
        boxes.xyxy[detection_index]
        .detach()
        .cpu()
        .numpy()
    )

    confidence = (
        boxes.conf[detection_index]
        .detach()
        .cpu()
        .item()
    )

    target = np.concatenate([
        xyxy,
        [confidence]
    ])

    return target


def get_detection_confidence(
    model,
    image_tensor,
    target_box,
    iou_threshold=0.5,
    imgsz=(960, 960),
    conf=0.001
):
    """
    Executa o YOLO na imagem modificada e procura
    a detecção que melhor corresponde à bounding box
    original.

    Retorna:

        confidence
        best_iou
    """

    predictions = model.predict(
        source=image_tensor,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    boxes = predictions[0].boxes

    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0

    target = target_box[:4]

    best_iou = 0.0
    best_confidence = 0.0

    for i in range(len(boxes)):

        box = (
            boxes.xyxy[i]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = (
            boxes.conf[i]
            .detach()
            .cpu()
            .item()
        )

        iou = calculate_box_iou(
            target,
            box
        )

        if iou > best_iou:

            best_iou = iou
            best_confidence = confidence

    if best_iou < iou_threshold:

        return 0.0, best_iou

    return best_confidence, best_iou