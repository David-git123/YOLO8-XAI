import cv2
import numpy as np
import torch
from ultralytics import YOLO

from easy_explain.methods.lrp.yolov8.yolo import YOLOv8LRP


MODEL_PATH = "weights/best.pt"
IMAGE_PATH = "image.jpg"

IMGSZ = 960

# Somente predições acima desse valor entram no EBPG.
CONF_THRESHOLD = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path, imgsz):
    """
    Carrega a imagem no formato utilizado pelo easy_explain.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (imgsz, imgsz)
    )

    tensor = torch.from_numpy(
        image_resized.transpose(2, 0, 1)
    ).float() / 255.0

    return image, tensor


def create_prediction_class_mask(
    boxes,
    classes,
    class_id,
    height,
    width
):
    """
    Cria uma única máscara para uma classe.

    Todas as bounding boxes preditas pelo modelo
    pertencentes à classe são unidas.

    Portanto:

        class 0:
            bbox 1
            bbox 2
            bbox 3

    gera UMA máscara para a classe 0.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    class_boxes = []

    for box, cls in zip(boxes, classes):

        if int(cls) != int(class_id):
            continue

        x1, y1, x2, y2 = box

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))

        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))

        if x2 <= x1 or y2 <= y1:
            continue

        mask[y1:y2, x1:x2] = 1.0

        class_boxes.append(
            [x1, y1, x2, y2]
        )

    return mask, class_boxes


def calculate_ebpg(
    explanation,
    mask
):
    """
    EBPG:

        energia dentro da região
        --------------------------
        energia total

    """

    if isinstance(
        explanation,
        torch.Tensor
    ):
        explanation = (
            explanation
            .detach()
            .cpu()
            .numpy()
        )

    explanation = np.squeeze(
        explanation
    )

    # O easy_explain retorna uma explicação
    # espacial. Usamos a magnitude da relevância.
    explanation = np.abs(
        explanation
    ).astype(np.float32)

    # Ajusta máscara ao tamanho da explicação
    if explanation.shape != mask.shape:

        mask = cv2.resize(
            mask,
            (
                explanation.shape[1],
                explanation.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    total_energy = np.sum(
        explanation
    )

    inside_energy = np.sum(
        explanation * mask
    )

    outside_energy = (
        total_energy -
        inside_energy
    )

    if total_energy <= 1e-12:
        return np.nan

    ebpg = (
        inside_energy /
        total_energy
    )

    return float(ebpg)


def main():

    # ======================================================
    # MODELO
    # ======================================================

    model = YOLO(
        MODEL_PATH
    )

    model.to(DEVICE)

    # ======================================================
    # LRP
    # ======================================================

    lrp = YOLOv8LRP(
        model=model,
        contrastive=False,
        power=1,
        positive=True,
        eps=1e-6,
        device=torch.device(
            DEVICE
        )
    )

    # ======================================================
    # IMAGEM
    # ======================================================

    original_image, frame = load_image(
        IMAGE_PATH,
        IMGSZ
    )

    frame = frame.to(
        DEVICE
    )

    height = frame.shape[1]
    width = frame.shape[2]

    # ======================================================
    # PREDIÇÕES DO MODELO
    # ======================================================

    results = model.predict(
        source=original_image,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
        device=DEVICE
    )

    result = results[0]

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    # ======================================================
    # MOSTRA PREDIÇÕES
    # ======================================================

    print("\nPredições utilizadas:")

    for i in range(len(boxes)):

        print(
            f"Detection {i}: "
            f"class={int(classes[i])}, "
            f"conf={confidences[i]:.4f}, "
            f"box={boxes[i]}"
        )

    # ======================================================
    # CLASSES PREDITAS
    # ======================================================

    predicted_classes = sorted(
        set(
            int(cls)
            for cls in classes
        )
    )

    # ======================================================
    # EBPG POR CLASSE
    # ======================================================

    all_results = []

    for class_id in predicted_classes:

        class_name = model.names[
            class_id
        ]

        print(
            f"\nClasse {class_id}: "
            f"{class_name}"
        )

        # --------------------------------------------------
        # Bounding boxes dessa classe
        # --------------------------------------------------

        class_mask, class_boxes = (
            create_prediction_class_mask(
                boxes=boxes,
                classes=classes,
                class_id=class_id,
                height=height,
                width=width
            )
        )

        print(
            f"Predições da classe: "
            f"{len(class_boxes)}"
        )

        # --------------------------------------------------
        # LRP DA CLASSE
        # --------------------------------------------------

        explanation = lrp.explain(
            frame,
            cls=class_id,
            conf=CONF_THRESHOLD,
            max_class_only=True,
            contrastive=False
        )

        # --------------------------------------------------
        # EBPG
        # --------------------------------------------------

        ebpg = calculate_ebpg(
            explanation,
            class_mask
        )

        print(
            f"EBPG = {ebpg:.6f}"
        )

        all_results.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_predictions": len(
                    class_boxes
                ),
                "ebpg": ebpg
            }
        )

    # ======================================================
    # RESULTADO FINAL
    # ======================================================

    print("\n" + "=" * 60)
    print("EBPG POR CLASSE")
    print("=" * 60)

    for item in all_results:

        print(
            f"{item['class_name']} "
            f"(class {item['class_id']}): "
            f"EBPG = {item['ebpg']:.6f} "
            f"| predictions = "
            f"{item['num_predictions']}"
        )


if __name__ == "__main__":
    main()