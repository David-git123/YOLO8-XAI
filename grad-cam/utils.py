import cv2
import numpy as np
import torch
import YOLOv8_Explainer


# =========================================================
# Grad-CAM
# =========================================================

def get_gradcam_map(model_cam, image_path):
    """
    Gera e retorna o mapa Grad-CAM numérico.

    Utiliza exatamente o mesmo pré-processamento
    empregado pelo YOLOv8_Explainer.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Não foi possível carregar a imagem: {image_path}"
        )

    # Mesmo pré-processamento do YOLOv8_Explainer
    img = YOLOv8_Explainer.letterbox(img)[0]

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = np.float32(img) / 255.0

    # HWC -> CHW
    tensor = (
        torch.from_numpy(
            np.transpose(
                img,
                axes=[2, 0, 1]
            )
        )
        .unsqueeze(0)
        .to(model_cam.device)
    )

    # Grad-CAM
    grayscale_cam = model_cam.method(
        tensor,
        [model_cam.target]
    )

    # Remover dimensão do batch
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


# =========================================================
# Detecções
# =========================================================

def get_detections(
    image_path,
    model,
    input_size=960,
    conf_threshold=0.4
):
    """
    Executa o YOLO e retorna todas as detecções.

    Formato:
        [x1, y1, x2, y2, confidence, class_id]
    """

    results = model.predict(
        source=image_path,
        imgsz=input_size,
        conf=conf_threshold,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return np.empty(
            (0, 6),
            dtype=np.float32
        )

    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confidence = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )

    class_id = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )

    detections = np.concatenate(
        [
            xyxy,
            confidence,
            class_id
        ],
        axis=1
    )

    return detections


# =========================================================
# EBPG individual
# =========================================================

def calculate_ebpg(
    saliency,
    bbox
):
    """
    Calcula o Energy-Based Pointing Game (EBPG).

    EBPG =
        energia dentro da bounding box /
        energia total do mapa Grad-CAM

    Parâmetros
    ----------
    saliency : np.ndarray
        Mapa Grad-CAM 2D.

    bbox : array-like
        Bounding box [x1, y1, x2, y2].

    Retorno
    -------
    float
        EBPG entre 0 e 1.
    """

    saliency = np.asarray(
        saliency,
        dtype=np.float32
    )

    if saliency.ndim != 2:
        raise ValueError(
            "O mapa Grad-CAM deve ser 2D."
        )

    # Grad-CAM deve representar energia positiva
    saliency = np.maximum(
        saliency,
        0
    )

    total_energy = np.sum(
        saliency
    )

    if total_energy <= 0:
        return 0.0

    height, width = saliency.shape

    x1, y1, x2, y2 = bbox

    # Converter para coordenadas válidas
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    x1 = max(
        0,
        min(x1, width)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Energia dentro da bounding box
    bbox_energy = np.sum(
        saliency[
            y1:y2,
            x1:x2
        ]
    )

    ebpg = (
        bbox_energy /
        total_energy
    )

    return float(ebpg)


# =========================================================
# EBPG para todas as caixas
# =========================================================

def calculate_ebpg_all_boxes(
    saliency,
    detections
):
    """
    Calcula EBPG para todas as bounding boxes.

    O mesmo mapa Grad-CAM é utilizado para todas
    as detecções.

    Retorna uma lista de dicionários.
    """

    results = []

    for i, detection in enumerate(
        detections
    ):

        bbox = detection[:4]

        confidence = float(
            detection[4]
        )

        class_id = int(
            detection[5]
        )

        ebpg = calculate_ebpg(
            saliency,
            bbox
        )

        results.append(
            {
                "detection_index": i,
                "bbox": bbox,
                "confidence": confidence,
                "class_id": class_id,
                "ebpg": ebpg
            }
        )

    return results


# =========================================================
# Estatísticas
# =========================================================

def calculate_ebpg_statistics(
    ebpg_results
):
    """
    Calcula média e desvio-padrão dos EBPGs.
    """

    if len(ebpg_results) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "n": 0
        }

    values = np.array(
        [
            result["ebpg"]
            for result in ebpg_results
        ],
        dtype=np.float32
    )

    mean = float(
        np.mean(values)
    )

    if len(values) > 1:
        std = float(
            np.std(
                values,
                ddof=1
            )
        )
    else:
        std = 0.0

    return {
        "mean": mean,
        "std": std,
        "n": len(values)
    }


# =========================================================
# Visualização opcional
# =========================================================

def visualize_ebpg(
    image_path,
    saliency,
    detections,
    save_path="ebpg_result.png"
):
    """
    Visualiza o Grad-CAM e todas as bounding boxes,
    mostrando o EBPG de cada detecção.
    """

    import matplotlib.pyplot as plt

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            f"Não foi possível carregar: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Redimensionar CAM para o tamanho da imagem
    saliency_resized = cv2.resize(
        saliency,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    # Normalização para visualização
    saliency_resized = (
        saliency_resized -
        saliency_resized.min()
    )

    max_value = saliency_resized.max()

    if max_value > 0:
        saliency_resized /= max_value

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(image)

    plt.imshow(
        saliency_resized,
        alpha=0.5,
        cmap="jet"
    )

    for i, detection in enumerate(
        detections
    ):

        bbox = detection[:4]

        ebpg = calculate_ebpg(
            saliency,
            bbox
        )

        x1, y1, x2, y2 = bbox

        plt.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            linewidth=2
        )

        plt.text(
            x1,
            y1,
            f"#{i} EBPG={ebpg:.3f}",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.7
            )
        )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()