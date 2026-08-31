
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO
from easy_explain.methods.lrp.yolov8.yolo import YOLOv8LRP


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

MODEL_PATH = "best.pt"
IMAGE_PATH = "TubastraeaZoomOut.jpg"

IMGSZ = 640

# Somente predições com confiança >= 0.70
# serão utilizadas no EBPG.
CONF_THRESHOLD = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# CARREGAMENTO DA IMAGEM
# ==========================================================

def load_image(image_path, imgsz):
    """
    Carrega a imagem e a redimensiona para a resolução
    utilizada pelo YOLO/LRP.

    Retorna:
        image_resized_bgr:
            imagem BGR em imgsz x imgsz

        tensor:
            tensor [3, H, W] em [0, 1]
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {image_path}"
        )

    image_resized = cv2.resize(
        image,
        (imgsz, imgsz),
        interpolation=cv2.INTER_LINEAR
    )

    image_rgb = cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        image_rgb.transpose(2, 0, 1)
    ).float() / 255.0

    return image_resized, tensor


# ==========================================================
# MÁSCARA DAS PREDIÇÕES DE UMA CLASSE
# ==========================================================

def create_prediction_class_mask(
    boxes,
    classes,
    class_id,
    height,
    width
):
    """
    Cria uma única máscara para uma classe.

    Todas as bounding boxes PREDITAS pelo modelo
    pertencentes à classe são unidas.

    Exemplo:

        classe 0:
            bbox 1
            bbox 2
            bbox 3

    gera:

        máscara classe 0 =
            bbox1 U bbox2 U bbox3

    O ground truth não é utilizado.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    class_boxes = []

    for box, cls in zip(
        boxes,
        classes
    ):

        if int(cls) != int(class_id):
            continue

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            min(int(x1), width - 1)
        )

        y1 = max(
            0,
            min(int(y1), height - 1)
        )

        x2 = max(
            0,
            min(int(x2), width)
        )

        y2 = max(
            0,
            min(int(y2), height)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        mask[
            y1:y2,
            x1:x2
        ] = 1.0

        class_boxes.append(
            [x1, y1, x2, y2]
        )

    return mask, class_boxes


# ==========================================================
# CÁLCULO DO EBPG
# ==========================================================

def calculate_ebpg(
    explanation,
    mask
):
    """
    Calcula o EBPG utilizando a energia do mapa LRP.

    EBPG =

        energia dentro das predições
        ----------------------------
        energia total da explicação

    A região utilizada é formada pelas bounding boxes
    PREDITAS pelo modelo para a classe.

    Não utiliza ground truth.
    """

    # ------------------------------------------------------
    # Tensor -> NumPy
    # ------------------------------------------------------

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

    # Remove dimensões extras
    explanation = np.squeeze(
        explanation
    )

    # ------------------------------------------------------
    # Magnitude da relevância
    # ------------------------------------------------------

    explanation = np.abs(
        explanation
    ).astype(
        np.float32
    )

    # ------------------------------------------------------
    # Normalização
    # ------------------------------------------------------

    min_value = explanation.min()
    max_value = explanation.max()

    if max_value > min_value:

        explanation = (
            explanation - min_value
        ) / (
            max_value - min_value
        )

    else:

        explanation = np.zeros_like(
            explanation
        )

    # ------------------------------------------------------
    # Ajusta máscara ao mapa
    # ------------------------------------------------------

    if explanation.shape != mask.shape:

        mask = cv2.resize(
            mask,
            (
                explanation.shape[1],
                explanation.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # ------------------------------------------------------
    # Energia total
    # ------------------------------------------------------

    total_energy = np.sum(
        explanation
    )

    # ------------------------------------------------------
    # Energia dentro das predições
    # ------------------------------------------------------

    inside_energy = np.sum(
        explanation * mask
    )

    # ------------------------------------------------------
    # Energia fora das predições
    # ------------------------------------------------------

    outside_energy = (
        total_energy -
        inside_energy
    )

    # ------------------------------------------------------
    # Evita divisão por zero
    # ------------------------------------------------------

    if total_energy <= 1e-12:

        return (
            np.nan,
            inside_energy,
            outside_energy,
            total_energy
        )

    # ------------------------------------------------------
    # EBPG
    # ------------------------------------------------------

    ebpg = (
        inside_energy /
        total_energy
    )

    return (
        float(ebpg),
        float(inside_energy),
        float(outside_energy),
        float(total_energy)
    )


# ==========================================================
# VISUALIZAÇÃO DO EBPG
# ==========================================================

def visualize_ebpg(
    image,
    explanation,
    class_mask,
    class_boxes,
    class_id,
    class_name,
    ebpg,
    save_path=None
):
    """
    Visualiza o EBPG de uma classe.

    A visualização contém:

        1. Imagem + bounding boxes preditas
        2. Mapa LRP da classe
        3. Máscara das predições
        4. LRP sobreposto à imagem

    Todas as bounding boxes utilizadas são provenientes
    das predições do YOLO.

    Parameters
    ----------
    image : np.ndarray
        Imagem BGR na mesma resolução utilizada pelo LRP.

    explanation : np.ndarray ou torch.Tensor
        Mapa de relevância produzido pelo easy_explain.

    class_mask : np.ndarray
        Máscara formada pela união das bounding boxes
        preditas para a classe.

    class_boxes : list
        Bounding boxes preditas para a classe.

    class_id : int
        ID da classe.

    class_name : str
        Nome da classe.

    ebpg : float
        Valor do EBPG.

    save_path : str, optional
        Caminho para salvar a figura.
    """

    # ======================================================
    # PREPARA IMAGEM
    # ======================================================

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ======================================================
    # PREPARA EXPLICAÇÃO
    # ======================================================

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

    # Magnitude da relevância
    explanation = np.abs(
        explanation
    ).astype(
        np.float32
    )

    # ------------------------------------------------------
    # Normalização para visualização
    # ------------------------------------------------------

    min_value = explanation.min()
    max_value = explanation.max()

    if max_value > min_value:

        explanation_norm = (
            explanation - min_value
        ) / (
            max_value - min_value
        )

    else:

        explanation_norm = np.zeros_like(
            explanation
        )

    # ======================================================
    # AJUSTA MÁSCARA
    # ======================================================

    if class_mask.shape != explanation_norm.shape:

        class_mask = cv2.resize(
            class_mask,
            (
                explanation_norm.shape[1],
                explanation_norm.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # ======================================================
    # FIGURA
    # ======================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ======================================================
    # 1. IMAGEM + BOUNDING BOXES
    # ======================================================

    image_boxes = image_rgb.copy()

    for index, box in enumerate(
        class_boxes
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        # Bounding box
        cv2.rectangle(
            image_boxes,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        # Identificação da detecção
        cv2.putText(
            image_boxes,
            f"{class_name} #{index + 1}",
            (
                x1,
                max(y1 - 8, 15)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    axes[0].imshow(
        image_boxes
    )

    axes[0].set_title(
        f"Predições do modelo\n"
        f"{class_name} — "
        f"{len(class_boxes)} detecções"
    )

    axes[0].axis("off")

    # ======================================================
    # 2. MAPA LRP
    # ======================================================

    axes[1].imshow(
        explanation_norm,
        cmap="hot"
    )

    axes[1].set_title(
        f"Mapa LRP\n"
        f"Classe {class_id}: {class_name}"
    )

    axes[1].axis("off")

    # ======================================================
    # 3. MÁSCARA
    # ======================================================

    axes[2].imshow(
        class_mask,
        cmap="gray"
    )

    axes[2].set_title(
        f"Região das predições\n"
        f"{len(class_boxes)} bounding boxes"
    )

    axes[2].axis("off")

    # ======================================================
    # 4. LRP + PREDIÇÕES
    # ======================================================

    axes[3].imshow(
        image_rgb
    )

    axes[3].imshow(
        explanation_norm,
        cmap="hot",
        alpha=0.5
    )

    # Bounding boxes
    for box in class_boxes:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        rectangle = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2
        )

        axes[3].add_patch(
            rectangle
        )

    axes[3].set_title(
        f"LRP + Predições\n"
        f"EBPG = {ebpg:.4f}"
    )

    axes[3].axis("off")

    # ======================================================
    # TÍTULO
    # ======================================================

    fig.suptitle(
        f"EBPG — Classe {class_id}: {class_name}",
        fontsize=16
    )

    plt.tight_layout()

    # ======================================================
    # SALVAR
    # ======================================================

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(
            f"Visualização salva em: "
            f"{save_path}"
        )

    # ======================================================
    # MOSTRAR
    # ======================================================

    plt.show()

    plt.close(fig)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("EBPG - YOLOv8 LRP")
    print("=" * 70)

    print(
        f"Modelo: {MODEL_PATH}"
    )

    print(
        f"Imagem: {IMAGE_PATH}"
    )

    print(
        f"Confidence threshold: "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # ======================================================
    # MODELO YOLO
    # ======================================================

    model = YOLO(
        MODEL_PATH
    )

    model.to(
        DEVICE
    )

    print(
        "\nClasses do modelo:"
    )

    print(
        model.names
    )

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

    # ======================================================
    # EXTRAI PREDIÇÕES
    # ======================================================

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

    if len(boxes) == 0:

        print(
            "Nenhuma detecção acima do "
            f"threshold {CONF_THRESHOLD}."
        )

        return

    for i in range(
        len(boxes)
    ):

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
    # RESULTADOS
    # ======================================================

    all_results = []

    # ======================================================
    # LOOP POR CLASSE
    # ======================================================

    for class_id in predicted_classes:

        class_name = model.names[
            class_id
        ]

        print(
            "\n" + "-" * 60
        )

        print(
            f"Classe {class_id}: "
            f"{class_name}"
        )

        # ==================================================
        # MÁSCARA DA CLASSE
        # ==================================================

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

        # ==================================================
        # LRP DA CLASSE
        # ==================================================

        explanation = lrp.explain(
            frame,
            cls=class_id,
            conf=CONF_THRESHOLD,
            max_class_only=True,
            contrastive=False
        )

        # ==================================================
        # EBPG
        # ==================================================

        (
            ebpg,
            energy_inside,
            energy_outside,
            total_energy
        ) = calculate_ebpg(
            explanation,
            class_mask
        )

        print(
            f"EBPG = {ebpg:.6f}"
        )

        print(
            f"Energy inside = "
            f"{energy_inside:.6f}"
        )

        print(
            f"Energy outside = "
            f"{energy_outside:.6f}"
        )

        print(
            f"Total energy = "
            f"{total_energy:.6f}"
        )

        # ==================================================
        # VISUALIZAÇÃO
        # ==================================================

        save_path = (
            f"ebpg_class_{class_id}.png"
        )

        visualize_ebpg(
            image=original_image,
            explanation=explanation,
            class_mask=class_mask,
            class_boxes=class_boxes,
            class_id=class_id,
            class_name=class_name,
            ebpg=ebpg,
            save_path=save_path
        )

        # ==================================================
        # SALVA RESULTADO
        # ==================================================

        all_results.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_predictions": len(
                    class_boxes
                ),
                "ebpg": ebpg,
                "energy_inside": energy_inside,
                "energy_outside": energy_outside,
                "total_energy": total_energy
            }
        )

    # ======================================================
    # RESULTADO FINAL
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EBPG POR CLASSE"
    )

    print(
        "=" * 70
    )

    for item in all_results:

        print(
            f"{item['class_name']} "
            f"(class {item['class_id']}): "
            f"EBPG = {item['ebpg']:.6f} "
            f"| predictions = "
            f"{item['num_predictions']}"
        )


# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    main()

