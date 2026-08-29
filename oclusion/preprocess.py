from PIL import Image
from torchvision.transforms import Resize, ToTensor
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_img(path,input_size):

    img = Image.open(path).convert("RGB")

    resize = Resize(input_size)
    img = resize(img)   

    to_tensor = ToTensor()
    x = to_tensor(img)
    x   = x.unsqueeze(0)
    return x   



def generate_masks(N, s, p1, input_size):

    # Tamanho de cada célula da grade
    cell_size = np.ceil(
        np.array(input_size) / s
    ).astype(int)

    # Tamanho necessário para fazer o upsampling
    up_size = (
        (s + 1) * cell_size
    )

    # ---------------------------------------------
    # 1. Gerar grades aleatórias
    # ---------------------------------------------

    grid = (
        np.random.rand(N, s, s) < p1
    ).astype(np.float32)

    # ---------------------------------------------
    # 2. Espaço para as máscaras finais
    # ---------------------------------------------

    masks = np.empty(
        (N, *input_size),
        dtype=np.float32
    )

    # ---------------------------------------------
    # 3. Upsampling + deslocamento aleatório
    # ---------------------------------------------

    for i in tqdm(
        range(N),
        desc="Generating masks"
    ):

        # Deslocamento aleatório
        x = np.random.randint(
            0,
            cell_size[0]
        )

        y = np.random.randint(
            0,
            cell_size[1]
        )

        # Upsampling da grade
        upsampled = resize(
            grid[i],
            up_size,
            order=1,
            mode="reflect",
            anti_aliasing=False
        )

        # Crop para o tamanho da imagem
        masks[i] = upsampled[
            x:x + input_size[0],
            y:y + input_size[1]
        ]

    # ---------------------------------------------
    # 4. Adicionar dimensão de canal
    # ---------------------------------------------

    masks = masks.reshape(
        N,
        1,
        *input_size
    )

    return masks

def detection_similarity(target, proposal):

    # Bounding box da detecção original
    target_box = target[:4]

    # Bounding box da detecção após a máscara
    proposal_box = proposal[:4]

    # Similaridade espacial
    iou = box_iou(
        target_box,
        proposal_box
    )

    # Confiança da detecção após a máscara
    confidence = proposal[4]

    # Como existe apenas uma classe,
    # a similaridade entre classes é 1
    class_similarity = 1.0

    # Similaridade D-RISE
    similarity = (
        iou
        * class_similarity
        * confidence
    )

    return similarity



def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union





def explain(
    model,
    inp,
    masks,
    batch_size=100
):

    N = masks.shape[0]

    # ==================================================
    # 1. Detecções da imagem original
    # ==================================================

    original_predictions = model.run_on_batch(inp)
    original_detections = original_predictions[0]

    if len(original_detections) == 0:
        raise ValueError(
            "O modelo não encontrou nenhuma detecção."
        )

    D = len(original_detections)

    # ==================================================
    # 2. Aplicar as máscaras
    # ==================================================

    masked = inp * masks

    # ==================================================
    # 3. Matriz de pesos
    #
    # D = número de detecções
    # N = número de máscaras
    #
    # weights[d, m]
    # ==================================================

    weights = np.zeros(
        (D, N),
        dtype=np.float32
    )

    # ==================================================
    # 4. Executar YOLO nas imagens mascaradas
    # ==================================================

    for i in tqdm(
        range(0, N, batch_size),
        desc="Explaining"
    ):

        batch = masked[
            i:min(i + batch_size, N)
        ]

        batch_predictions = model.run_on_batch(
            batch
        )

        # ----------------------------------------------
        # Cada imagem mascarada
        # ----------------------------------------------

        for j, proposals in enumerate(
            batch_predictions
        ):

            mask_index = i + j

            if len(proposals) == 0:
                continue

            # ------------------------------------------
            # Para cada detecção original
            # ------------------------------------------

            for detection_index, target in enumerate(
                original_detections
            ):

                best_similarity = 0.0

                # --------------------------------------
                # Procurar a proposta que melhor
                # corresponde àquela detecção
                # --------------------------------------

                for proposal in proposals:

                    similarity = detection_similarity(
                        target,
                        proposal
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity

                weights[
                    detection_index,
                    mask_index
                ] = best_similarity

    # ==================================================
    # 5. Gerar um mapa para cada detecção
    # ==================================================

    masks_flat = masks.reshape(
        N,
        -1
    )

    saliency_maps = []

    for detection_index in range(D):

        detection_weights = weights[
            detection_index
        ]

        saliency = (
            detection_weights @ masks_flat
        )

        saliency = saliency.reshape(
            *model.input_size
        )

        # ----------------------------------------------
        # Normalização individual
        # ----------------------------------------------

        if saliency.max() > saliency.min():

            saliency = (
                saliency - saliency.min()
            ) / (
                saliency.max() - saliency.min()
            )

        saliency_maps.append(
            saliency
        )

    saliency_maps = np.stack(
        saliency_maps,
        axis=0
    )

    return saliency_maps, original_detections





def visualize_pipeline(
    inp,
    masks,
    saliency,
    num_masks=6,
    pipeline_path="pipeline.png",
    masks_path="masks.png",
    saliency_path="saliency_map.png"
):

    # ==================================================
    # Imagem original
    # ==================================================

    original = inp[0].detach().cpu().numpy()

    # (C, H, W) -> (H, W, C)
    original = np.transpose(
        original,
        (1, 2, 0)
    )

    # ==================================================
    # Selecionar máscaras
    # ==================================================

    num_masks = min(
        num_masks,
        masks.shape[0]
    )

    indices = np.linspace(
        0,
        masks.shape[0] - 1,
        num_masks,
        dtype=int
    )

    # ==================================================
    # 1. PIPELINE COMPLETO
    #
    # Linha 1: imagem original
    # Linha 2: máscaras
    # Linha 3: imagens mascaradas
    # ==================================================

    fig, axes = plt.subplots(
        3,
        num_masks,
        figsize=(3 * num_masks, 9)
    )

    if num_masks == 1:
        axes = axes.reshape(3, 1)

    # --------------------------------------------------
    # Imagem original
    # --------------------------------------------------

    for ax in axes[0]:

        ax.imshow(original)
        ax.axis("off")

    axes[0, 0].set_title(
        "Imagem original"
    )

    # --------------------------------------------------
    # Máscaras
    # --------------------------------------------------

    for col, idx in enumerate(indices):

        mask = masks[idx, 0]

        axes[1, col].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, col].set_title(
            f"Máscara {idx}"
        )

        axes[1, col].axis("off")

    # --------------------------------------------------
    # Imagens mascaradas
    # --------------------------------------------------

    for col, idx in enumerate(indices):

        mask = masks[idx, 0]

        masked_image = (
            original * mask[..., None]
        )

        axes[2, col].imshow(
            masked_image
        )

        axes[2, col].set_title(
            f"Imagem × Máscara {idx}"
        )

        axes[2, col].axis("off")

    plt.tight_layout()

    plt.savefig(
        pipeline_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ==================================================
    # 2. SOMENTE AS MÁSCARAS
    # ==================================================

    fig, axes = plt.subplots(
        1,
        num_masks,
        figsize=(3 * num_masks, 3)
    )

    if num_masks == 1:
        axes = [axes]

    for col, idx in enumerate(indices):

        mask = masks[idx, 0]

        axes[col].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[col].set_title(
            f"Máscara {idx}"
        )

        axes[col].axis("off")

    plt.tight_layout()

    plt.savefig(
        masks_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ==================================================
    # 3. SALIENCY MAP
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        original,
        alpha=0.5
    )

    im = ax.imshow(
        saliency,
        cmap="jet",
        alpha=0.5
    )

    ax.set_title(
        "D-RISE Saliency Map"
    )

    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax
    )

    plt.tight_layout()

    plt.savefig(
        saliency_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def visualize_detection_and_saliency(
    inp,
    saliency,
    detection,
    save_path="detection_saliency.png"
):

    # ==================================================
    # Imagem
    # ==================================================

    image = inp[0].detach().cpu().numpy()

    # (C, H, W) -> (H, W, C)
    image = np.transpose(
        image,
        (1, 2, 0)
    )

    # ==================================================
    # Dados da detecção
    # ==================================================

    x1, y1, x2, y2 = detection[:4]

    confidence = detection[4]

    # ==================================================
    # Criar figura
    # ==================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    # ==================================================
    # 1. Imagem original + bounding box
    # ==================================================

    axes[0].imshow(image)

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[0].add_patch(rect)

    axes[0].set_title(
        f"Detecção — Confidence: {confidence:.3f}"
    )

    axes[0].axis("off")

    # ==================================================
    # 2. Imagem + Saliency Map
    # ==================================================

    axes[1].imshow(
        image,
        alpha=0.5
    )

    im = axes[1].imshow(
        saliency,
        cmap="jet",
        alpha=0.6
    )

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[1].add_patch(rect)

    axes[1].set_title(
        "D-RISE Saliency Map"
    )

    axes[1].axis("off")

    # Colorbar
    fig.colorbar(
        im,
        ax=axes[1]
    )

    # ==================================================
    # Salvar
    # ==================================================

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)



def xai_box_iou(saliency, detection, percentile=80):

    # Bounding box
    x1, y1, x2, y2 = detection[:4]

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    # Máscara da bounding box
    box_mask = np.zeros(
        saliency.shape,
        dtype=bool
    )

    box_mask[y1:y2, x1:x2] = True

    # Máscara XAI
    threshold = np.percentile(
        saliency,
        percentile
    )

    xai_mask = saliency >= threshold

    # Interseção
    intersection = np.logical_and(
        xai_mask,
        box_mask
    ).sum()

    # União
    union = np.logical_or(
        xai_mask,
        box_mask
    ).sum()

    # IoU
    if union == 0:
        return 0.0

    return intersection / union


def visualize_xai_iou(
    inp,
    saliency,
    detection,
    percentile=80,
    save_path="xai_iou.png"
):

    image = inp[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    x1, y1, x2, y2 = detection[:4]

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    # Máscara XAI
    threshold = np.percentile(
        saliency,
        percentile
    )

    xai_mask = saliency >= threshold

    iou = xai_box_iou(
        saliency,
        detection,
        percentile
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(image)

    # Mostrar somente regiões XAI
    ax.imshow(
        np.ma.masked_where(
            ~xai_mask,
            saliency
        ),
        cmap="jet",
        alpha=0.6
    )

    # Bounding box
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    ax.add_patch(rect)

    ax.set_title(
        f"XAI × Bounding Box | IoU = {iou:.4f}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
# =========================================================
# EBPG
# =========================================================

def calculate_ebpg(
    saliency,
    detection
):
    """
    Calcula o Energy-Based Pointing Game (EBPG)
    para uma bounding box.

    EBPG =
        energia dentro da bounding box /
        energia total do mapa de saliência

    Parameters
    ----------
    saliency : np.ndarray
        Mapa de saliência 2D.

    detection : array-like
        Detecção no formato:

        [x1, y1, x2, y2, confidence]

    Returns
    -------
    float
        EBPG da detecção.
    """

    saliency = np.asarray(
        saliency,
        dtype=np.float32
    )

    if saliency.ndim != 2:
        raise ValueError(
            f"Saliency deve ser 2D. "
            f"Recebido: {saliency.shape}"
        )

    # -----------------------------------------------------
    # Garantir energia não negativa
    # -----------------------------------------------------

    saliency = np.maximum(
        saliency,
        0.0
    )

    # -----------------------------------------------------
    # Energia total
    # -----------------------------------------------------

    total_energy = np.sum(
        saliency
    )

    if total_energy <= 0:
        return 0.0

    # -----------------------------------------------------
    # Bounding box
    # -----------------------------------------------------

    x1, y1, x2, y2 = detection[:4]

    height, width = saliency.shape

    x1 = max(
        0,
        min(
            int(round(x1)),
            width
        )
    )

    x2 = max(
        0,
        min(
            int(round(x2)),
            width
        )
    )

    y1 = max(
        0,
        min(
            int(round(y1)),
            height
        )
    )

    y2 = max(
        0,
        min(
            int(round(y2)),
            height
        )
    )

    # Bounding box inválida
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # -----------------------------------------------------
    # Energia dentro da bounding box
    # -----------------------------------------------------

    bbox_energy = np.sum(
        saliency[
            y1:y2,
            x1:x2
        ]
    )

    # -----------------------------------------------------
    # EBPG
    # -----------------------------------------------------

    ebpg = (
        bbox_energy /
        total_energy
    )

    return float(
        ebpg
    )


# =========================================================
# EBPG para TODAS as bounding boxes
# =========================================================
def calculate_ebpg_all_boxes(
    saliency_maps,
    detections
):
    """
    Calcula EBPG individualmente para cada detecção.

    Parameters
    ----------
    saliency_maps : np.ndarray
        Mapas de saliência.

        Shape:
        (D, H, W)

        onde D é o número de detecções.

    detections : np.ndarray
        Detecções correspondentes.

        Shape:
        (D, 5)

        Formato:
        [x1, y1, x2, y2, confidence]

    Returns
    -------
    results : list
        Um resultado para cada detecção.
    """

    if len(detections) == 0:
        return []

    if len(saliency_maps) != len(detections):
        raise ValueError(
            "O número de mapas de saliência deve "
            "ser igual ao número de detecções. "
            f"Mapas: {len(saliency_maps)}, "
            f"Detecções: {len(detections)}"
        )

    results = []

    for i, detection in enumerate(detections):

        # =============================================
        # Mapa EXCLUSIVO desta detecção
        # =============================================

        saliency = saliency_maps[i]

        # =============================================
        # EBPG desta detecção
        # =============================================

        ebpg = calculate_ebpg(
            saliency=saliency,
            detection=detection
        )

        results.append(
            {
                "detection_index": i,

                "bbox": detection[:4].copy(),

                "confidence": float(
                    detection[4]
                ),

                "ebpg": float(ebpg)
            }
        )

    return results


# =========================================================
# Estatísticas do EBPG
# =========================================================

def calculate_ebpg_statistics(
    ebpg_results
):
    """
    Calcula média e desvio-padrão
    dos EBPGs das detecções.
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