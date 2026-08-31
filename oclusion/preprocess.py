from PIL import Image
from torchvision.transforms import Resize, ToTensor
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2

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
    batch_size=100,
    confidence_threshold = 0.7
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

    filtered_detections = original_detections[
        original_detections[:, 4] >= confidence_threshold
    ]

    if len(filtered_detections) == 0:
        raise ValueError(
            f"Nenhuma detecção possui confiança >= "
            f"{confidence_threshold:.2f}."
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


def create_class_prediction_mask(
    detections,
    class_id,
    image_shape
):
    """
    Cria uma única máscara formada pela união de todas
    as bounding boxes de uma determinada classe.

    Parameters
    ----------
    detections:
        Array contendo as detecções.

        Esperado:
            [x1, y1, x2, y2, confidence, class_id]

    class_id:
        Classe que será avaliada.

    image_shape:
        (height, width)

    Returns
    -------
    mask:
        Máscara binária da classe.

    class_boxes:
        Bounding boxes pertencentes à classe.
    """

    height, width = image_shape

    mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    class_boxes = []

    for detection in detections:

        x1, y1, x2, y2, conf, cls = detection

        if int(cls) != int(class_id):
            continue

        x1 = max(
            0,
            min(
                int(x1),
                width - 1
            )
        )

        y1 = max(
            0,
            min(
                int(y1),
                height - 1
            )
        )

        x2 = max(
            0,
            min(
                int(x2),
                width
            )
        )

        y2 = max(
            0,
            min(
                int(y2),
                height
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # União das bounding boxes
        mask[
            y1:y2,
            x1:x2
        ] = 1.0

        class_boxes.append(
            [x1, y1, x2, y2]
        )

    return mask, class_boxes


def aggregate_occlusion_by_class(
    saliency_maps,
    detections
):
    """
    Agrupa os mapas de oclusão por classe.

    Se existirem:

        classe 0:
            detecção 0
            detecção 1
            detecção 2

        classe 1:
            detecção 3

    será produzido:

        class_maps[0] =
            mapa0 + mapa1 + mapa2

        class_maps[1] =
            mapa3

    Parameters
    ----------
    saliency_maps:
        Array:

            [num_detections, height, width]

    detections:
        Array:

            [num_detections, 6]

            [x1, y1, x2, y2, confidence, class_id]

    Returns
    -------
    class_maps:
        Dicionário:

            {
                class_id: mapa
            }
    """

    class_maps = {}

    class_ids = (
        detections[:, 5]
        .astype(int)
    )

    unique_classes = sorted(
        np.unique(
            class_ids
        )
    )

    for class_id in unique_classes:

        indices = np.where(
            class_ids == class_id
        )[0]

        # Mapas das detecções dessa classe
        maps = saliency_maps[
            indices
        ]

        # Agregação
        class_map = np.sum(
            maps,
            axis=0
        )

        # Normalização
        min_value = class_map.min()
        max_value = class_map.max()

        if max_value > min_value:

            class_map = (
                class_map - min_value
            ) / (
                max_value - min_value
            )

        else:

            class_map = np.zeros_like(
                class_map
            )

        class_maps[
            int(class_id)
        ] = class_map

    return class_maps


def calculate_class_ebpg(
    saliency_map,
    class_mask
):
    """
    Calcula o EBPG para uma classe.

    EBPG =

        energia dentro das predições
        /
        energia total do mapa

    Parameters
    ----------
    saliency_map:
        Mapa de oclusão agregado da classe.

    class_mask:
        União das bounding boxes da classe.

    Returns
    -------
    ebpg
    energy_inside
    energy_outside
    total_energy
    """

    if isinstance(
        saliency_map,
        torch.Tensor
    ):

        saliency_map = (
            saliency_map
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        class_mask,
        torch.Tensor
    ):

        class_mask = (
            class_mask
            .detach()
            .cpu()
            .numpy()
        )

    saliency_map = np.squeeze(
        saliency_map
    )

    class_mask = np.squeeze(
        class_mask
    )

    saliency_map = np.abs(
        saliency_map
    ).astype(
        np.float32
    )

    # Garantir que mapa e máscara possuem
    # exatamente a mesma resolução
    if saliency_map.shape != class_mask.shape:

        class_mask = cv2.resize(
            class_mask,
            (
                saliency_map.shape[1],
                saliency_map.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # Energia total
    total_energy = np.sum(
        saliency_map
    )

    # Energia dentro das bounding boxes
    energy_inside = np.sum(
        saliency_map *
        class_mask
    )

    # Energia fora das bounding boxes
    energy_outside = (
        total_energy -
        energy_inside
    )

    if total_energy <= 1e-12:

        return (
            np.nan,
            energy_inside,
            energy_outside,
            total_energy
        )

    ebpg = (
        energy_inside /
        total_energy
    )

    return (
        float(ebpg),
        float(energy_inside),
        float(energy_outside),
        float(total_energy)
    )


def calculate_ebpg_by_class(
    saliency_maps,
    detections,
    image_shape
):
    """
    Calcula o EBPG para todas as classes presentes
    nas predições.

    O resultado é UM EBPG por classe.

    Não é calculado EBPG individual para cada
    bounding box.

    Returns
    -------
    results:
        Lista contendo os resultados.

    class_maps:
        Mapas de oclusão agregados por classe.

    class_masks:
        Máscaras das bounding boxes por classe.
    """

    class_maps = (
        aggregate_occlusion_by_class(
            saliency_maps,
            detections
        )
    )

    results = []

    class_masks = {}

    # Classes presentes
    class_ids = sorted(
        np.unique(
            detections[:, 5].astype(int)
        )
    )

    for class_id in class_ids:

        # --------------------------------------------
        # Mapa agregado da classe
        # --------------------------------------------

        class_map = class_maps[
            class_id
        ]

        # --------------------------------------------
        # Máscara das predições
        # --------------------------------------------

        class_mask, class_boxes = (
            create_class_prediction_mask(
                detections=detections,
                class_id=class_id,
                image_shape=image_shape
            )
        )

        class_masks[
            class_id
        ] = class_mask

        # --------------------------------------------
        # EBPG
        # --------------------------------------------

        (
            ebpg,
            energy_inside,
            energy_outside,
            total_energy
        ) = calculate_class_ebpg(
            saliency_map=class_map,
            class_mask=class_mask
        )

        results.append(
            {
                "class_id": int(class_id),
                "num_predictions": len(
                    class_boxes
                ),
                "ebpg": ebpg,
                "energy_inside": float(
                    energy_inside
                ),
                "energy_outside": float(
                    energy_outside
                ),
                "total_energy": float(
                    total_energy
                ),
                "boxes": class_boxes
            }
        )

    return (
        results,
        class_maps,
        class_masks
    )


def visualize_class_ebpg(
    image,
    class_map,
    class_mask,
    class_boxes,
    class_name,
    class_id,
    ebpg,
    save_path=None
):
    """
    Visualiza o EBPG de uma classe.

    Mostra:

        1. imagem + bounding boxes
        2. mapa de oclusão da classe
        3. máscara das predições
        4. overlay do mapa sobre a imagem
    """

    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    if isinstance(
        image,
        torch.Tensor
    ):

        image = (
            image
            .detach()
            .cpu()
        )

        if image.ndim == 4:
            image = image[0]

        image = (
            image
            .permute(1, 2, 0)
            .numpy()
        )

        if image.max() <= 1.0:

            image = (
                image * 255
            )

        image = image.astype(
            np.uint8
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    if image.shape[-1] == 3:

        image_rgb = image.copy()

    else:

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ========================================================
    # 1. IMAGEM + BOUNDING BOXES
    # ========================================================

    image_boxes = image_rgb.copy()

    for i, box in enumerate(
        class_boxes
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            image_boxes,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        cv2.putText(
            image_boxes,
            f"{class_name} #{i + 1}",
            (
                x1,
                max(
                    y1 - 5,
                    15
                )
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
        f"Predições — {class_name}\n"
        f"{len(class_boxes)} bounding boxes"
    )

    axes[0].axis("off")

    # ========================================================
    # 2. MAPA DE OCLUSÃO
    # ========================================================

    axes[1].imshow(
        class_map,
        cmap="hot"
    )

    axes[1].set_title(
        f"Mapa de oclusão\n"
        f"Classe {class_id}: {class_name}"
    )

    axes[1].axis("off")

    # ========================================================
    # 3. MÁSCARA
    # ========================================================

    axes[2].imshow(
        class_mask,
        cmap="gray"
    )

    axes[2].set_title(
        "Região das predições"
    )

    axes[2].axis("off")

    # ========================================================
    # 4. OVERLAY
    # ========================================================

    axes[3].imshow(
        image_rgb
    )

    axes[3].imshow(
        class_map,
        cmap="hot",
        alpha=0.5
    )

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
        f"Occlusion + Predições\n"
        f"EBPG = {ebpg:.4f}"
    )

    axes[3].axis("off")

    # ========================================================
    # TÍTULO
    # ========================================================

    fig.suptitle(
        f"EBPG por classe — "
        f"{class_name}",
        fontsize=16
    )

    plt.tight_layout()

    # ========================================================
    # SALVAR
    # ========================================================

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)

