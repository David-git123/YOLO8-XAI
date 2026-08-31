import cv2
import numpy as np
import torch


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b,
):
    """
    IoU entre duas boxes [x1,y1,x2,y2].
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# EBPG
# ============================================================

def calculate_ebpg(
    saliency,
    box,
):
    """
    Energy-Based Pointing Game.

    EBPG =
        energia dentro da box
        ---------------------
        energia total

    Saliency deve estar em [0,1].
    """

    saliency = np.asarray(
        saliency,
        dtype=np.float32,
    )

    saliency = np.abs(
        saliency
    )

    h, w = saliency.shape

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))

    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    total_energy = saliency.sum()

    if total_energy <= 1e-12:
        return 0.0

    box_energy = saliency[
        y1:y2,
        x1:x2
    ].sum()

    return float(
        box_energy / total_energy
    )


# ============================================================
# POINTING GAME OPCIONAL
# ============================================================

def calculate_pointing_game(
    saliency,
    box,
):
    """
    PG tradicional.

    Retorna:
        1 -> máximo dentro da box
        0 -> máximo fora
    """

    y, x = np.unravel_index(
        np.argmax(saliency),
        saliency.shape,
    )

    x1, y1, x2, y2 = box

    return float(
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ============================================================
# SCORE DA DETECÇÃO
# ============================================================

def get_matching_detection(
    model,
    image,
    target_box,
    target_class,
    conf=0.001,
    iou_nms=0.70,
    matching_iou=0.10,
):
    """
    Executa YOLO na imagem perturbada e encontra a detecção
    da mesma classe que possui maior IoU com a box original.

    Retorna:

        matched_score
        matched_box
        matched_iou
    """

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou_nms,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None:
        return 0.0, None, 0.0

    if len(result.boxes) == 0:
        return 0.0, None, 0.0

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    best_iou = 0.0
    best_score = 0.0
    best_box = None

    for box, score, cls in zip(
        boxes,
        scores,
        classes,
    ):

        if cls != target_class:
            continue

        current_iou = calculate_iou(
            target_box,
            box,
        )

        if current_iou > best_iou:

            best_iou = current_iou
            best_score = float(score)
            best_box = box.copy()

    if best_iou < matching_iou:
        return 0.0, None, best_iou

    return (
        best_score,
        best_box,
        best_iou,
    )


# ============================================================
# ORDENAÇÃO DA SALIÊNCIA
# ============================================================

def get_saliency_order(
    saliency,
):
    """
    Retorna os pixels em ordem decrescente de importância.
    """

    flat = np.abs(
        saliency
    ).reshape(-1)

    order = np.argsort(
        -flat
    )

    return order


# ============================================================
# MÁSCARA DE PERTURBAÇÃO
# ============================================================

def create_top_fraction_mask(
    saliency,
    fraction,
):
    """
    Cria máscara contendo os pixels mais importantes.

    fraction:
        0.0 -> nenhum pixel
        1.0 -> todos os pixels
    """

    h, w = saliency.shape

    total_pixels = h * w

    k = int(
        round(
            total_pixels * fraction
        )
    )

    if k <= 0:
        return np.zeros(
            (h, w),
            dtype=np.float32,
        )

    if k >= total_pixels:
        return np.ones(
            (h, w),
            dtype=np.float32,
        )

    order = get_saliency_order(
        saliency
    )

    mask_flat = np.zeros(
        total_pixels,
        dtype=np.float32,
    )

    mask_flat[
        order[:k]
    ] = 1.0

    return mask_flat.reshape(
        h,
        w
    )


# ============================================================
# PERTURBAÇÃO
# ============================================================

def apply_mask(
    image,
    mask,
    baseline=0.0,
):
    """
    Substitui os pixels selecionados pelo baseline.
    """

    image_float = image.astype(
        np.float32
    )

    mask3 = mask[..., None]

    result = (
        image_float * (1.0 - mask3)
        +
        baseline * mask3
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def insert_mask(
    image,
    mask,
    baseline=0.0,
):
    """
    Começa com baseline e insere os pixels selecionados.
    """

    image_float = image.astype(
        np.float32
    )

    mask3 = mask[..., None]

    result = (
        image_float * mask3
        +
        baseline * (1.0 - mask3)
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# AUC
# ============================================================

def calculate_auc(
    x,
    y,
):
    """
    Calcula AUC utilizando regra trapezoidal.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(y, x)
        )

    return float(
        np.trapz(y, x)
    )


# ============================================================
# DELETION
# ============================================================

def deletion_test(
    model,
    image,
    saliency,
    target_box,
    target_class,
    original_score,
    steps=20,
    baseline=0.0,
    iou_nms=0.70,
    matching_iou=0.10,
):
    """
    Deletion test.

    0%:
        imagem original

    100%:
        imagem completamente substituída pelo baseline.

    A cada passo são removidas as regiões de maior saliência.
    """

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    scores = []

    for fraction in fractions:

        mask = create_top_fraction_mask(
            saliency,
            fraction,
        )

        perturbed = apply_mask(
            image,
            mask,
            baseline=baseline,
        )

        score, _, _ = get_matching_detection(
            model=model,
            image=perturbed,
            target_box=target_box,
            target_class=target_class,
            conf=0.001,
            iou_nms=iou_nms,
            matching_iou=matching_iou,
        )

        scores.append(
            score
        )

    auc = calculate_auc(
        fractions,
        scores,
    )

    # Normalização opcional pelo score original.
    normalized_auc = (
        auc / original_score
        if original_score > 1e-12
        else 0.0
    )

    return {
        "fractions": fractions,
        "scores": np.asarray(
            scores,
            dtype=np.float32,
        ),
        "auc": float(auc),
        "normalized_auc": float(
            normalized_auc
        ),
    }


# ============================================================
# INSERTION
# ============================================================

def insertion_test(
    model,
    image,
    saliency,
    target_box,
    target_class,
    original_score,
    steps=20,
    baseline=0.0,
    iou_nms=0.70,
    matching_iou=0.10,
):
    """
    Insertion test.

    0%:
        imagem totalmente baseline

    100%:
        imagem original.

    As regiões são adicionadas da mais importante
    para a menos importante.
    """

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    scores = []

    for fraction in fractions:

        mask = create_top_fraction_mask(
            saliency,
            fraction,
        )

        perturbed = insert_mask(
            image,
            mask,
            baseline=baseline,
        )

        score, _, _ = get_matching_detection(
            model=model,
            image=perturbed,
            target_box=target_box,
            target_class=target_class,
            conf=0.001,
            iou_nms=iou_nms,
            matching_iou=matching_iou,
        )

        scores.append(
            score
        )

    auc = calculate_auc(
        fractions,
        scores,
    )

    normalized_auc = (
        auc / original_score
        if original_score > 1e-12
        else 0.0
    )

    return {
        "fractions": fractions,
        "scores": np.asarray(
            scores,
            dtype=np.float32,
        ),
        "auc": float(auc),
        "normalized_auc": float(
            normalized_auc
        ),
    }


# ============================================================
# SALVAR CURVAS
# ============================================================

def save_curve(
    x,
    y,
    path,
    title,
    ylabel,
):
    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        x * 100,
        y,
        marker="o",
    )

    plt.xlabel(
        "Perturbation (%)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()