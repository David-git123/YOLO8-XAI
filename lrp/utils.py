# utils.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from scipy import stats

from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.core import Stabilizer


# ============================================================
# CONFIGURAÇÕES PADRÃO
# ============================================================

DEFAULT_IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ============================================================

def load_image(image_path, img_size=DEFAULT_IMG_SIZE):
    """
    Carrega uma imagem e a converte para tensor BCHW.

    A imagem é redimensionada diretamente para img_size x img_size.

    Retorna
    -------
    image_bgr : np.ndarray
        Imagem redimensionada em BGR.

    tensor : torch.Tensor
        Tensor no formato [1, 3, H, W], normalizado para [0, 1].
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem: {image_path}"
        )

    image_bgr = cv2.resize(
        image_bgr,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        image_rgb
    ).permute(2, 0, 1).float() / 255.0

    tensor = tensor.unsqueeze(0)

    return image_bgr, tensor


# ============================================================
# CARREGAMENTO DO YOLO
# ============================================================

def load_yolov8(weights, device=None):
    """
    Carrega um modelo YOLOv8 da Ultralytics.

    Parameters
    ----------
    weights : str
        Caminho para o .pt.

    device : str, optional
        "cuda" ou "cpu".

    Returns
    -------
    model : ultralytics.YOLO
    device : str
    """

    from ultralytics import YOLO

    model = YOLO(weights)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model.to(device)

    model.model.eval()

    return model, device


# ============================================================
# INFERÊNCIA
# ============================================================

def predict_tensor(
    model,
    image_tensor,
    img_size=DEFAULT_IMG_SIZE,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device="cpu"
):
    """
    Executa a inferência da YOLO diretamente sobre o tensor.

    Isso evita diferenças de pré-processamento entre a imagem
    usada na explicação e a imagem utilizada pela YOLO.

    Retorna
    -------
    detections : list[dict]
    """

    image_tensor = image_tensor.to(device)

    results = model.predict(
        source=image_tensor,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    boxes = (
        result.boxes.xyxy
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

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
    )

    for i in range(len(boxes)):

        detections.append({
            "index": i,
            "box": boxes[i].astype(np.float32),
            "confidence": float(
                confidences[i]
            ),
            "class_id": int(
                classes[i]
            )
        })

    return detections


def get_predictions(
    model,
    image_path,
    img_size=DEFAULT_IMG_SIZE,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device="cpu"
):
    """
    Compatibilidade com o fluxo anterior.

    Carrega a imagem e executa a inferência.
    """

    _, image_tensor = load_image(
        image_path,
        img_size
    )

    return predict_tensor(
        model=model,
        image_tensor=image_tensor,
        img_size=img_size,
        conf=conf,
        iou=iou,
        device=device
    )


# ============================================================
# SAÍDA RAW DA YOLO
# ============================================================

def get_raw_predictions(
    model,
    image_tensor
):
    """
    Obtém a saída RAW da rede YOLO antes do NMS.

    Para uma YOLOv8 de uma classe, normalmente temos:

        [B, 5, N]

    onde:

        4 = bounding box
        1 = score da classe
    """

    model.model.eval()

    with torch.no_grad():

        output = model.model(
            image_tensor
        )

    if isinstance(output, (tuple, list)):

        output = output[0]

    if not torch.is_tensor(output):

        raise RuntimeError(
            "A saída RAW da YOLO não é um tensor."
        )

    return output


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):
    """
    Calcula IoU entre duas caixas [x1, y1, x2, y2].
    """

    x1 = max(
        float(box1[0]),
        float(box2[0])
    )

    y1 = max(
        float(box1[1]),
        float(box2[1])
    )

    x2 = min(
        float(box1[2]),
        float(box2[2])
    )

    y2 = min(
        float(box1[3]),
        float(box2[3])
    )

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
        max(
            0.0,
            float(box1[2]) - float(box1[0])
        )
        *
        max(
            0.0,
            float(box1[3]) - float(box1[1])
        )
    )

    area2 = (
        max(
            0.0,
            float(box2[2]) - float(box2[0])
        )
        *
        max(
            0.0,
            float(box2[3]) - float(box2[1])
        )
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CONVERSÃO XYWH -> XYXY
# ============================================================

def xywh_to_xyxy(boxes):
    """
    Converte caixas:

        [cx, cy, w, h]

    para:

        [x1, y1, x2, y2]
    """

    cx = boxes[:, 0]
    cy = boxes[:, 1]

    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2

    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack(
        [x1, y1, x2, y2],
        dim=1
    )


# ============================================================
# CORRESPONDÊNCIA NMS -> RAW
# ============================================================

def find_closest_raw_detection(
    raw_output,
    detection
):
    """
    Encontra a predição RAW mais próxima da detecção
    produzida pelo NMS.

    A correspondência considera:

        1. distância entre centros
        2. diferença de confiança

    Retorna
    -------
    int
        Índice da predição RAW.
    """

    if raw_output.ndim != 3:

        raise RuntimeError(
            f"Formato RAW inesperado: "
            f"{raw_output.shape}"
        )

    raw = raw_output[0]

    # --------------------------------------------------------
    # Esperado:
    #
    # [5, N]
    #
    # Convertemos para:
    #
    # [N, 5]
    # --------------------------------------------------------

    if raw.shape[0] <= 10:

        raw = raw.transpose(0, 1)

    elif raw.shape[1] <= 10:

        pass

    else:

        raise RuntimeError(
            "Não foi possível identificar "
            "o formato da saída RAW."
        )

    if raw.shape[1] < 5:

        raise RuntimeError(
            f"A saída RAW possui apenas "
            f"{raw.shape[1]} canais."
        )

    raw_boxes = raw[:, :4]
    raw_scores = raw[:, 4]

    raw_boxes_xyxy = xywh_to_xyxy(
        raw_boxes
    )

    target_box = torch.tensor(
        detection["box"],
        dtype=raw_boxes.dtype,
        device=raw_boxes.device
    )

    target_center_x = (
        target_box[0] +
        target_box[2]
    ) / 2

    target_center_y = (
        target_box[1] +
        target_box[3]
    ) / 2

    raw_center_x = (
        raw_boxes_xyxy[:, 0] +
        raw_boxes_xyxy[:, 2]
    ) / 2

    raw_center_y = (
        raw_boxes_xyxy[:, 1] +
        raw_boxes_xyxy[:, 3]
    ) / 2

    center_distance = (
        (raw_center_x - target_center_x) ** 2
        +
        (raw_center_y - target_center_y) ** 2
    )

    score_difference = torch.abs(
        raw_scores -
        detection["confidence"]
    )

    # Normalizamos a distância espacial.
    width = max(
        float(target_box[2] - target_box[0]),
        1.0
    )

    height = max(
        float(target_box[3] - target_box[1]),
        1.0
    )

    spatial_weight = (
        center_distance /
        (width * height)
    )

    metric = (
        spatial_weight +
        score_difference
    )

    index = torch.argmin(
        metric
    )

    return int(
        index.item()
    )


# ============================================================
# TARGET PARA LRP
# ============================================================

class YOLODetectionTarget(nn.Module):
    """
    Wrapper que transforma uma predição RAW individual
    da YOLO em um escalar.

    Para YOLOv8 de uma classe:

        [x, y, w, h, score]

    O score da classe é utilizado como alvo da explicação.
    """

    def __init__(
        self,
        model,
        detection_index
    ):
        super().__init__()

        self.model = model
        self.detection_index = (
            detection_index
        )

    def forward(self, x):
        with torch.inference_mode(False):
            output = self.model(x)

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        if not torch.is_tensor(output):

            raise RuntimeError(
                "A saída da YOLO não é um tensor."
            )

        if output.ndim != 3:

            raise RuntimeError(
                f"Formato não suportado: "
                f"{output.shape}"
            )

        # ----------------------------------------------------
        # [B, C, N]
        # ----------------------------------------------------

        if output.shape[1] <= 10:

            output = output.transpose(
                1,
                2
            )

        # ----------------------------------------------------
        # [B, N, C]
        # ----------------------------------------------------

        if self.detection_index >= output.shape[1]:

            raise IndexError(
                f"Índice RAW "
                f"{self.detection_index} "
                f"fora do intervalo."
            )

        detection = output[
            0,
            self.detection_index
        ]

        if detection.shape[0] < 5:

            raise RuntimeError(
                "A predição não possui "
                "4 coordenadas + score."
            )

        score = detection[4]

        return score.unsqueeze(0)


# ============================================================
# LRP
# ============================================================

def compute_lrp(
    model,
    image_tensor,
    raw_detection_index,
    epsilon=1e-6
):
    """
    Calcula LRP para uma única predição RAW da YOLOv8.
    """

    device = image_tensor.device

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Entrada normal de autograd
    # --------------------------------------------------------

    x = image_tensor.detach().clone().to(device)
    x.requires_grad_(True)

    # --------------------------------------------------------
    # Target que extrai somente o score da detecção desejada
    # --------------------------------------------------------

    class YOLOTarget(nn.Module):

        def __init__(self, model, detection_index):
            super().__init__()

            self.model = model
            self.detection_index = detection_index

        def forward(self, x):

            # IMPORTANTE:
            # garantir que esta execução não esteja em
            # torch.inference_mode()

            with torch.inference_mode(False):

                output = self.model(x)

            if isinstance(output, (tuple, list)):
                output = output[0]

            if not torch.is_tensor(output):
                raise RuntimeError(
                    "A saída da YOLO não é um tensor."
                )

            # YOLOv8 normalmente retorna [B, 4+nc, N]
            if output.ndim != 3:
                raise RuntimeError(
                    f"Formato RAW inesperado: {output.shape}"
                )

            # [B, C, N] -> [B, N, C]
            if output.shape[1] <= 10:
                output = output.transpose(1, 2)

            if self.detection_index >= output.shape[1]:
                raise IndexError(
                    f"Índice RAW {self.detection_index} "
                    f"fora do intervalo "
                    f"[0, {output.shape[1] - 1}]"
                )

            detection = output[
                0,
                self.detection_index
            ]

            if detection.shape[0] < 5:
                raise RuntimeError(
                    "A detecção não possui "
                    "4 coordenadas + score."
                )

            # Score da classe
            score = detection[4]

            # IMPORTANTE:
            # clone() cria um tensor normal fora de qualquer
            # inference tensor problemático.
            score = score.clone()

            return score.unsqueeze(0)

    target_model = YOLOTarget(
        model,
        raw_detection_index
    ).to(device)

    target_model.eval()

    # --------------------------------------------------------
    # Zennit
    # --------------------------------------------------------

    composite = EpsilonGammaBox(
        low=0.0,
        high=1.0,
        epsilon=Stabilizer(
            epsilon=epsilon,
            norm_scale=True
        )
    )

    # --------------------------------------------------------
    # LRP
    # --------------------------------------------------------

    with torch.inference_mode(False):

        with Gradient(
            model=target_model,
            composite=composite
        ) as attributor:

            output, relevance = attributor(
                x,
                torch.ones_like
            )

    return (
        relevance.detach(),
        output.detach()
    )
# ============================================================
# RELEVÂNCIA -> SALIÊNCIA
# ============================================================

def relevance_to_saliency(
    relevance,
    positive_only=True
):
    """
    Converte:

        [B, C, H, W]

    em:

        [H, W]

    agregando os canais RGB.

    Parameters
    ----------
    positive_only : bool

        Se True, considera apenas relevâncias positivas.
    """

    if positive_only:

        relevance = torch.relu(
            relevance
        )

    else:

        relevance = relevance.abs()

    saliency = relevance.sum(
        dim=1
    )

    saliency = saliency.squeeze(
        0
    )

    saliency = saliency.cpu().numpy()

    saliency = np.nan_to_num(
        saliency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    min_value = saliency.min()
    max_value = saliency.max()

    if (
        max_value -
        min_value
        < 1e-12
    ):

        return np.zeros_like(
            saliency,
            dtype=np.float32
        )

    saliency = (
        saliency - min_value
    ) / (
        max_value - min_value
    )

    return saliency.astype(
        np.float32
    )


# ============================================================
# EBPG
# ============================================================

def calculate_ebpg(
    saliency,
    box,
    original_width=None,
    original_height=None
):
    """
    Calcula EBPG para uma predição.

    Como a saliency e a box estão no mesmo espaço 640x640
    quando utilizadas pelo pipeline principal, as dimensões
    originais são opcionais.

    Fórmula:

        EBPG =
        relevância dentro da box
        -------------------------
        relevância total
    """

    saliency = np.asarray(
        saliency,
        dtype=np.float32
    )

    height, width = saliency.shape

    x1, y1, x2, y2 = box

    if (
        original_width is not None
        and original_height is not None
    ):

        x1 = (
            x1 /
            original_width *
            width
        )

        x2 = (
            x2 /
            original_width *
            width
        )

        y1 = (
            y1 /
            original_height *
            height
        )

        y2 = (
            y2 /
            original_height *
            height
        )

    x1 = int(
        np.clip(
            x1,
            0,
            width - 1
        )
    )

    x2 = int(
        np.clip(
            x2,
            0,
            width
        )
    )

    y1 = int(
        np.clip(
            y1,
            0,
            height - 1
        )
    )

    y2 = int(
        np.clip(
            y2,
            0,
            height
        )
    )

    if x2 <= x1 or y2 <= y1:

        return 0.0

    total_relevance = float(
        saliency.sum()
    )

    if total_relevance <= 1e-12:

        return 0.0

    box_relevance = float(
        saliency[
            y1:y2,
            x1:x2
        ].sum()
    )

    ebpg = (
        box_relevance /
        total_relevance
    )

    return float(
        np.clip(
            ebpg,
            0.0,
            1.0
        )
    )


# ============================================================
# MÁSCARA DE SALIÊNCIA
# ============================================================

def get_saliency_order(
    saliency
):
    """
    Retorna os pixels em ordem decrescente de relevância.
    """

    flat = saliency.reshape(-1)

    return np.argsort(
        flat
    )[::-1]


def create_pixel_mask(
    saliency,
    percentage
):
    """
    Cria uma máscara contendo os pixels mais relevantes.

    percentage:

        0.0 -> nenhum pixel

        1.0 -> todos os pixels
    """

    h, w = saliency.shape

    total_pixels = h * w

    number_pixels = int(
        total_pixels *
        percentage
    )

    order = get_saliency_order(
        saliency
    )

    mask = np.zeros(
        total_pixels,
        dtype=np.float32
    )

    if number_pixels > 0:

        mask[
            order[:number_pixels]
        ] = 1.0

    return mask.reshape(
        h,
        w
    )


# ============================================================
# LOCALIZAR DETECÇÃO CORRESPONDENTE
# ============================================================

def find_matching_detection(
    detections,
    target_box,
    iou_threshold=0.1
):
    """
    Procura a detecção que melhor corresponde à caixa-alvo.
    """

    best_detection = None
    best_iou = 0.0

    for detection in detections:

        iou = calculate_iou(
            target_box,
            detection["box"]
        )

        if iou > best_iou:

            best_iou = iou
            best_detection = detection

    if best_iou < iou_threshold:

        return None

    return best_detection


# ============================================================
# CONFIDENCE DA DETECÇÃO-ALVO
# ============================================================

def get_target_confidence(
    model,
    image_tensor,
    target_box,
    img_size=DEFAULT_IMG_SIZE,
    device="cpu",
    iou_threshold=0.1
):
    """
    Executa YOLO sobre uma imagem perturbada e procura
    a detecção correspondente à caixa original.

    Se a detecção desaparecer:

        retorna 0.0
    """

    detections = predict_tensor(
        model=model,
        image_tensor=image_tensor,
        img_size=img_size,
        conf=0.001,
        iou=DEFAULT_IOU,
        device=device
    )

    detection = find_matching_detection(
        detections,
        target_box,
        iou_threshold
    )

    if detection is None:

        return 0.0

    return float(
        detection["confidence"]
    )


# ============================================================
# PREPARAÇÃO DA SALIÊNCIA
# ============================================================

def prepare_saliency_for_image(
    saliency,
    image_tensor
):
    """
    Garante que o mapa de saliência tenha a mesma resolução
    espacial da imagem utilizada nos testes.
    """

    image_height = image_tensor.shape[-2]
    image_width = image_tensor.shape[-1]

    if saliency.shape == (
        image_height,
        image_width
    ):

        return saliency

    return cv2.resize(
        saliency,
        (
            image_width,
            image_height
        ),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# CONVERTER TENSOR -> NUMPY
# ============================================================

def tensor_to_numpy_image(
    image_tensor
):
    """
    Converte:

        [1, 3, H, W]

    para:

        [H, W, 3]
    """

    image = (
        image_tensor
        .detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    return image.astype(
        np.float32
    )


# ============================================================
# DELETION TEST
# ============================================================

def deletion_test(
    model,
    image_tensor,
    saliency,
    target_box,
    original_confidence,
    img_size=DEFAULT_IMG_SIZE,
    device="cpu",
    steps=20,
    baseline_value=0.0,
    iou_threshold=0.1
):
    """
    Deletion Test.

    Os pixels mais relevantes são removidos progressivamente.

    percentage = 0%
        imagem original

    percentage = 100%
        todos os pixels substituídos pelo baseline.

    Retorna
    -------
    dict
    """

    if steps < 1:

        raise ValueError(
            "steps deve ser >= 1."
        )

    saliency = prepare_saliency_for_image(
        saliency,
        image_tensor
    )

    original = tensor_to_numpy_image(
        image_tensor
    )

    height, width = original.shape[:2]

    flat_original = original.reshape(
        -1,
        3
    )

    order = get_saliency_order(
        saliency
    )

    percentages = np.linspace(
        0.0,
        1.0,
        steps + 1
    )

    confidences = []

    for percentage in percentages:

        perturbed = flat_original.copy()

        number_pixels = int(
            percentage *
            len(order)
        )

        if number_pixels > 0:

            selected_pixels = order[
                :number_pixels
            ]

            perturbed[
                selected_pixels
            ] = baseline_value

        perturbed = perturbed.reshape(
            height,
            width,
            3
        )

        tensor = torch.from_numpy(
            perturbed
        ).permute(
            2,
            0,
            1
        ).float()

        tensor = tensor.unsqueeze(
            0
        ).to(device)

        confidence = get_target_confidence(
            model=model,
            image_tensor=tensor,
            target_box=target_box,
            img_size=img_size,
            device=device,
            iou_threshold=iou_threshold
        )

        confidences.append(
            confidence
        )

    confidences = np.asarray(
        confidences,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = np.trapezoid(
        confidences,
        percentages
    )

    if original_confidence > 1e-12:

        normalized_auc = (
            auc /
            original_confidence
        )

    else:

        normalized_auc = 0.0

    # --------------------------------------------------------
    # Queda relativa de confiança
    # --------------------------------------------------------

    final_confidence = confidences[-1]

    confidence_drop = (
        original_confidence -
        final_confidence
    )

    relative_drop = (
        confidence_drop /
        original_confidence
        if original_confidence > 1e-12
        else 0.0
    )

    return {
        "percentages": percentages,
        "confidences": confidences,
        "auc": float(auc),
        "normalized_auc": float(
            normalized_auc
        ),
        "initial_confidence": float(
            original_confidence
        ),
        "final_confidence": float(
            final_confidence
        ),
        "confidence_drop": float(
            confidence_drop
        ),
        "relative_drop": float(
            relative_drop
        )
    }


# ============================================================
# INSERTION TEST
# ============================================================

def insertion_test(
    model,
    image_tensor,
    saliency,
    target_box,
    original_confidence,
    img_size=DEFAULT_IMG_SIZE,
    device="cpu",
    steps=20,
    baseline_value=0.0,
    iou_threshold=0.1
):
    """
    Insertion Test.

    Começa com uma imagem-base e adiciona progressivamente
    os pixels mais relevantes.

    Retorna
    -------
    dict
    """

    if steps < 1:

        raise ValueError(
            "steps deve ser >= 1."
        )

    saliency = prepare_saliency_for_image(
        saliency,
        image_tensor
    )

    original = tensor_to_numpy_image(
        image_tensor
    )

    height, width = original.shape[:2]

    flat_original = original.reshape(
        -1,
        3
    )

    order = get_saliency_order(
        saliency
    )

    percentages = np.linspace(
        0.0,
        1.0,
        steps + 1
    )

    confidences = []

    for percentage in percentages:

        perturbed = np.full_like(
            flat_original,
            baseline_value
        )

        number_pixels = int(
            percentage *
            len(order)
        )

        if number_pixels > 0:

            selected_pixels = order[
                :number_pixels
            ]

            perturbed[
                selected_pixels
            ] = flat_original[
                selected_pixels
            ]

        perturbed = perturbed.reshape(
            height,
            width,
            3
        )

        tensor = torch.from_numpy(
            perturbed
        ).permute(
            2,
            0,
            1
        ).float()

        tensor = tensor.unsqueeze(
            0
        ).to(device)

        confidence = get_target_confidence(
            model=model,
            image_tensor=tensor,
            target_box=target_box,
            img_size=img_size,
            device=device,
            iou_threshold=iou_threshold
        )

        confidences.append(
            confidence
        )

    confidences = np.asarray(
        confidences,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = np.trapezoid(
        confidences,
        percentages
    )

    if original_confidence > 1e-12:

        normalized_auc = (
            auc /
            original_confidence
        )

    else:

        normalized_auc = 0.0

    return {
        "percentages": percentages,
        "confidences": confidences,
        "auc": float(auc),
        "normalized_auc": float(
            normalized_auc
        ),
        "initial_confidence": float(
            confidences[0]
        ),
        "final_confidence": float(
            confidences[-1]
        )
    }


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calculate_statistics(
    values
):
    """
    Estatísticas descritivas para uma métrica.

    Calcula:

        N
        média
        mediana
        desvio padrão
        variância
        mínimo
        máximo
        Q1
        Q3
        IQR
        IC 95%
        Shapiro-Wilk
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {}

    n = len(values)

    mean = np.mean(
        values
    )

    median = np.median(
        values
    )

    if n > 1:

        std = np.std(
            values,
            ddof=1
        )

        variance = np.var(
            values,
            ddof=1
        )

    else:

        std = 0.0
        variance = 0.0

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    q1 = np.percentile(
        values,
        25
    )

    q3 = np.percentile(
        values,
        75
    )

    iqr = q3 - q1

    # --------------------------------------------------------
    # Intervalo de confiança de 95%
    # --------------------------------------------------------

    if n > 1:

        sem = stats.sem(
            values
        )

        ci_low, ci_high = stats.t.interval(
            0.95,
            df=n - 1,
            loc=mean,
            scale=sem
        )

    else:

        ci_low = mean
        ci_high = mean

    # --------------------------------------------------------
    # Shapiro-Wilk
    # --------------------------------------------------------

    if 3 <= n <= 5000:

        shapiro_stat, shapiro_p = (
            stats.shapiro(values)
        )

    else:

        shapiro_stat = np.nan
        shapiro_p = np.nan

    return {
        "n": int(n),

        "mean": float(mean),

        "median": float(median),

        "std": float(std),

        "variance": float(
            variance
        ),

        "min": float(
            minimum
        ),

        "max": float(
            maximum
        ),

        "q1": float(
            q1
        ),

        "q3": float(
            q3
        ),

        "iqr": float(
            iqr
        ),

        "ci95_low": float(
            ci_low
        ),

        "ci95_high": float(
            ci_high
        ),

        "shapiro_stat": float(
            shapiro_stat
        )
        if np.isfinite(shapiro_stat)
        else None,

        "shapiro_p": float(
            shapiro_p
        )
        if np.isfinite(shapiro_p)
        else None
    }


# ============================================================
# VISUALIZAÇÃO DO MAPA
# ============================================================

def create_heatmap(
    saliency,
    image
):
    """
    Cria:

        heatmap
        overlay

    """

    saliency = np.asarray(
        saliency,
        dtype=np.float32
    )

    saliency_uint8 = (
        np.clip(
            saliency,
            0,
            1
        ) * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        saliency_uint8,
        cv2.COLORMAP_JET
    )

    if heatmap.shape[:2] != image.shape[:2]:

        heatmap = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0]
            )
        )

    overlay = cv2.addWeighted(
        image,
        0.5,
        heatmap,
        0.5,
        0
    )

    return heatmap, overlay


# ============================================================
# DESENHAR DETECÇÃO
# ============================================================

def draw_detection(
    image,
    box,
    confidence,
    ebpg,
    index
):
    """
    Desenha bounding box, confiança e EBPG.
    """

    x1, y1, x2, y2 = map(
        int,
        box
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"#{index} "
        f"Conf: {confidence:.3f} "
        f"EBPG: {ebpg:.3f}"
    )

    text_y = max(
        y1 - 10,
        20
    )

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# PLOT DAS CURVAS DE INSERTION / DELETION
# ============================================================

def plot_evaluation_curves(
    deletion,
    insertion,
    save_path=None
):
    """
    Gera uma figura contendo as curvas de Deletion e Insertion.

    Não é necessário utilizar esta função para calcular as
    métricas; ela serve apenas para visualização.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        deletion["percentages"],
        deletion["confidences"],
        label="Deletion"
    )

    ax.plot(
        insertion["percentages"],
        insertion["confidences"],
        label="Insertion"
    )

    ax.set_xlabel(
        "Fraction of pixels"
    )

    ax.set_ylabel(
        "Detection confidence"
    )

    ax.set_title(
        "Deletion and Insertion"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    else:

        plt.show()


# ============================================================
# PROCESSAMENTO DE UMA IMAGEM
# ============================================================

def explain_image(
    model,
    image_path,
    img_size=DEFAULT_IMG_SIZE,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device="cpu",
    save_dir=None,
    run_deletion=True,
    run_insertion=True,
    perturbation_steps=20,
    baseline_value=0.0,
    matching_iou=0.1,
    lrp_epsilon=1e-6
):
    """
    Pipeline completo para uma imagem.

    Para cada predição:

        YOLO
          |
          +--> LRP
          |     |
          |     +--> mapa de saliência
          |
          +--> EBPG
          |
          +--> Deletion
          |
          +--> Insertion

    Retorna
    -------
    list[dict]
        Uma entrada para cada predição.
    """

    # ========================================================
    # 1. CARREGAR IMAGEM
    # ========================================================

    image_bgr, image_tensor = load_image(
        image_path,
        img_size
    )

    image_tensor = image_tensor.to(
        device
    )

    # ========================================================
    # 2. PREDIÇÕES NMS
    # ========================================================

    detections = predict_tensor(
        model=model,
        image_tensor=image_tensor,
        img_size=img_size,
        conf=conf,
        iou=iou,
        device=device
    )

    if len(detections) == 0:

        print(
            f"Nenhuma detecção encontrada: "
            f"{os.path.basename(image_path)}"
        )

        return []

    # ========================================================
    # 3. SAÍDA RAW
    # ========================================================

    raw_output = get_raw_predictions(
        model,
        image_tensor
    )

    results = []

    # ========================================================
    # 4. PROCESSAR CADA DETECÇÃO
    # ========================================================

    for detection in detections:

        detection_index = (
            detection["index"]
        )

        print(
            f"Detecção "
            f"{detection_index + 1}/"
            f"{len(detections)}"
        )

        # ----------------------------------------------------
        # Encontrar RAW correspondente
        # ----------------------------------------------------

        raw_index = find_closest_raw_detection(
            raw_output,
            detection
        )

        print(
            f"  NMS index: {detection_index}"
        )

        print(
            f"  RAW index: {raw_index}"
        )

        # ----------------------------------------------------
        # LRP
        # ----------------------------------------------------

        relevance, target_score = compute_lrp(
            model=model.model,
            image_tensor=image_tensor,
            raw_detection_index=raw_index,
            epsilon=lrp_epsilon
        )

        # ----------------------------------------------------
        # Saliency
        # ----------------------------------------------------

        saliency = relevance_to_saliency(
            relevance,
            positive_only=True
        )

        # ----------------------------------------------------
        # EBPG
        # ----------------------------------------------------

        ebpg = calculate_ebpg(
            saliency=saliency,
            box=detection["box"]
        )

        print(
            f"  Confidence: "
            f"{detection['confidence']:.4f}"
        )

        print(
            f"  Target score: "
            f"{float(target_score.flatten()[0]):.4f}"
        )

        print(
            f"  EBPG: "
            f"{ebpg:.4f}"
        )

        # ====================================================
        # DELETION
        # ====================================================

        if run_deletion:

            deletion = deletion_test(
                model=model,
                image_tensor=image_tensor,
                saliency=saliency,
                target_box=detection["box"],
                original_confidence=detection[
                    "confidence"
                ],
                img_size=img_size,
                device=device,
                steps=perturbation_steps,
                baseline_value=baseline_value,
                iou_threshold=matching_iou
            )

            print(
                f"  Deletion AUC: "
                f"{deletion['auc']:.4f}"
            )

            print(
                f"  Deletion normalized AUC: "
                f"{deletion['normalized_auc']:.4f}"
            )

        else:

            deletion = None

        # ====================================================
        # INSERTION
        # ====================================================

        if run_insertion:

            insertion = insertion_test(
                model=model,
                image_tensor=image_tensor,
                saliency=saliency,
                target_box=detection["box"],
                original_confidence=detection[
                    "confidence"
                ],
                img_size=img_size,
                device=device,
                steps=perturbation_steps,
                baseline_value=baseline_value,
                iou_threshold=matching_iou
            )

            print(
                f"  Insertion AUC: "
                f"{insertion['auc']:.4f}"
            )

            print(
                f"  Insertion normalized AUC: "
                f"{insertion['normalized_auc']:.4f}"
            )

        else:

            insertion = None

        # ====================================================
        # RESULTADO
        # ====================================================

        result = {
            "image": image_path,

            "detection_index":
                detection_index,

            "raw_index":
                raw_index,

            "box":
                detection["box"].tolist(),

            "confidence":
                detection["confidence"],

            "class_id":
                detection["class_id"],

            "target_score":
                float(
                    target_score.flatten()[0]
                    .detach()
                    .cpu()
                ),

            "saliency":
                saliency,

            "ebpg":
                ebpg,

            "deletion":
                deletion,

            "insertion":
                insertion
        }

        results.append(
            result
        )

        # ====================================================
        # SALVAR RESULTADOS VISUAIS
        # ====================================================

        if save_dir is not None:

            os.makedirs(
                save_dir,
                exist_ok=True
            )

            basename = os.path.splitext(
                os.path.basename(
                    image_path
                )
            )[0]

            index = detection_index

            # ------------------------------------------------
            # Heatmap
            # ------------------------------------------------

            heatmap, overlay = create_heatmap(
                saliency,
                image_bgr
            )

            overlay = draw_detection(
                overlay,
                detection["box"],
                detection["confidence"],
                ebpg,
                detection_index
            )

            heatmap_path = os.path.join(
                save_dir,
                f"{basename}_det_{index}_heatmap.png"
            )

            overlay_path = os.path.join(
                save_dir,
                f"{basename}_det_{index}_lrp.png"
            )

            cv2.imwrite(
                heatmap_path,
                heatmap
            )

            cv2.imwrite(
                overlay_path,
                overlay
            )

            # ------------------------------------------------
            # Curvas
            # ------------------------------------------------

            if (
                deletion is not None
                and insertion is not None
            ):

                curve_path = os.path.join(
                    save_dir,
                    f"{basename}_det_{index}_curves.png"
                )

                plot_evaluation_curves(
                    deletion,
                    insertion,
                    curve_path
                )

    return results


# ============================================================
# CONVERTER RESULTADOS PARA CSV
# ============================================================

def result_to_csv_dict(
    result
):
    """
    Converte uma predição para um dicionário adequado
    para pandas.DataFrame.
    """

    box = result["box"]

    deletion = result[
        "deletion"
    ]

    insertion = result[
        "insertion"
    ]

    return {
        "image":
            result["image"],

        "detection_index":
            result["detection_index"],

        "raw_index":
            result["raw_index"],

        "x1":
            box[0],

        "y1":
            box[1],

        "x2":
            box[2],

        "y2":
            box[3],

        "confidence":
            result["confidence"],

        "class_id":
            result["class_id"],

        "target_score":
            result["target_score"],

        "ebpg":
            result["ebpg"],

        "deletion_auc":
            deletion["auc"]
            if deletion is not None
            else np.nan,

        "deletion_normalized_auc":
            deletion["normalized_auc"]
            if deletion is not None
            else np.nan,

        "deletion_initial_confidence":
            deletion["initial_confidence"]
            if deletion is not None
            else np.nan,

        "deletion_final_confidence":
            deletion["final_confidence"]
            if deletion is not None
            else np.nan,

        "deletion_relative_drop":
            deletion["relative_drop"]
            if deletion is not None
            else np.nan,

        "insertion_auc":
            insertion["auc"]
            if insertion is not None
            else np.nan,

        "insertion_normalized_auc":
            insertion["normalized_auc"]
            if insertion is not None
            else np.nan,

        "insertion_initial_confidence":
            insertion["initial_confidence"]
            if insertion is not None
            else np.nan,

        "insertion_final_confidence":
            insertion["final_confidence"]
            if insertion is not None
            else np.nan
    }


# ============================================================
# ESTATÍSTICAS COMPLETAS DO EXPERIMENTO
# ============================================================

def calculate_experiment_statistics(
    results
):
    """
    Calcula estatísticas para:

        EBPG
        Deletion normalized AUC
        Insertion normalized AUC

    Parameters
    ----------
    results : list[dict]

    Returns
    -------
    dict
    """

    ebpg_values = []
    deletion_values = []
    insertion_values = []

    for result in results:

        if np.isfinite(
            result["ebpg"]
        ):

            ebpg_values.append(
                result["ebpg"]
            )

        if result["deletion"] is not None:

            value = result[
                "deletion"
            ]["normalized_auc"]

            if np.isfinite(value):

                deletion_values.append(
                    value
                )

        if result["insertion"] is not None:

            value = result[
                "insertion"
            ]["normalized_auc"]

            if np.isfinite(value):

                insertion_values.append(
                    value
                )

    return {
        "EBPG":
            calculate_statistics(
                ebpg_values
            ),

        "Deletion_normalized_AUC":
            calculate_statistics(
                deletion_values
            ),

        "Insertion_normalized_AUC":
            calculate_statistics(
                insertion_values
            )
    }   