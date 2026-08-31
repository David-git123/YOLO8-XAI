import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def letterbox_image(
    image: np.ndarray,
    imgsz: int = 640,
):
    """
    Reproduz o letterbox utilizado pelo YOLO.

    Retorna:
        image_lb : imagem letterboxed
        gain      : fator de escala
        pad       : padding (dw, dh)
    """

    h, w = image.shape[:2]

    gain = min(imgsz / h, imgsz / w)

    new_w = int(round(w * gain))
    new_h = int(round(h * gain))

    dw = imgsz - new_w
    dh = imgsz - new_h

    dw /= 2.0
    dh /= 2.0

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    image_lb = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return image_lb, gain, (dw, dh)


def xyxy_to_letterbox(
    box: np.ndarray,
    gain: float,
    pad,
):
    """
    Converte uma bounding box da imagem original
    para as coordenadas da imagem letterboxed.
    """

    dw, dh = pad

    box = np.asarray(box, dtype=np.float32).copy()

    box[0] = box[0] * gain + dw
    box[1] = box[1] * gain + dh
    box[2] = box[2] * gain + dw
    box[3] = box[3] * gain + dh

    return box


def prepare_input(
    image: np.ndarray,
    imgsz: int,
    device: torch.device,
):
    """
    Prepara a imagem para o forward diferenciável.
    """

    image_lb, gain, pad = letterbox_image(
        image,
        imgsz=imgsz,
    )

    rgb = cv2.cvtColor(
        image_lb,
        cv2.COLOR_BGR2RGB,
    )

    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1)
    ).float() / 255.0

    tensor = tensor.unsqueeze(0).contiguous()

    tensor = tensor.to(device)

    return tensor, image_lb, gain, pad


# ============================================================
# WRAPPER YOLOv8 EXPLICÁVEL
# ============================================================

class YOLOv8ExplainableWrapper(nn.Module):
    """
    Wrapper para executar o YOLOv8 em modo raw/training
    somente durante o Grad-CAM.

    A cabeça Detect não passa pelo _inference().
    """

    def __init__(self, yolo_model):
        super().__init__()

        self.yolo = yolo_model
        self.model = yolo_model.model

        self.detect_head = self.model.model[-1]

    def forward(self, x):

        # Guardamos o estado original.
        previous_state = self.detect_head.training

        # IMPORTANTE:
        #
        # Não fazemos:
        #
        # self.detect_head.train()
        #
        # porque isso colocaria também os módulos internos
        # em modo training.
        #
        # Alteramos somente o flag da cabeça.
        self.detect_head.training = True

        try:

            output = self.model._predict_once(
                x,
                profile=False,
                visualize=False,
                embed=None,
            )

        finally:

            self.detect_head.training = previous_state

        return output


# ============================================================
# EXTRAÇÃO DA SAÍDA RAW
# ============================================================

def get_one_to_many_predictions(output):
    """
    Obtém a saída one-to-many.

    Nas versões mais novas da Ultralytics, quando end2end está
    habilitado, o output possui:

        output["one2many"]
        output["one2one"]

    Para explicabilidade utilizamos one2many, pois ele permanece
    conectado ao backbone.
    """

    if isinstance(output, dict):

        if "one2many" in output:
            return output["one2many"]

        return output

    return output


def decode_boxes_from_raw(
    preds,
    detect_head,
):
    """
    Decodifica as bounding boxes da saída RAW.

    Compatível com a estrutura moderna do Detect da Ultralytics.
    """

    if not isinstance(preds, dict):
        raise RuntimeError(
            "A saída RAW do modelo não é um dicionário. "
            "Esta implementação foi preparada para a estrutura "
            "moderna do YOLOv8/Ultralytics. "
            f"Tipo recebido: {type(preds)}"
        )

    if "boxes" not in preds:
        raise RuntimeError(
            "A saída RAW não possui a chave 'boxes'. "
            f"Chaves encontradas: {list(preds.keys())}"
        )

    if "feats" not in preds:
        raise RuntimeError(
            "A saída RAW não possui a chave 'feats'."
        )

    # A própria implementação da Ultralytics realiza:
    #
    # anchors + DFL + dist2bbox
    #
    # sem passar pelo _inference().
    boxes = detect_head._get_decode_boxes(
        preds
    )

    return boxes


# ============================================================
# TARGET DO GRAD-CAM
# ============================================================

class YOLOBoxTarget:
    """
    Target específico para uma bounding box.

    A box original é utilizada para localizar, entre as
    predições RAW, o anchor cuja box decodificada apresenta
    maior IoU com a detecção que queremos explicar.

    O target final é o score da classe correspondente.
    """

    def __init__(
        self,
        box,
        class_id,
        detect_head,
    ):

        self.box = torch.tensor(
            box,
            dtype=torch.float32,
        )

        self.class_id = int(class_id)

        self.detect_head = detect_head

        self.selected_index = None
        self.selected_iou = None

    @staticmethod
    def box_iou_one_to_many(
        target_box,
        boxes,
    ):
        """
        target_box: [4]
        boxes: [N,4]
        """

        x1 = torch.maximum(
            target_box[0],
            boxes[:, 0],
        )

        y1 = torch.maximum(
            target_box[1],
            boxes[:, 1],
        )

        x2 = torch.minimum(
            target_box[2],
            boxes[:, 2],
        )

        y2 = torch.minimum(
            target_box[3],
            boxes[:, 3],
        )

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)

        intersection = inter_w * inter_h

        target_area = (
            (target_box[2] - target_box[0]).clamp(min=0)
            *
            (target_box[3] - target_box[1]).clamp(min=0)
        )

        boxes_area = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            *
            (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        )

        union = (
            target_area
            + boxes_area
            - intersection
            + 1e-7
        )

        return intersection / union

    def __call__(self, model_output):

        preds = get_one_to_many_predictions(
            model_output
        )

        if not isinstance(preds, dict):
            raise RuntimeError(
                "YOLOBoxTarget recebeu uma saída RAW incompatível."
            )

        scores = preds["scores"]

        # scores:
        #
        # [batch, classes, anchors]
        #
        if scores.ndim != 3:
            raise RuntimeError(
                f"Formato inesperado para scores: {scores.shape}"
            )

        boxes = decode_boxes_from_raw(
            preds,
            self.detect_head,
        )

        # boxes:
        #
        # [batch, 4, anchors]
        #
        if boxes.ndim != 3:
            raise RuntimeError(
                f"Formato inesperado para boxes: {boxes.shape}"
            )

        boxes = boxes[0].transpose(0, 1)

        target_box = self.box.to(
            device=boxes.device,
            dtype=boxes.dtype,
        )

        ious = self.box_iou_one_to_many(
            target_box,
            boxes,
        )

        best_index = torch.argmax(
            ious
        )

        self.selected_index = int(
            best_index.detach().cpu()
        )

        self.selected_iou = float(
            ious[best_index].detach().cpu()
        )

        if self.class_id >= scores.shape[1]:
            raise RuntimeError(
                f"class_id={self.class_id} excede "
                f"o número de classes={scores.shape[1]}"
            )

        # Score da classe.
        #
        # Aplicamos sigmoid para aproximar o confidence
        # utilizado pelo detector.
        score = scores[
            0,
            self.class_id,
            best_index,
        ]

        score = torch.sigmoid(score)

        return score


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    yolo_model,
    image,
    box,
    class_id,
    imgsz=640,
    target_layer_index=-2,
    device=None,
):
    """
    Gera um mapa Grad-CAM para UMA detecção.

    Retorna:

        saliency_map
        target_score
        matched_raw_index
        matched_iou
        input_tensor
        letterboxed_image
    """

    if device is None:

        device = next(
            yolo_model.model.parameters()
        ).device

    input_tensor, image_lb, gain, pad = prepare_input(
        image,
        imgsz=imgsz,
        device=device,
    )

    # Converter a box original para o espaço 640x640.
    box_lb = xyxy_to_letterbox(
        box,
        gain,
        pad,
    )

    wrapper = YOLOv8ExplainableWrapper(
        yolo_model
    )

    wrapper.to(device)
    wrapper.eval()

    # O target layer continua sendo o módulo ORIGINAL
    # pertencente ao modelo.
    target_layer = (
        yolo_model.model.model[
            target_layer_index
        ]
    )

    target = YOLOBoxTarget(
        box=box_lb,
        class_id=class_id,
        detect_head=wrapper.detect_head,
    )

    # GradCAM trabalha diretamente com o wrapper.
    with GradCAM(
        model=wrapper,
        target_layers=[target_layer],
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[target],
        )

    saliency_map = grayscale_cam[0]

    # Segurança numérica.
    saliency_map = np.maximum(
        saliency_map,
        0
    )

    max_value = saliency_map.max()

    if max_value > 0:
        saliency_map /= max_value

    return {
        "saliency": saliency_map,
        "target_score": float(
            target_score_from_mapless_cam(
                wrapper,
                input_tensor,
                target,
            )
        ),
        "raw_index": target.selected_index,
        "raw_iou": target.selected_iou,
        "input_tensor": input_tensor,
        "letterbox_image": image_lb,
        "letterbox_box": box_lb,
    }


def target_score_from_mapless_cam(
    wrapper,
    input_tensor,
    target,
):
    """
    Executa um forward separado para registrar o score
    do target.

    Não utiliza inference_mode.
    """

    wrapper.zero_grad(set_to_none=True)

    output = wrapper(
        input_tensor
    )

    score = target(output)

    return score.detach().cpu().item()


# ============================================================
# VISUALIZAÇÃO
# ============================================================

def save_gradcam_visualization(
    image,
    saliency,
    box,
    output_path,
    alpha=0.45,
):
    """
    Salva:

        imagem original + Grad-CAM + bounding box
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )

    visualization = show_cam_on_image(
        rgb_float,
        saliency,
        use_rgb=True,
    )

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR,
    )

    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    cv2.rectangle(
        visualization,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    cv2.imwrite(
        output_path,
        visualization,
    )