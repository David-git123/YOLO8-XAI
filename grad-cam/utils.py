import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def load_image(image_path, input_size=960, device="cuda"):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image_rgb.copy()

    resized = cv2.resize(
        image_rgb,
        (input_size, input_size)
    )

    tensor = torch.from_numpy(
        resized.astype(np.float32)
    )

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor / 255.0

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    return tensor, original_image


# ============================================================
# HOOKS DO GRAD-CAM
# ============================================================

class GradCAMHooks:

    def __init__(self, layer):

        self.activations = None
        self.gradients = None

        self.forward_handle = layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def remove(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# FORWARD
# ============================================================

def forward_model(model, x):

    output = model(x)

    if isinstance(output, tuple):
        output = output[0]

    return output


# ============================================================
# EXTRAIR SAÍDA DO YOLOv8
# ============================================================

def extract_predictions(output):
    """
    Extrai as bounding boxes e scores da saída bruta
    do YOLOv8.

    Para um modelo com uma única classe, a saída possui:

        [batch, 5, num_predictions]

    onde:

        predictions[0] -> x_center
        predictions[1] -> y_center
        predictions[2] -> width
        predictions[3] -> height
        predictions[4] -> class score

    As caixas são convertidas de:

        xywh

    para:

        xyxy
    """

    predictions = output[0]

    # --------------------------------------------------------
    # Garantir formato:
    #
    # [num_predictions, 5]
    # --------------------------------------------------------

    predictions = predictions.transpose(0, 1)

    # --------------------------------------------------------
    # Bounding boxes em XYWH
    # --------------------------------------------------------

    boxes_xywh = predictions[:, :4]

    # --------------------------------------------------------
    # Converter XYWH -> XYXY
    #
    # x1 = xc - w/2
    # y1 = yc - h/2
    # x2 = xc + w/2
    # y2 = yc + h/2
    # --------------------------------------------------------

    x_center = boxes_xywh[:, 0]
    y_center = boxes_xywh[:, 1]

    width = boxes_xywh[:, 2]
    height = boxes_xywh[:, 3]

    x1 = x_center - width / 2
    y1 = y_center - height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes_xyxy = torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=1
    )

    # --------------------------------------------------------
    # Score da única classe
    # --------------------------------------------------------

    class_scores = predictions[:, 4]

    return boxes_xyxy, class_scores
# ============================================================
# OBTER DETECÇÕES
# ============================================================

def get_detections(
    model,
    x,
    conf_threshold=0.25
):
    """
    Obtém as detecções da imagem.

    Como o modelo possui apenas uma classe, o score
    utilizado é diretamente predictions[:, 4].

    Retorna uma lista contendo:

        prediction_index
        box
        confidence

    O prediction_index identifica a mesma predição
    utilizada posteriormente pelo Grad-CAM,
    Deletion e Insertion.
    """

    # --------------------------------------------------------
    # Forward sem gradiente
    # --------------------------------------------------------

    with torch.no_grad():

        output = forward_model(
            model,
            x
        )

    # --------------------------------------------------------
    # Extrair boxes e scores
    # --------------------------------------------------------

    boxes, scores = extract_predictions(
        output
    )

    # --------------------------------------------------------
    # Filtrar pelo confidence threshold
    # --------------------------------------------------------

    valid_indices = torch.where(
        scores >= conf_threshold
    )[0]

    detections = []

    for idx in valid_indices:

        idx = idx.item()

        detection = {

            "prediction_index":
                idx,

            "box":
                boxes[idx]
                .detach()
                .cpu()
                .numpy(),

            "confidence":
                float(
                    scores[idx]
                    .detach()
                    .cpu()
                    .item()
                )
        }

        detections.append(
            detection
        )

    return detections

# ============================================================
# GRAD-CAM PARA UMA DETECÇÃO
# ============================================================

def generate_gradcam(
    model,
    x,
    hooks,
    prediction_index
):

    model.zero_grad(
        set_to_none=True
    )

    hooks.activations = None
    hooks.gradients = None
    with torch.enable_grad(): 
        output = forward_model(
            model,
            x
        )

        predictions = output[0]

        class_scores = predictions[4]

        # --------------------------------------------------------
        # TARGET DA DETECÇÃO
        # --------------------------------------------------------

        target = class_scores[
            prediction_index
        ]

    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------
    print("===== DEBUG GRAD-CAM =====")
    print("inference mode:",
      torch.is_inference_mode_enabled())

    print("target:",
      target)

    print("target requires_grad:",
      target.requires_grad)

    print("target grad_fn:",
      target.grad_fn)

    print("==========================")
    target.backward()

    if hooks.activations is None:

        raise RuntimeError(
            "As ativações não foram capturadas."
        )

    if hooks.gradients is None:

        raise RuntimeError(
            "Os gradientes não foram capturados."
        )

    activations = hooks.activations

    gradients = hooks.gradients

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    # --------------------------------------------------------
    # REDIMENSIONAR
    # --------------------------------------------------------

    cam = F.interpolate(
        cam,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0]

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    cam_min = cam.min()
    cam_max = cam.max()

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + 1e-8
    )

    return cam.detach().cpu().numpy()


# ============================================================
# CONVERTER BOX PARA O TAMANHO DO SALIENCY MAP
# ============================================================

def scale_box(
    box,
    original_size,
    target_size
):

    original_width, original_height = original_size

    target_width, target_height = target_size

    x1, y1, x2, y2 = box

    x1 = x1 * target_width / original_width
    x2 = x2 * target_width / original_width

    y1 = y1 * target_height / original_height
    y2 = y2 * target_height / original_height

    return np.array([
        x1,
        y1,
        x2,
        y2
    ])


# ============================================================
# EBPG
# ============================================================

def calculate_ebpg(
    saliency_map,
    box
):

    height, width = saliency_map.shape

    x1, y1, x2, y2 = box

    x1 = int(
        np.clip(
            x1,
            0,
            width - 1
        )
    )

    y1 = int(
        np.clip(
            y1,
            0,
            height - 1
        )
    )

    x2 = int(
        np.clip(
            x2,
            0,
            width
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

    total_energy = np.sum(
        saliency_map
    )

    if total_energy <= 0:
        return 0.0

    box_energy = np.sum(
        saliency_map[
            y1:y2,
            x1:x2
        ]
    )

    ebpg = (
        box_energy /
        total_energy
    )

    return float(ebpg)


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calculate_statistics(
    ebpg_values
):

    values = np.asarray(
        ebpg_values,
        dtype=np.float64
    )

    if len(values) == 0:

        return {}

    return {

        "n": int(len(values)),

        "mean": float(
            np.mean(values)
        ),

        "median": float(
            np.median(values)
        ),

        "std": float(
            np.std(
                values,
                ddof=1
            )
        ) if len(values) > 1 else 0.0,

        "min": float(
            np.min(values)
        ),

        "max": float(
            np.max(values)
        ),

        "q1": float(
            np.percentile(
                values,
                25
            )
        ),

        "q3": float(
            np.percentile(
                values,
                75
            )
        )
    }


# ============================================================
# SALVAR MAPA DE SALIÊNCIA
# ============================================================

def save_saliency_map(
    saliency_map,
    path
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        saliency_map,
        cmap="jet"
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


# ============================================================
# OVERLAY
# ============================================================

def save_overlay(
    image,
    saliency_map,
    box,
    detection_id,
    confidence,
    ebpg,
    path
):

    image = cv2.resize(
        image,
        (
            saliency_map.shape[1],
            saliency_map.shape[0]
        )
    )

    heatmap = (
        saliency_map * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        0.5,
        heatmap,
        0.5,
        0
    )

    x1, y1, x2, y2 = map(
        int,
        box
    )

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    plt.figure(
        figsize=(10, 10)
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"Detection {detection_id} | "
        f"Confidence: {confidence:.4f} | "
        f"EBPG: {ebpg:.4f}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SALVAR RESULTADOS JSON
# ============================================================

def save_results(
    results,
    statistics,
    path
):

    data = {

        "detections": [],

        "statistics": statistics
    }

    for result in results:

        data["detections"].append({

            "detection_id":
                result["detection_id"],

            "prediction_index":
                result["prediction_index"],

            "confidence":
                result["confidence"],

            "box":
                result["box"].tolist(),

            "ebpg":
                result["ebpg"]
        })

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )
def get_detection_confidence(
    model,
    image_tensor,
    prediction_index
):
    """
    Obtém o score bruto da classe para o mesmo
    prediction_index utilizado pelo Grad-CAM.

    Não executa NMS.
    """

    output = forward_model(
        model,
        image_tensor
    )

    predictions = output[0]

    # Modelo com apenas uma classe
    class_scores = predictions[4]

    score = class_scores[
        prediction_index
    ]

    return float(
        score.detach().cpu().item()
    )


# ============================================================
# RANKING DA SALIÊNCIA
# ============================================================

def prepare_saliency_order(
    saliency_map
):
    """
    Ordena os pixels do maior para o menor
    valor de saliência.

    Retorna coordenadas no formato:

        [[y, x],
         [y, x],
         ...]
    """

    height, width = saliency_map.shape

    flattened = saliency_map.reshape(-1)

    order = np.argsort(
        flattened
    )[::-1]

    coordinates = np.array(
        np.unravel_index(
            order,
            (height, width)
        )
    ).T

    return coordinates


# ============================================================
# MÁSCARA DE SALIÊNCIA
# ============================================================

def create_mask_from_saliency(
    saliency_map,
    fraction
):
    """
    Cria uma máscara contendo os pixels mais
    importantes segundo o Grad-CAM.

    fraction = 0.0
        nenhum pixel selecionado

    fraction = 0.5
        50% dos pixels mais importantes

    fraction = 1.0
        todos os pixels
    """

    height, width = saliency_map.shape

    total_pixels = height * width

    number_pixels = int(
        fraction * total_pixels
    )

    order = prepare_saliency_order(
        saliency_map
    )

    mask = np.zeros(
        total_pixels,
        dtype=np.float32
    )

    if number_pixels > 0:

        selected = order[
            :number_pixels
        ]

        flat_indices = (
            selected[:, 0] * width
            + selected[:, 1]
        )

        mask[
            flat_indices
        ] = 1.0

    mask = mask.reshape(
        height,
        width
    )

    return mask


# ============================================================
# BASELINE
# ============================================================

def create_baseline(
    image_tensor
):
    """
    Cria uma imagem baseline utilizando a média
    de cada canal da imagem original.

    Mantém o mesmo tamanho e device do input.
    """

    baseline = torch.zeros_like(
        image_tensor
    )

    for channel in range(
        image_tensor.shape[1]
    ):

        channel_mean = image_tensor[
            0,
            channel
        ].mean()

        baseline[
            0,
            channel
        ] = channel_mean

    return baseline


# ============================================================
# APLICAR MÁSCARA
# ============================================================

def apply_mask(
    image_tensor,
    mask,
    baseline
):
    """
    Combina imagem original e baseline.

    mask = 1:
        mantém pixel original

    mask = 0:
        utiliza baseline
    """

    mask_tensor = torch.from_numpy(
        mask
    ).to(
        device=image_tensor.device,
        dtype=image_tensor.dtype
    )

    mask_tensor = mask_tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    mask_tensor = mask_tensor.expand(
        image_tensor.shape[0],
        image_tensor.shape[1],
        -1,
        -1
    )

    modified_image = (
        image_tensor * mask_tensor
        +
        baseline * (1.0 - mask_tensor)
    )

    return modified_image


# ============================================================
# AUC
# ============================================================

def calculate_auc(
    fractions,
    scores
):
    """
    Calcula a área sob a curva utilizando
    a regra trapezoidal.
    """

    fractions = np.asarray(
        fractions,
        dtype=np.float64
    )

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    if len(fractions) != len(scores):

        raise ValueError(
            "fractions e scores devem "
            "possuir o mesmo tamanho."
        )

    if len(fractions) < 2:

        return 0.0

    return float(
        np.trapezoid(
            scores,
            fractions
        )
    )


# ============================================================
# DELETION TEST
# ============================================================

def deletion_test(
    model,
    image_tensor,
    saliency_map,
    prediction_index,
    steps=20
):
    """
    Deletion Test.

    Começa com a imagem original e remove
    progressivamente os pixels mais importantes.

    fraction:

        0.0 -> imagem original

        0.25 -> remove os 25% mais importantes

        0.50 -> remove os 50% mais importantes

        1.0 -> todos os pixels importantes removidos

    O score é sempre obtido para o mesmo
    prediction_index da detecção original.
    """

    baseline = create_baseline(
        image_tensor
    )

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1
    )

    scores = []

    for fraction in fractions:

        # ----------------------------------------------------
        # Quantidade de pixels que permanecerão
        # ----------------------------------------------------

        keep_fraction = (
            1.0 - fraction
        )

        mask = create_mask_from_saliency(
            saliency_map,
            keep_fraction
        )

        # ----------------------------------------------------
        # Imagem modificada
        # ----------------------------------------------------

        modified_image = apply_mask(
            image_tensor,
            mask,
            baseline
        )

        # ----------------------------------------------------
        # Score da MESMA detecção
        # ----------------------------------------------------

        score = get_detection_confidence(
            model,
            modified_image,
            prediction_index
        )

        scores.append(
            score
        )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = calculate_auc(
        fractions,
        scores
    )

    return {

        "fractions":
            fractions,

        "scores":
            scores,

        "auc":
            auc
    }


# ============================================================
# INSERTION TEST
# ============================================================

def insertion_test(
    model,
    image_tensor,
    saliency_map,
    prediction_index,
    steps=20
):
    """
    Insertion Test.

    Começa com a imagem baseline e adiciona
    progressivamente os pixels mais importantes.

    fraction:

        0.0 -> somente baseline

        0.25 -> 25% dos pixels mais importantes

        0.50 -> 50% dos pixels mais importantes

        1.0 -> imagem original

    O score é sempre obtido para o mesmo
    prediction_index da detecção original.
    """

    baseline = create_baseline(
        image_tensor
    )

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1
    )

    scores = []

    for fraction in fractions:

        # ----------------------------------------------------
        # Pixels mais importantes que serão inseridos
        # ----------------------------------------------------

        mask = create_mask_from_saliency(
            saliency_map,
            fraction
        )

        # ----------------------------------------------------
        # Imagem modificada
        # ----------------------------------------------------

        modified_image = apply_mask(
            image_tensor,
            mask,
            baseline
        )

        # ----------------------------------------------------
        # Score da MESMA detecção
        # ----------------------------------------------------

        score = get_detection_confidence(
            model,
            modified_image,
            prediction_index
        )

        scores.append(
            score
        )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = calculate_auc(
        fractions,
        scores
    )

    return {

        "fractions":
            fractions,

        "scores":
            scores,

        "auc":
            auc
    }


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calculate_test_statistics(
    values
):
    """
    Estatísticas das AUCs de Deletion ou Insertion.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:

        return {

            "n": 0,

            "mean": np.nan,

            "median": np.nan,

            "std": np.nan,

            "min": np.nan,

            "max": np.nan,

            "q1": np.nan,

            "q3": np.nan
        }

    return {

        "n":
            int(len(values)),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "min":
            float(
                np.min(values)
            ),

        "max":
            float(
                np.max(values)
            ),

        "q1":
            float(
                np.percentile(
                    values,
                    25
                )
            ),

        "q3":
            float(
                np.percentile(
                    values,
                    75
                )
            )
    }


# ============================================================
# PLOT DELETION + INSERTION
# ============================================================

def save_deletion_insertion_plot(
    deletion_result,
    insertion_result,
    detection_id,
    path
):
    """
    Salva as curvas de Deletion e Insertion
    para uma detecção específica.
    """

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        deletion_result["fractions"],
        deletion_result["scores"],
        label="Deletion"
    )

    plt.plot(
        insertion_result["fractions"],
        insertion_result["scores"],
        label="Insertion"
    )

    plt.xlabel(
        "Fraction of pixels"
    )

    plt.ylabel(
        "Detection confidence"
    )

    plt.title(
        f"Detection {detection_id} | "
        f"Deletion AUC = "
        f"{deletion_result['auc']:.4f} | "
        f"Insertion AUC = "
        f"{insertion_result['auc']:.4f}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()