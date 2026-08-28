import os
import torch
import numpy as np

from ultralytics import YOLO

from YOLOv8_Explainer import (
    yolov8_heatmap,
    display_images
)

from utils import (
    get_gradcam_map,
    get_detections,
    calculate_ebpg_all_boxes,
    calculate_ebpg_statistics,
    visualize_ebpg
)


# =========================================================
# CONFIGURAÇÃO
# =========================================================

IMAGE_PATH = "/content/teste.png"
MODEL_PATH = "/content/best.pt"

INPUT_SIZE = 960
CONF_THRESHOLD = 0.4


# =========================================================
# VERIFICAÇÃO
# =========================================================

print(
    "Modelo encontrado:",
    os.path.exists(MODEL_PATH)
)

print(
    "Imagem encontrada:",
    os.path.exists(IMAGE_PATH)
)


# =========================================================
# PATCH DO TORCH.LOAD
# =========================================================

_original_torch_load = torch.load

torch.load = lambda *args, **kwargs: (
    _original_torch_load(
        *args,
        **{
            **kwargs,
            "weights_only": False
        }
    )
)


# =========================================================
# MODELO YOLO PARA DETECÇÕES
# =========================================================

model = YOLO(
    MODEL_PATH
)


# =========================================================
# CONFIGURAÇÃO DO GRAD-CAM
# =========================================================

model_cam = yolov8_heatmap(
    weight=MODEL_PATH,
    conf_threshold=CONF_THRESHOLD,
    method="GradCAM",
    layer=[
        10,
        12,
        14,
        16,
        18,
        -3
    ],
    ratio=0.02,
    show_box=True,
    renormalize=False
)

print(
    "\nGrad-CAM configurado."
)


# =========================================================
# GERAR VISUALIZAÇÃO ORIGINAL
# =========================================================

output_images = model_cam(
    IMAGE_PATH
)

display_images(
    output_images
)


# =========================================================
# OBTER MAPA GRAD-CAM NUMÉRICO
# =========================================================

saliency = get_gradcam_map(
    model_cam,
    IMAGE_PATH
)

print(
    "\nGrad-CAM:"
)

print(
    "Shape:",
    saliency.shape
)

print(
    "Min:",
    saliency.min()
)

print(
    "Max:",
    saliency.max()
)


# =========================================================
# OBTER TODAS AS DETECÇÕES
# =========================================================

detections = get_detections(
    image_path=IMAGE_PATH,
    model=model,
    input_size=INPUT_SIZE,
    conf_threshold=CONF_THRESHOLD
)

print(
    "\nNúmero de detecções:",
    len(detections)
)


# =========================================================
# CALCULAR EBPG
# =========================================================

ebpg_results = calculate_ebpg_all_boxes(
    saliency=saliency,
    detections=detections
)


# =========================================================
# MOSTRAR RESULTADOS INDIVIDUAIS
# =========================================================

print(
    "\n=========================================="
)

print(
    "EBPG POR DETECÇÃO"
)

print(
    "=========================================="
)

for result in ebpg_results:

    print(
        f"Detecção #{result['detection_index']} | "
        f"Classe: {result['class_id']} | "
        f"Confiança: {result['confidence']:.4f} | "
        f"EBPG: {result['ebpg']:.4f}"
    )


# =========================================================
# ESTATÍSTICAS
# =========================================================

statistics = calculate_ebpg_statistics(
    ebpg_results
)

print(
    "\n=========================================="
)

print(
    "ESTATÍSTICAS"
)

print(
    "=========================================="
)

print(
    f"Número de detecções: {statistics['n']}"
)

print(
    f"EBPG médio: {statistics['mean']:.4f}"
)

print(
    f"Desvio-padrão: {statistics['std']:.4f}"
)


# =========================================================
# VISUALIZAÇÃO
# =========================================================

visualize_ebpg(
    image_path=IMAGE_PATH,
    saliency=saliency,
    detections=detections,
    save_path="/content/ebpg_result.png"
)

print(
    "\nResultado salvo em:"
)

print(
    "/content/ebpg_result.png"
)