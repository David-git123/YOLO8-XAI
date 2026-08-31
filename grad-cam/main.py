import os
import torch
import json

from ultralytics import YOLO

from utils import (
    load_image,
    GradCAMHooks,
    get_detections,
    generate_gradcam,
    calculate_ebpg,
    calculate_statistics,
    deletion_test,
    insertion_test,
    calculate_test_statistics,
    save_saliency_map,
    save_overlay,
    save_deletion_insertion_plot,
    save_results
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = "/content/best.pt"

IMAGE_PATH = "/content/teste.png"

INPUT_SIZE = 640

CONF_THRESHOLD = 0.7

TARGET_LAYER_INDEX = -2

STEPS = 20

OUTPUT_DIR = "/content/gradcam_results"

SALIENCY_DIR = os.path.join(
    OUTPUT_DIR,
    "saliency_maps"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

CURVES_DIR = os.path.join(
    OUTPUT_DIR,
    "deletion_insertion"
)

os.makedirs(
    SALIENCY_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)

os.makedirs(
    CURVES_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)


# ============================================================
# MODELO
# ============================================================

print(
    "\nCarregando modelo..."
)

yolo = YOLO(
    MODEL_PATH
)

model = yolo.model

model.to(
    device
)

model.eval()

print(
    "Modelo carregado."
)


# ============================================================
# CAMADA DO GRAD-CAM
# ============================================================

target_layer = model.model[
    TARGET_LAYER_INDEX
]

print(
    "\nCamada do Grad-CAM:"
)

print(
    f"Índice: {TARGET_LAYER_INDEX}"
)

print(
    target_layer
)


# ============================================================
# HOOKS
# ============================================================

hooks = GradCAMHooks(
    target_layer
)


# ============================================================
# IMAGEM
# ============================================================

x, original_image = load_image(
    IMAGE_PATH,
    input_size=INPUT_SIZE,
    device=device
)

print(
    "\nInput:",
    x.shape
)


# ============================================================
# DETECÇÕES
# ============================================================

detections = get_detections(
    model,
    x,
    conf_threshold=CONF_THRESHOLD
)

print(
    f"\nDetecções encontradas: "
    f"{len(detections)}"
)


# ============================================================
# RESULTADOS
# ============================================================

results = []

ebpg_values = []

deletion_auc_values = []

insertion_auc_values = []


# ============================================================
# PROCESSAR CADA DETECÇÃO
# ============================================================

for detection_id, detection in enumerate(
    detections
):

    prediction_index = (
        detection[
            "prediction_index"
        ]
    )

    box = detection["box"]

    confidence = (
        detection["confidence"]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DETECÇÃO {detection_id}"
    )

    print(
        f"Prediction index: "
        f"{prediction_index}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    saliency_map = generate_gradcam(
        model=model,
        x=x,
        hooks=hooks,
        prediction_index=prediction_index
    )


    # ========================================================
    # EBPG
    # ========================================================

    ebpg = calculate_ebpg(
        saliency_map,
        box
    )

    ebpg_values.append(
        ebpg
    )


    print(
        f"EBPG: {ebpg:.4f}"
    )


    # ========================================================
    # DELETION
    # ========================================================

    print(
        "Executando Deletion Test..."
    )

    deletion_result = deletion_test(
        model=model,
        image_tensor=x,
        saliency_map=saliency_map,
        prediction_index=prediction_index,
        steps=STEPS
    )

    deletion_auc = (
        deletion_result["auc"]
    )

    deletion_auc_values.append(
        deletion_auc
    )

    print(
        f"Deletion AUC: "
        f"{deletion_auc:.4f}"
    )


    # ========================================================
    # INSERTION
    # ========================================================

    print(
        "Executando Insertion Test..."
    )

    insertion_result = insertion_test(
        model=model,
        image_tensor=x,
        saliency_map=saliency_map,
        prediction_index=prediction_index,
        steps=STEPS
    )

    insertion_auc = (
        insertion_result["auc"]
    )

    insertion_auc_values.append(
        insertion_auc
    )

    print(
        f"Insertion AUC: "
        f"{insertion_auc:.4f}"
    )


    # ========================================================
    # SALIENCY MAP
    # ========================================================

    saliency_path = os.path.join(
        SALIENCY_DIR,
        f"detection_{detection_id}.png"
    )

    save_saliency_map(
        saliency_map,
        saliency_path
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay_path = os.path.join(
        OVERLAY_DIR,
        f"detection_{detection_id}.png"
    )

    save_overlay(
        image=original_image,
        saliency_map=saliency_map,
        box=box,
        detection_id=detection_id,
        confidence=confidence,
        ebpg=ebpg,
        path=overlay_path
    )


    # ========================================================
    # CURVAS
    # ========================================================

    curve_path = os.path.join(
        CURVES_DIR,
        f"detection_{detection_id}.png"
    )

    save_deletion_insertion_plot(
        deletion_result,
        insertion_result,
        detection_id,
        curve_path
    )


    # ========================================================
    # ARMAZENAR
    # ========================================================

    results.append({

        "detection_id":
            detection_id,

        "prediction_index":
            prediction_index,

        "confidence":
            confidence,

        "box":
            box,

        "ebpg":
            ebpg,

        "deletion": {

            "auc":
                deletion_auc,

            "fractions":
                deletion_result[
                    "fractions"
                ].tolist(),

            "scores":
                deletion_result[
                    "scores"
                ]
        },

        "insertion": {

            "auc":
                insertion_auc,

            "fractions":
                insertion_result[
                    "fractions"
                ].tolist(),

            "scores":
                insertion_result[
                    "scores"
                ]
        }
    })


# ============================================================
# ESTATÍSTICAS EBPG
# ============================================================

ebpg_statistics = calculate_statistics(
    ebpg_values
)


# ============================================================
# ESTATÍSTICAS DELETION
# ============================================================

deletion_statistics = calculate_test_statistics(
    deletion_auc_values
)


# ============================================================
# ESTATÍSTICAS INSERTION
# ============================================================

insertion_statistics = calculate_test_statistics(
    insertion_auc_values
)


# ============================================================
# RESULTADO FINAL
# ============================================================

final_statistics = {

    "EBPG":
        ebpg_statistics,

    "Deletion_AUC":
        deletion_statistics,

    "Insertion_AUC":
        insertion_statistics
}


print(
    "\n\n"
    + "=" * 70
)

print(
    "RESULTADOS ESTATÍSTICOS"
)

print(
    "=" * 70
)


# ============================================================
# EBPG
# ============================================================

print(
    "\nEBPG"
)

for key, value in (
    ebpg_statistics.items()
):

    print(
        f"{key:10s}: {value:.4f}"
        if isinstance(
            value,
            float
        )
        else
        f"{key:10s}: {value}"
    )


# ============================================================
# DELETION
# ============================================================

print(
    "\nDeletion AUC"
)

for key, value in (
    deletion_statistics.items()
):

    print(
        f"{key:10s}: {value:.4f}"
        if isinstance(
            value,
            float
        )
        else
        f"{key:10s}: {value}"
    )


# ============================================================
# INSERTION
# ============================================================

print(
    "\nInsertion AUC"
)

for key, value in (
    insertion_statistics.items()
):

    print(
        f"{key:10s}: {value:.4f}"
        if isinstance(
            value,
            float
        )
        else
        f"{key:10s}: {value}"
    )


# ============================================================
# SALVAR JSON
# ============================================================

output = {

    "image":
        IMAGE_PATH,

    "num_detections":
        len(detections),

    "detections":
        results,

    "statistics":
        final_statistics
}


json_path = os.path.join(
    OUTPUT_DIR,
    "results.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# REMOVER HOOKS
# ============================================================

hooks.remove()


print(
    "\n"
    + "=" * 70
)

print(
    "Experimento concluído."
)

print(
    f"Resultados: {OUTPUT_DIR}"
)

print(
    f"JSON: {json_path}"
)
# ============================================================
# OBTER SCORE DA DETECÇÃO
# ============================================================

