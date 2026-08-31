import os
import numpy as np

from Model import Model

from preprocess import (
    load_img,
    generate_masks,
    explain,
    calculate_ebpg_all_boxes,
    calculate_ebpg_statistics,visualize_detection_and_saliency
)




# =========================================================
# CONFIGURAÇÃO
# =========================================================

IMAGE_PATH = "/content/TubastraeaZoomOut.jpg"
MODEL_PATH = "assets/models/best.pt"

INPUT_SIZE = (640, 640)

N = 100
S = 16
P1 = 0.5

BATCH_SIZE = 10


# =========================================================
# VERIFICAÇÃO DOS ARQUIVOS
# =========================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Imagem não encontrada: {IMAGE_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo não encontrado: {MODEL_PATH}"
    )


# =========================================================
# MODELO
# =========================================================

print("\n==========================================")
print("CARREGANDO MODELO")
print("==========================================")

model = Model(
    model_path=MODEL_PATH,
    input_size=INPUT_SIZE,
    conf=0.7
)


# =========================================================
# IMAGEM
# =========================================================

print("\n==========================================")
print("CARREGANDO IMAGEM")
print("==========================================")

inp = load_img(
    IMAGE_PATH,
    INPUT_SIZE
)

print("Input:", inp.shape)


# =========================================================
# GERAR MÁSCARAS
# =========================================================

print("\n==========================================")
print("GERANDO MÁSCARAS")
print("==========================================")

masks = generate_masks(
    N=N,
    s=S,
    p1=P1,
    input_size=INPUT_SIZE
)

print("Masks:", masks.shape)


# =========================================================
# DETECÇÕES DA IMAGEM ORIGINAL
# =========================================================

print("\n==========================================")
print("OBTENDO DETECÇÕES")
print("==========================================")

original_predictions = model.run_on_batch(
    inp
)

detections = original_predictions[0]

print(
    f"Número de detecções: {len(detections)}"
)


if len(detections) == 0:
    raise ValueError(
        "Nenhuma detecção encontrada na imagem."
    )


# =========================================================
# D-RISE / OCCLUSION
# =========================================================

print("\n==========================================")
print("EXECUTANDO D-RISE / OCCLUSION")
print("==========================================")

saliency_maps, detections = explain(
    model=model,
    inp=inp,
    masks=masks,
    batch_size=BATCH_SIZE
)

print(
    "Saliency maps:",
    saliency_maps.shape
)

print(
    "Detections:",
    detections.shape
)


# =========================================================
# VERIFICAÇÃO DE COMPATIBILIDADE
# =========================================================

if len(saliency_maps) != len(detections):

    raise ValueError(
        "Número de mapas de saliência diferente "
        "do número de detecções."
    )


# =========================================================
# EBPG
# =========================================================

print("\n==========================================")
print("CALCULANDO EBPG")
print("==========================================")

ebpg_results = calculate_ebpg_all_boxes(
    saliency_maps=saliency_maps,
    detections=detections
)


# =========================================================
# VISUALIZAÇÃO
# =========================================================

print("\n==========================================")
print("GERANDO VISUALIZAÇÕES")
print("==========================================")

for detection_index, detection in enumerate(detections):

    # Mapa correspondente à detecção
    saliency = saliency_maps[detection_index]

    output_path = (
        f"detection_{detection_index}_saliency.png"
    )

    visualize_detection_and_saliency(
        inp,
        saliency,
        detection,
        output_path
    )

    print(
        f"Detecção #{detection_index} | "
        f"Mapa: {saliency.shape} | "
        f"Mapa salvo em: {output_path}"
    )


# =========================================================
# RESULTADOS EBPG
# =========================================================

print("\n==========================================")
print("RESULTADOS EBPG")
print("==========================================")

for result in ebpg_results:

    print(
        f"Detecção #{result['detection_index']} | "
        f"Confidence: {result['confidence']:.4f} | "
        f"EBPG: {result['ebpg']:.4f}"
    )


# =========================================================
# ESTATÍSTICAS
# =========================================================

statistics = calculate_ebpg_statistics(
    ebpg_results
)


# =========================================================
# RESULTADO FINAL
# =========================================================

print("\n==========================================")
print("ESTATÍSTICAS EBPG")
print("==========================================")

print(
    f"Número de detecções: "
    f"{statistics['n']}"
)

print(
    f"EBPG médio: "
    f"{statistics['mean']:.4f}"
)

print(
    f"EBPG desvio-padrão: "
    f"{statistics['std']:.4f}"
)

print(
    f"EBPG ± DP: "
    f"{statistics['mean']:.4f} ± "
    f"{statistics['std']:.4f}"
)

print("\n==========================================")
print("FINALIZADO")
print("==========================================")