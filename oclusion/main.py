IMAGE_PATH = "/content/assets/img/TubastraeaZoomOut.jpg"
MODEL_PATH = "assets/models/best.pt"

INPUT_SIZE = (960, 960)

# Comece pequeno para testar
N = 100

S = 16
P1 = 0.5

BATCH_SIZE = 10

# ---------------------------------------------
# Modelo
# ---------------------------------------------

model = Model(
    model_path=MODEL_PATH,
    input_size=INPUT_SIZE,
    conf=0.001
)

# ---------------------------------------------
# Imagem
# ---------------------------------------------

inp = load_img(
    IMAGE_PATH,
    INPUT_SIZE
)

print("Input:", inp.shape)

# Deve aparecer:
#
# Input: torch.Size([1, 3, 960, 960])

# ---------------------------------------------
# Gerar máscaras
# ---------------------------------------------

masks = generate_masks(
    N=N,
    s=S,
    p1=P1,
    input_size=INPUT_SIZE
)

print("Masks:", masks.shape)

# Deve aparecer:
#
# Masks: (100, 1, 960, 960)

# ---------------------------------------------
# D-RISE
# ---------------------------------------------

saliency, target = explain(
    model=model,
    inp=inp,
    masks=masks,
    detection_index=0,
    batch_size=BATCH_SIZE
)

print("Saliency:", saliency.shape)

# Deve aparecer:
#
# Saliency: (960, 960)
visualize_pipeline(
    inp,
    masks,
    saliency,
    num_masks=6,
    pipeline_path="pipeline.png",
    masks_path="masks.png",
    saliency_path="saliency_map.png"
)

visualize_detection_and_saliency(
    inp,
    saliency,
    target,
    save_path="detection_saliency.png"
)