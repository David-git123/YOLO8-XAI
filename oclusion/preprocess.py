from PIL import Image
from torchvision.transforms import Resize, ToTensor
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_img(path,input_size):

    img = Image.open(path)

    resize = Resize(input_size)
    img = resize(img)   

    to_tensor = ToTensor()
    x = to_tensor(img)
    x   = x.unsqueeze(0)
    return x   







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
    detection_index=0,
    batch_size=100
):

    N = masks.shape[0]

    # ==================================================
    # 1. Obter as detecções da imagem ORIGINAL
    # ==================================================

    original_predictions = model.run_on_batch(inp)

    original_detections = original_predictions[0]

    if len(original_detections) == 0:
        raise ValueError(
            "O modelo não encontrou nenhuma detecção "
            "na imagem original."
        )

    if detection_index >= len(original_detections):
        raise ValueError(
            f"detection_index={detection_index}, "
            f"mas existem apenas "
            f"{len(original_detections)} detecções."
        )

    # Detecção que queremos explicar
    target = original_detections[detection_index]

    print("Detecção escolhida:")
    print("Box:", target[:4])
    print("Objectness:", target[4])
    print("Class probability:", target[5])


    # ==================================================
    # 2. Aplicar as máscaras
    # ==================================================

    masked = inp * masks


    # ==================================================
    # 3. Executar YOLO nas imagens mascaradas
    # ==================================================

    weights = np.zeros(
        N,
        dtype=np.float32
    )

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


        # ==============================================
        # 4. Para cada máscara, encontrar a melhor
        #    correspondência com a detecção original
        # ==============================================

        for j, proposals in enumerate(
            batch_predictions
        ):

            mask_index = i + j

            # Nenhuma detecção após aplicar a máscara
            if len(proposals) == 0:
                weights[mask_index] = 0
                continue

            best_similarity = 0.0

            for proposal in proposals:

                similarity = detection_similarity(
                    target,
                    proposal
                )

                if similarity > best_similarity:
                    best_similarity = similarity

            weights[mask_index] = best_similarity


    # ==================================================
    # 5. Soma ponderada das máscaras
    # ==================================================

    masks_flat = masks.reshape(
        N,
        -1
    )

    saliency = weights @ masks_flat

    saliency = saliency.reshape(
        *model.input_size
    )


    # ==================================================
    # 6. Normalização
    # ==================================================

    if saliency.max() > saliency.min():

        saliency = (
            saliency - saliency.min()
        ) / (
            saliency.max() - saliency.min()
        )

    return saliency





def visualize_pipeline(inp, masks, saliency, num_masks=6):
    """
    Visualiza:
    1. imagem original
    2. algumas máscaras
    3. imagens mascaradas
    4. saliency map
    """

    # ---------------------------------------------
    # Imagem original
    # ---------------------------------------------

    original = inp[0].detach().cpu().numpy()

    # (C,H,W) -> (H,W,C)
    original = np.transpose(original, (1, 2, 0))

    # ---------------------------------------------
    # Selecionar algumas máscaras
    # ---------------------------------------------

    num_masks = min(num_masks, masks.shape[0])

    indices = np.linspace(
        0,
        masks.shape[0] - 1,
        num_masks,
        dtype=int
    )

    # ---------------------------------------------
    # Criar figura
    # ---------------------------------------------

    fig, axes = plt.subplots(
        3,
        num_masks,
        figsize=(3 * num_masks, 9)
    )

    if num_masks == 1:
        axes = axes.reshape(3, 1)

    # ---------------------------------------------
    # Linha 1: imagem original
    # ---------------------------------------------

    for ax in axes[0]:
        ax.imshow(original)
        ax.axis("off")

    axes[0, 0].set_title("Imagem original")

    # ---------------------------------------------
    # Linha 2: máscaras
    # ---------------------------------------------

    for col, idx in enumerate(indices):

        mask = masks[idx]

        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()

        mask = np.squeeze(mask)

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

    # ---------------------------------------------
    # Linha 3: imagens mascaradas
    # ---------------------------------------------

    for col, idx in enumerate(indices):

        mask = masks[idx]

        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()

        mask = np.squeeze(mask)

        masked_image = original * mask[..., None]

        axes[2, col].imshow(
            masked_image
        )

        axes[2, col].set_title(
            f"Imagem × Máscara {idx}"
        )

        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------
    # Saliency map separado
    # ---------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(
        original,
        alpha=0.5
    )

    plt.imshow(
        saliency,
        cmap="jet",
        alpha=0.5
    )

    plt.title("D-RISE Saliency Map")
    plt.axis("off")
    plt.colorbar()

    plt.show()



def visualize_detection_and_saliency(
    inp,
    saliency,
    detection
):

    image = inp[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    box = detection[:4]
    confidence = detection[4]

    x1, y1, x2, y2 = box

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    # Imagem original
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
        f"Detecção — conf. {confidence:.2f}"
    )

    axes[0].axis("off")

    # Saliency
    axes[1].imshow(image, alpha=0.5)

    axes[1].imshow(
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

    plt.tight_layout()
    plt.show()