import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO

from preprocess import (
    load_img,
    generate_masks
)

from Model import Model

from ebpg_occlusion import (
    calculate_ebpg_by_class,
    visualize_class_ebpg
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "best.pt"

IMAGE_PATH = "TubastraeaZoomOut.jpg"

IMGSZ = 640

# Somente predições acima deste valor serão utilizadas
# no cálculo do EBPG.
CONF_THRESHOLD = 0.70

# Parâmetros da oclusão
N = 100
S = 16
P1 = 0.5

# Batch utilizado para executar as imagens mascaradas
BATCH_SIZE = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RESULTS_DIR = "results_ebpg"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCCLUSION — EBPG POR CLASSE")
    print("=" * 70)

    print(f"Modelo:       {MODEL_PATH}")
    print(f"Imagem:       {IMAGE_PATH}")
    print(f"Image size:   {IMGSZ}")
    print(f"Confidence:   {CONF_THRESHOLD}")
    print(f"N máscaras:   {N}")
    print(f"S:            {S}")
    print(f"P1:           {P1}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Device:       {DEVICE}")

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. CARREGAR YOLO
    # ========================================================

    print("\nCarregando modelo...")

    model = YOLO(
        MODEL_PATH
    )

    model.to(
        DEVICE
    )

    # ========================================================
    # 2. PREDIÇÕES ORIGINAIS
    # ========================================================

    print("\nExecutando predição original...")

    results = model.predict(
        source=IMAGE_PATH,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:

        raise RuntimeError(
            "O modelo não encontrou nenhuma "
            "predição acima do confidence threshold."
        )

    # ========================================================
    # 3. EXTRAIR PREDIÇÕES
    # ========================================================

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

    # --------------------------------------------------------
    # Monta:
    #
    # [x1, y1, x2, y2, confidence, class_id]
    # --------------------------------------------------------

    detections = np.column_stack(
        [
            boxes,
            confidences,
            classes
        ]
    ).astype(
        np.float32
    )

    print(
        f"\nDetecções encontradas: "
        f"{len(detections)}"
    )

    # ========================================================
    # 4. MOSTRAR PREDIÇÕES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDIÇÕES UTILIZADAS"
    )

    print(
        "=" * 70
    )

    for i, detection in enumerate(
        detections
    ):

        x1, y1, x2, y2, conf, cls = (
            detection
        )

        class_id = int(
            cls
        )

        class_name = model.names[
            class_id
        ]

        print(
            f"Detection {i}: "
            f"class={class_id} "
            f"({class_name}) | "
            f"confidence={conf:.4f} | "
            f"box=["
            f"{x1:.1f}, "
            f"{y1:.1f}, "
            f"{x2:.1f}, "
            f"{y2:.1f}"
            f"]"
        )

    # ========================================================
    # 5. CARREGAR IMAGEM PARA O EASY EXPLAIN
    # ========================================================

    print(
        "\nCarregando imagem..."
    )

    inp = load_img(
        IMAGE_PATH,
        (IMGSZ, IMGSZ)
    )

    inp = inp.to(
        DEVICE
    )

    print(
        f"Tensor: {tuple(inp.shape)}"
    )

    # ========================================================
    # 6. GERAR MÁSCARAS
    # ========================================================

    print(
        "\nGerando máscaras de oclusão..."
    )

    masks = generate_masks(
        N=N,
        s=S,
        p1=P1,
        input_size=(IMGSZ, IMGSZ)
    )

    print(
        f"Masks shape: {masks.shape}"
    )

    # ========================================================
    # 7. MODELO UTILIZADO PELO YOLO8-XAI
    # ========================================================

    print(
        "\nInicializando modelo de oclusão..."
    )

    xai_model = Model(
        model=model,
        input_size=(IMGSZ, IMGSZ),
        conf=0.001
    )

    # ========================================================
    # 8. DETECÇÕES ORIGINAIS DO MODELO XAI
    # ========================================================

    original_predictions = (
        xai_model.run_on_batch(
            inp
        )
    )

    original_detections = (
        original_predictions[0]
    )

    if len(original_detections) == 0:

        raise RuntimeError(
            "O modelo utilizado pelo método de "
            "oclusão não produziu detecções."
        )

    # ========================================================
    # 9. FILTRAR CONFIDENCE
    # ========================================================

    original_detections = (
        original_detections[
            original_detections[:, 4]
            >= CONF_THRESHOLD
        ]
    )

    if len(original_detections) == 0:

        raise RuntimeError(
            f"Nenhuma detecção acima de "
            f"{CONF_THRESHOLD}."
        )

    print(
        "\nDetecções utilizadas no EBPG:"
    )

    for i, detection in enumerate(
        original_detections
    ):

        x1, y1, x2, y2, conf, cls = (
            detection
        )

        class_id = int(
            cls
        )

        class_name = model.names[
            class_id
        ]

        print(
            f"{i}: "
            f"{class_name} "
            f"(class {class_id}) "
            f"conf={conf:.4f}"
        )

    # ========================================================
    # 10. EXECUTAR O MÉTODO DE OCLUSÃO
    # ========================================================
    #
    # Aqui utilizamos a lógica do método do
    # YOLO8-XAI para produzir:
    #
    #     saliency_maps
    #
    # um mapa para cada detecção.
    #
    # ========================================================

    print(
        "\nExecutando oclusão..."
    )

    D = len(
        original_detections
    )

    weights = np.zeros(
        (D, N),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Aplicar máscaras em batches
    # --------------------------------------------------------

    masked_images = (
        inp *
        torch.from_numpy(
            masks
        ).to(
            DEVICE,
            dtype=inp.dtype
        )
    )

    # --------------------------------------------------------
    # Loop das máscaras
    # --------------------------------------------------------

    for start in range(
        0,
        N,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            N
        )

        batch = masked_images[
            start:end
        ]

        batch_predictions = (
            xai_model.run_on_batch(
                batch
            )
        )

        for local_index, proposals in enumerate(
            batch_predictions
        ):

            mask_index = (
                start +
                local_index
            )

            if len(proposals) == 0:
                continue

            for detection_index, target in enumerate(
                original_detections
            ):

                best_similarity = 0.0

                for proposal in proposals:

                    # ------------------------------------------------
                    # proposal:
                    #
                    # [x1, y1, x2, y2, confidence, class]
                    #
                    # target:
                    #
                    # [x1, y1, x2, y2, confidence, class]
                    #
                    # ------------------------------------------------

                    if int(
                        target[5]
                    ) != int(
                        proposal[5]
                    ):
                        continue

                    # ------------------------------------------------
                    # IoU
                    # ------------------------------------------------

                    x1 = max(
                        target[0],
                        proposal[0]
                    )

                    y1 = max(
                        target[1],
                        proposal[1]
                    )

                    x2 = min(
                        target[2],
                        proposal[2]
                    )

                    y2 = min(
                        target[3],
                        proposal[3]
                    )

                    intersection = (
                        max(
                            0,
                            x2 - x1
                        )
                        *
                        max(
                            0,
                            y2 - y1
                        )
                    )

                    target_area = (
                        max(
                            0,
                            target[2] -
                            target[0]
                        )
                        *
                        max(
                            0,
                            target[3] -
                            target[1]
                        )
                    )

                    proposal_area = (
                        max(
                            0,
                            proposal[2] -
                            proposal[0]
                        )
                        *
                        max(
                            0,
                            proposal[3] -
                            proposal[1]
                        )
                    )

                    union = (
                        target_area +
                        proposal_area -
                        intersection
                    )

                    if union <= 0:
                        continue

                    iou = (
                        intersection /
                        union
                    )

                    # ------------------------------------------------
                    # Similaridade
                    # ------------------------------------------------

                    similarity = (
                        iou *
                        proposal[4]
                    )

                    if similarity > best_similarity:

                        best_similarity = (
                            similarity
                        )

                weights[
                    detection_index,
                    mask_index
                ] = best_similarity

    # ========================================================
    # 11. GERAR MAPAS DE SALIÊNCIA
    # ========================================================

    print(
        "\nGerando mapas de oclusão..."
    )

    masks_flat = masks.reshape(
        N,
        -1
    )

    saliency_maps = []

    for detection_index in range(
        D
    ):

        saliency = (
            weights[
                detection_index
            ]
            @
            masks_flat
        )

        saliency = saliency.reshape(
            IMGSZ,
            IMGSZ
        )

        # Normalização
        min_value = saliency.min()
        max_value = saliency.max()

        if max_value > min_value:

            saliency = (
                saliency -
                min_value
            ) / (
                max_value -
                min_value
            )

        else:

            saliency = np.zeros_like(
                saliency
            )

        saliency_maps.append(
            saliency
        )

    saliency_maps = np.stack(
        saliency_maps,
        axis=0
    )

    print(
        f"Saliency maps: "
        f"{saliency_maps.shape}"
    )

    # ========================================================
    # 12. EBPG POR CLASSE
    # ========================================================

    print(
        "\nCalculando EBPG..."
    )

    (
        ebpg_results,
        class_maps,
        class_masks
    ) = calculate_ebpg_by_class(
        saliency_maps=saliency_maps,
        detections=original_detections,
        image_shape=(IMGSZ, IMGSZ)
    )

    # ========================================================
    # 13. IMAGEM ORIGINAL
    # ========================================================

    original_image = cv2.imread(
        IMAGE_PATH
    )

    if original_image is None:

        raise FileNotFoundError(
            IMAGE_PATH
        )

    original_image = cv2.resize(
        original_image,
        (IMGSZ, IMGSZ)
    )

    # ========================================================
    # 14. RESULTADOS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EBPG POR CLASSE"
    )

    print(
        "=" * 70
    )

    for result in ebpg_results:

        class_id = result[
            "class_id"
        ]

        class_name = model.names[
            class_id
        ]

        ebpg = result[
            "ebpg"
        ]

        num_predictions = result[
            "num_predictions"
        ]

        print(
            f"\nClasse: {class_name}"
        )

        print(
            f"Class ID: {class_id}"
        )

        print(
            f"Predições: "
            f"{num_predictions}"
        )

        print(
            f"EBPG: "
            f"{ebpg:.6f}"
        )

        print(
            f"Energy inside: "
            f"{result['energy_inside']:.6f}"
        )

        print(
            f"Energy outside: "
            f"{result['energy_outside']:.6f}"
        )

        print(
            f"Total energy: "
            f"{result['total_energy']:.6f}"
        )

        # ====================================================
        # VISUALIZAÇÃO
        # ====================================================

        output_path = os.path.join(
            RESULTS_DIR,
            f"ebpg_class_{class_id}.png"
        )

        visualize_class_ebpg(
            image=original_image,
            class_map=class_maps[
                class_id
            ],
            class_mask=class_masks[
                class_id
            ],
            class_boxes=result[
                "boxes"
            ],
            class_name=class_name,
            class_id=class_id,
            ebpg=ebpg,
            save_path=output_path
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINALIZADO"
    )

    print(
        "=" * 70
    )

    print(
        f"Resultados salvos em: "
        f"{RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()
