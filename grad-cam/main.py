import os
import glob
import traceback

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

from gradcam_yolo import (
    generate_gradcam,
    save_gradcam_visualization,
)

from metrics import (
    calculate_ebpg,
    calculate_pointing_game,
    deletion_test,
    insertion_test,
    save_curve,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = "best.pt"

IMAGE_DIR = "images"

OUTPUT_DIR = "results_gradcam"

IMAGE_SIZE = 640

# Somente detecções acima deste threshold serão explicadas.
CONF_THRESHOLD = 0.70

# NMS do YOLO.
IOU_THRESHOLD = 0.70

# Número de intervalos:
#
# 20 -> 0%, 5%, ..., 100%
#
# Portanto:
#
# 21 inferências por curva.
PERTURBATION_STEPS = 20

BASELINE_VALUE = 0.0

# IoU usado para identificar a mesma detecção durante
# Deletion/Insertion.
MATCHING_IOU = 0.10

# Camada utilizada pelo Grad-CAM.
#
# No seu modelo você já verificou que -2 é C2f.
TARGET_LAYER_INDEX = -2

# Classe Tubastrea.
TARGET_CLASS = 0

# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
#
# Se você possuir:
#
# labels/
#     TubastraeaZoomOut.txt
#
# coloque:
#
# LABELS_DIR = "labels"
#
# Caso contrário:
#
# LABELS_DIR = None
#
# Nesse caso EBPG será calculado em relação à bounding box
# detectada pelo próprio modelo.
#
# Para avaliação metodológica de localização, o ideal é usar
# ground truth.
LABELS_DIR = None

SAVE_CURVES = True


# ============================================================
# UTILITÁRIOS
# ============================================================

def load_images(directory):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp",
    ]

    files = []

    for extension in extensions:

        files.extend(
            glob.glob(
                os.path.join(
                    directory,
                    extension,
                )
            )
        )

    return sorted(files)


def load_ground_truth(
    image_path,
    labels_dir,
):
    """
    Lê labels no formato YOLO:

        class x_center y_center width height

    valores normalizados em [0,1].
    """

    if labels_dir is None:
        return []

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    label_path = os.path.join(
        labels_dir,
        filename + ".txt",
    )

    if not os.path.exists(
        label_path
    ):
        return []

    image = cv2.imread(
        image_path
    )

    if image is None:
        return []

    h, w = image.shape[:2]

    ground_truth = []

    with open(
        label_path,
        "r",
    ) as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 5:
                continue

            cls = int(
                float(values[0])
            )

            xc = float(
                values[1]
            ) * w

            yc = float(
                values[2]
            ) * h

            bw = float(
                values[3]
            ) * w

            bh = float(
                values[4]
            ) * h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            ground_truth.append(
                {
                    "class_id": cls,
                    "box": np.array(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        dtype=np.float32,
                    ),
                }
            )

    return ground_truth


def find_best_ground_truth(
    predicted_box,
    predicted_class,
    ground_truth,
):
    """
    Encontra a GT com maior IoU para a detecção.
    """

    best_gt = None
    best_iou = 0.0

    from metrics import calculate_iou

    for gt in ground_truth:

        if gt["class_id"] != predicted_class:
            continue

        iou = calculate_iou(
            predicted_box,
            gt["box"],
        )

        if iou > best_iou:

            best_iou = iou
            best_gt = gt

    return best_gt, best_iou


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "YOLOv8 + pytorch-grad-cam + EBPG + DELETION + INSERTION"
    )

    print("=" * 70)

    print(
        f"Modelo: {MODEL_PATH}"
    )

    print(
        f"Imagens: {IMAGE_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Confidence threshold: {CONF_THRESHOLD}"
    )

    print(
        f"NMS IoU threshold: {IOU_THRESHOLD}"
    )

    print(
        f"Matching IoU: {MATCHING_IOU}"
    )

    print(
        f"Perturbation steps: {PERTURBATION_STEPS}"
    )

    print(
        f"Baseline: {BASELINE_VALUE}"
    )

    print(
        f"Grad-CAM layer: {TARGET_LAYER_INDEX}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    print(
        "\nCarregando modelo..."
    )

    yolo = YOLO(
        MODEL_PATH
    )

    yolo.model.to(
        device
    )

    yolo.model.eval()

    print(
        "Modelo carregado."
    )

    print(
        f"Classes: {yolo.names}"
    )

    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    target_layer = (
        yolo.model.model[
            TARGET_LAYER_INDEX
        ]
    )

    print(
        "\nCamada Grad-CAM:"
    )

    print(
        f"Índice: {TARGET_LAYER_INDEX}"
    )

    print(
        target_layer
    )

    # --------------------------------------------------------
    # IMAGENS
    # --------------------------------------------------------

    image_paths = load_images(
        IMAGE_DIR
    )

    print(
        f"\nImagens encontradas: {len(image_paths)}"
    )

    if len(image_paths) == 0:

        print(
            "Nenhuma imagem encontrada."
        )

        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    all_results = []

    processed_images = 0
    failed_images = 0
    total_explanations = 0

    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"IMAGEM {image_number}/{len(image_paths)}"
        )

        print(
            os.path.basename(image_path)
        )

        print(
            "=" * 70
        )

        try:

            image = cv2.imread(
                image_path
            )

            if image is None:
                raise RuntimeError(
                    f"Não foi possível abrir {image_path}"
                )

            # ------------------------------------------------
            # DETECÇÃO NORMAL
            # ------------------------------------------------

            results = yolo.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:

                print(
                    "Nenhuma bounding box encontrada."
                )

                continue

            number_detections = len(
                result.boxes
            )

            print(
                f"Detecções acima de {CONF_THRESHOLD}: "
                f"{number_detections}"
            )

            if number_detections == 0:
                continue

            # ------------------------------------------------
            # GROUND TRUTH
            # ------------------------------------------------

            ground_truth = load_ground_truth(
                image_path,
                LABELS_DIR,
            )

            if LABELS_DIR is not None:

                if ground_truth:

                    print(
                        f"Ground truth encontrada: "
                        f"{len(ground_truth)} boxes"
                    )

                else:

                    print(
                        "Nenhuma ground truth encontrada. "
                        "EBPG usará a box prevista."
                    )

            # ------------------------------------------------
            # DETECÇÕES
            # ------------------------------------------------

            for detection_index in range(
                number_detections
            ):

                print(
                    "\n" + "-" * 70
                )

                print(
                    f"DETECÇÃO {detection_index}"
                )

                box = (
                    result.boxes.xyxy[
                        detection_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                confidence = float(
                    result.boxes.conf[
                        detection_index
                    ]
                    .detach()
                    .cpu()
                )

                class_id = int(
                    result.boxes.cls[
                        detection_index
                    ]
                    .detach()
                    .cpu()
                )

                print(
                    f"Classe: {class_id}"
                )

                print(
                    f"Confidence: {confidence:.4f}"
                )

                print(
                    f"Box: {box.tolist()}"
                )

                # ------------------------------------------------
                # BOX DE REFERÊNCIA PARA EBPG
                # ------------------------------------------------

                reference_box = box.copy()

                reference_type = (
                    "predicted_box"
                )

                gt_iou = np.nan

                if ground_truth:

                    gt, gt_iou_value = (
                        find_best_ground_truth(
                            box,
                            class_id,
                            ground_truth,
                        )
                    )

                    if gt is not None:

                        reference_box = (
                            gt["box"].copy()
                        )

                        reference_type = (
                            "ground_truth"
                        )

                        gt_iou = (
                            gt_iou_value
                        )

                        print(
                            f"GT IoU: {gt_iou:.4f}"
                        )

                # ------------------------------------------------
                # GRAD-CAM
                # ------------------------------------------------

                print(
                    "\nGerando Grad-CAM..."
                )

                cam_result = generate_gradcam(
                    yolo_model=yolo,
                    image=image,
                    box=box,
                    class_id=class_id,
                    imgsz=IMAGE_SIZE,
                    target_layer_index=TARGET_LAYER_INDEX,
                    device=device,
                )

                saliency = (
                    cam_result["saliency"]
                )

                raw_index = (
                    cam_result["raw_index"]
                )

                raw_iou = (
                    cam_result["raw_iou"]
                )

                print(
                    f"RAW index: {raw_index}"
                )

                print(
                    f"RAW box IoU: {raw_iou:.4f}"
                )

                # ------------------------------------------------
                # EBPG
                # ------------------------------------------------

                ebpg = calculate_ebpg(
                    saliency,
                    reference_box,
                )

                pointing_game = (
                    calculate_pointing_game(
                        saliency,
                        reference_box,
                    )
                )

                print(
                    f"EBPG: {ebpg:.6f}"
                )

                print(
                    f"Pointing Game: "
                    f"{pointing_game:.0f}"
                )

                # ------------------------------------------------
                # VISUALIZAÇÃO
                # ------------------------------------------------

                image_name = os.path.splitext(
                    os.path.basename(
                        image_path
                    )
                )[0]

                cam_filename = (
                    f"{image_name}"
                    f"_det{detection_index}"
                    f"_gradcam.jpg"
                )

                cam_path = os.path.join(
                    OUTPUT_DIR,
                    cam_filename,
                )

                save_gradcam_visualization(
                    image=image,
                    saliency=saliency,
                    box=box,
                    output_path=cam_path,
                )

                print(
                    f"Grad-CAM salvo: {cam_path}"
                )

                # ------------------------------------------------
                # DELETION
                # ------------------------------------------------

                print(
                    "\nExecutando Deletion..."
                )

                deletion = deletion_test(
                    model=yolo,
                    image=image,
                    saliency=saliency,
                    target_box=box,
                    target_class=class_id,
                    original_score=confidence,
                    steps=PERTURBATION_STEPS,
                    baseline=BASELINE_VALUE,
                    iou_nms=IOU_THRESHOLD,
                    matching_iou=MATCHING_IOU,
                )

                print(
                    f"Deletion AUC: "
                    f"{deletion['auc']:.6f}"
                )

                print(
                    f"Deletion normalized AUC: "
                    f"{deletion['normalized_auc']:.6f}"
                )

                # ------------------------------------------------
                # INSERTION
                # ------------------------------------------------

                print(
                    "\nExecutando Insertion..."
                )

                insertion = insertion_test(
                    model=yolo,
                    image=image,
                    saliency=saliency,
                    target_box=box,
                    target_class=class_id,
                    original_score=confidence,
                    steps=PERTURBATION_STEPS,
                    baseline=BASELINE_VALUE,
                    iou_nms=IOU_THRESHOLD,
                    matching_iou=MATCHING_IOU,
                )

                print(
                    f"Insertion AUC: "
                    f"{insertion['auc']:.6f}"
                )

                print(
                    f"Insertion normalized AUC: "
                    f"{insertion['normalized_auc']:.6f}"
                )

                # ------------------------------------------------
                # CURVAS
                # ------------------------------------------------

                if SAVE_CURVES:

                    deletion_path = os.path.join(
                        OUTPUT_DIR,
                        f"{image_name}"
                        f"_det{detection_index}"
                        f"_deletion.png",
                    )

                    save_curve(
                        deletion["fractions"],
                        deletion["scores"],
                        deletion_path,
                        f"Deletion - "
                        f"{image_name} "
                        f"Detection {detection_index}",
                        "Detection confidence",
                    )

                    insertion_path = os.path.join(
                        OUTPUT_DIR,
                        f"{image_name}"
                        f"_det{detection_index}"
                        f"_insertion.png",
                    )

                    save_curve(
                        insertion["fractions"],
                        insertion["scores"],
                        insertion_path,
                        f"Insertion - "
                        f"{image_name} "
                        f"Detection {detection_index}",
                        "Detection confidence",
                    )

                # ------------------------------------------------
                # RESULTADO
                # ------------------------------------------------

                all_results.append(
                    {
                        "image": os.path.basename(
                            image_path
                        ),

                        "detection_index":
                            detection_index,

                        "class_id":
                            class_id,

                        "confidence":
                            confidence,

                        "x1":
                            float(box[0]),

                        "y1":
                            float(box[1]),

                        "x2":
                            float(box[2]),

                        "y2":
                            float(box[3]),

                        "raw_index":
                            raw_index,

                        "raw_iou":
                            raw_iou,

                        "reference_type":
                            reference_type,

                        "gt_iou":
                            gt_iou,

                        "ebpg":
                            ebpg,

                        "pointing_game":
                            pointing_game,

                        "deletion_auc":
                            deletion["auc"],

                        "deletion_normalized_auc":
                            deletion[
                                "normalized_auc"
                            ],

                        "insertion_auc":
                            insertion["auc"],

                        "insertion_normalized_auc":
                            insertion[
                                "normalized_auc"
                            ],
                    }
                )

                total_explanations += 1

                print(
                    "\nDetecção processada."
                )

            processed_images += 1

        except Exception as e:

            failed_images += 1

            print(
                "\nERRO AO PROCESSAR:"
            )

            print(
                image_path
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

    # ========================================================
    # RESULTADOS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESSAMENTO FINALIZADO"
    )

    print(
        "=" * 70
    )

    print(
        f"Imagens encontradas: "
        f"{len(image_paths)}"
    )

    print(
        f"Imagens processadas: "
        f"{processed_images}"
    )

    print(
        f"Imagens com erro: "
        f"{failed_images}"
    )

    print(
        f"Total de predições explicadas: "
        f"{total_explanations}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if all_results:

        dataframe = pd.DataFrame(
            all_results
        )

        csv_path = os.path.join(
            OUTPUT_DIR,
            "metrics.csv",
        )

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"\nCSV salvo: {csv_path}"
        )

        # ----------------------------------------------------
        # ESTATÍSTICAS
        # ----------------------------------------------------

        print(
            "\nMÉDIAS"
        )

        print(
            f"EBPG: "
            f"{dataframe['ebpg'].mean():.6f}"
        )

        print(
            f"Pointing Game: "
            f"{dataframe['pointing_game'].mean():.6f}"
        )

        print(
            f"Deletion AUC: "
            f"{dataframe['deletion_auc'].mean():.6f}"
        )

        print(
            f"Insertion AUC: "
            f"{dataframe['insertion_auc'].mean():.6f}"
        )

        print(
            "\n" + dataframe[
                [
                    "image",
                    "detection_index",
                    "confidence",
                    "ebpg",
                    "deletion_auc",
                    "insertion_auc",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNenhuma predição foi explicada."
        )


if __name__ == "__main__":
    main()