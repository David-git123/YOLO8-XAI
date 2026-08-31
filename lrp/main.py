# main.py

import os
import glob
import json

import numpy as np
import pandas as pd

from utils import (
    load_yolov8,
    explain_image,
    result_to_csv_dict,
    calculate_experiment_statistics
)


# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================

MODEL_PATH = "best.pt"

IMAGE_DIR = "images"

OUTPUT_DIR = "results_lrp_ebpg"

IMG_SIZE = 640

CONF_THRESHOLD = 0.70

IOU_THRESHOLD = 0.70

# Número de pontos das curvas.
#
# steps=20:
#   0%, 5%, 10%, ..., 100%
#
# Cada teste realiza 21 inferências por predição.
PERTURBATION_STEPS = 20

# Valor utilizado como baseline no Deletion/Insertion.
#
# 0.0 = preto
BASELINE_VALUE = 0.0

# IoU mínimo para considerar que uma detecção perturbada
# corresponde à detecção original.
MATCHING_IOU = 0.10

# Parâmetro epsilon da regra LRP.
LRP_EPSILON = 1e-6

# Ativar/desativar os testes.
RUN_DELETION = True
RUN_INSERTION = True


# ============================================================
# EXTENSÕES ACEITAS
# ============================================================

IMAGE_EXTENSIONS = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.BMP"
]


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

def get_image_paths(image_dir):
    """
    Localiza todas as imagens no diretório.
    """

    image_paths = []

    for extension in IMAGE_EXTENSIONS:

        image_paths.extend(
            glob.glob(
                os.path.join(
                    image_dir,
                    extension
                )
            )
        )

    return sorted(
        list(set(image_paths))
    )


# ============================================================
# SALVAR JSON
# ============================================================

def save_json(
    data,
    path
):
    """
    Salva um objeto Python em JSON.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# IMPRIMIR ESTATÍSTICAS
# ============================================================

def print_metric_statistics(
    name,
    statistics
):
    """
    Imprime as estatísticas de uma métrica.
    """

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    if not statistics:

        print("Nenhum valor disponível.")

        return

    print(
        f"N                  : "
        f"{statistics['n']}"
    )

    print(
        f"Média              : "
        f"{statistics['mean']:.6f}"
    )

    print(
        f"Mediana            : "
        f"{statistics['median']:.6f}"
    )

    print(
        f"Desvio padrão      : "
        f"{statistics['std']:.6f}"
    )

    print(
        f"Variância           : "
        f"{statistics['variance']:.6f}"
    )

    print(
        f"Mínimo              : "
        f"{statistics['min']:.6f}"
    )

    print(
        f"Q1                  : "
        f"{statistics['q1']:.6f}"
    )

    print(
        f"Q3                  : "
        f"{statistics['q3']:.6f}"
    )

    print(
        f"IQR                 : "
        f"{statistics['iqr']:.6f}"
    )

    print(
        f"Máximo              : "
        f"{statistics['max']:.6f}"
    )

    print(
        f"IC 95% inferior     : "
        f"{statistics['ci95_low']:.6f}"
    )

    print(
        f"IC 95% superior     : "
        f"{statistics['ci95_high']:.6f}"
    )

    if statistics["shapiro_p"] is not None:

        print(
            f"Shapiro-Wilk p      : "
            f"{statistics['shapiro_p']:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DIRETÓRIOS
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # CABEÇALHO
    # ========================================================

    print("=" * 70)
    print("YOLOv8 + LRP + EBPG + DELETION + INSERTION")
    print("=" * 70)

    print(
        f"\nModelo: {MODEL_PATH}"
    )

    print(
        f"Imagens: {IMAGE_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Image size: {IMG_SIZE}"
    )

    print(
        f"Confidence threshold: {CONF_THRESHOLD}"
    )

    print(
        f"Perturbation steps: {PERTURBATION_STEPS}"
    )

    print(
        f"Deletion: {RUN_DELETION}"
    )

    print(
        f"Insertion: {RUN_INSERTION}"
    )

    # ========================================================
    # VERIFICAÇÕES
    # ========================================================

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Modelo não encontrado: "
            f"{MODEL_PATH}"
        )

    if not os.path.isdir(
        IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: "
            f"{IMAGE_DIR}"
        )

    # ========================================================
    # CARREGAR MODELO
    # ========================================================

    print("\n" + "=" * 70)
    print("CARREGANDO MODELO")
    print("=" * 70)

    model, device = load_yolov8(
        MODEL_PATH
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # VERIFICAR NÚMERO DE CLASSES
    # ========================================================

    try:

        names = model.names

        number_classes = len(
            names
        )

        print(
            f"Classes detectadas: "
            f"{number_classes}"
        )

        print(
            f"Names: {names}"
        )

        if number_classes != 1:

            raise ValueError(
                "Este experimento foi configurado "
                "para um modelo YOLOv8 com exatamente "
                "uma classe."
            )

    except Exception as error:

        print(
            "\nAviso: não foi possível verificar "
            "automaticamente o número de classes."
        )

        print(
            f"Detalhes: {error}"
        )

    # ========================================================
    # LOCALIZAR IMAGENS
    # ========================================================

    image_paths = get_image_paths(
        IMAGE_DIR
    )

    print(
        f"\nImagens encontradas: "
        f"{len(image_paths)}"
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "Nenhuma imagem encontrada."
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    all_results = []

    successful_images = 0

    failed_images = 0

    total_detections = 0

    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"IMAGEM "
            f"{image_number}/{len(image_paths)}"
        )

        print(
            os.path.basename(
                image_path
            )
        )

        print("=" * 70)

        try:

            results = explain_image(
                model=model,

                image_path=image_path,

                img_size=IMG_SIZE,

                conf=CONF_THRESHOLD,

                iou=IOU_THRESHOLD,

                device=device,

                save_dir=OUTPUT_DIR,

                run_deletion=RUN_DELETION,

                run_insertion=RUN_INSERTION,

                perturbation_steps=PERTURBATION_STEPS,

                baseline_value=BASELINE_VALUE,

                matching_iou=MATCHING_IOU,

                lrp_epsilon=LRP_EPSILON
            )

            successful_images += 1

        except Exception as error:
            import traceback
            failed_images += 1


            print("\nERRO AO PROCESSAR:", flush=True)
            print(image_path, flush=True)

            traceback.print_exc()

            continue

        # ----------------------------------------------------
        # Resultados da imagem
        # ----------------------------------------------------

        total_detections += len(
            results
        )

        all_results.extend(
            results
        )

        print(
            f"\nDetecções explicadas nesta imagem: "
            f"{len(results)}"
        )

    # ========================================================
    # VERIFICAR RESULTADOS
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 70)

    print(
        f"Imagens encontradas: "
        f"{len(image_paths)}"
    )

    print(
        f"Imagens processadas: "
        f"{successful_images}"
    )

    print(
        f"Imagens com erro: "
        f"{failed_images}"
    )

    print(
        f"Total de predições explicadas: "
        f"{total_detections}"
    )

    if len(all_results) == 0:

        print(
            "\nNenhuma predição foi encontrada."
        )

        return

    # ========================================================
    # CONVERTER RESULTADOS PARA DATAFRAME
    # ========================================================

    rows = []

    for result in all_results:

        rows.append(
            result_to_csv_dict(
                result
            )
        )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # SALVAR CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nCSV salvo em:"
        f"\n{csv_path}"
    )

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    statistics = calculate_experiment_statistics(
        all_results
    )

    # ========================================================
    # IMPRIMIR ESTATÍSTICAS
    # ========================================================

    print("\n" + "=" * 70)
    print("ESTATÍSTICAS")
    print("=" * 70)

    # --------------------------------------------------------
    # EBPG
    # --------------------------------------------------------

    print_metric_statistics(
        "EBPG",
        statistics.get(
            "EBPG",
            {}
        )
    )

    # --------------------------------------------------------
    # DELETION
    # --------------------------------------------------------

    print_metric_statistics(
        "Deletion — Normalized AUC",
        statistics.get(
            "Deletion_normalized_AUC",
            {}
        )
    )

    # --------------------------------------------------------
    # INSERTION
    # --------------------------------------------------------

    print_metric_statistics(
        "Insertion — Normalized AUC",
        statistics.get(
            "Insertion_normalized_AUC",
            {}
        )
    )

    # ========================================================
    # SALVAR ESTATÍSTICAS
    # ========================================================

    statistics_path = os.path.join(
        OUTPUT_DIR,
        "statistics.json"
    )

    save_json(
        statistics,
        statistics_path
    )

    print(
        f"\nEstatísticas salvas em:"
        f"\n{statistics_path}"
    )

    # ========================================================
    # RESUMO DO EXPERIMENTO
    # ========================================================

    summary = {
        "model": MODEL_PATH,

        "image_directory": IMAGE_DIR,

        "image_size": IMG_SIZE,

        "confidence_threshold":
            CONF_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "perturbation_steps":
            PERTURBATION_STEPS,

        "baseline_value":
            BASELINE_VALUE,

        "matching_iou":
            MATCHING_IOU,

        "lrp_epsilon":
            LRP_EPSILON,

        "run_deletion":
            RUN_DELETION,

        "run_insertion":
            RUN_INSERTION,

        "total_images":
            len(image_paths),

        "successful_images":
            successful_images,

        "failed_images":
            failed_images,

        "total_detections":
            total_detections
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment_summary.json"
    )

    save_json(
        summary,
        summary_path
    )

    print(
        f"Resumo salvo em:"
        f"\n{summary_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    if statistics.get("EBPG"):

        print(
            f"EBPG médio: "
            f"{statistics['EBPG']['mean']:.4f}"
        )

        print(
            f"EBPG mediano: "
            f"{statistics['EBPG']['median']:.4f}"
        )

    if statistics.get(
        "Deletion_normalized_AUC"
    ):

        print(
            f"Deletion AUC médio: "
            f"{statistics['Deletion_normalized_AUC']['mean']:.4f}"
        )

    if statistics.get(
        "Insertion_normalized_AUC"
    ):

        print(
            f"Insertion AUC médio: "
            f"{statistics['Insertion_normalized_AUC']['mean']:.4f}"
        )

    print(
        "\nResultados:"
        f"\n{OUTPUT_DIR}"
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()