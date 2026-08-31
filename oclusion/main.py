
import os
import numpy as np

from Model import Model

from preprocess import *
# ============================================================
# CONFIGURAÇÃO
# ============================================================

IMAGE_PATH = "/content/TubastraeaZoomOut.jpg"

MODEL_PATH = "best.pt"

INPUT_SIZE = (640, 640)

# Somente predições com confiança >= 0.70
# entram na explicação.
CONF_THRESHOLD = 0.70

# ============================================================
# PARÂMETROS DO MÉTODO DE OCLUSÃO
# ============================================================

N = 100

S = 16

P1 = 0.5

BATCH_SIZE = 10


# ============================================================
# RESULTADOS
# ============================================================

RESULTS_DIR = "results_ebpg"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("OCCLUSION — EBPG POR CLASSE")
    print("=" * 70)

    print(f"\nImagem: {IMAGE_PATH}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"N máscaras: {N}")
    print(f"S: {S}")
    print(f"P1: {P1}")
    print(f"Batch size: {BATCH_SIZE}")

    # ========================================================
    # VERIFICAR ARQUIVOS
    # ========================================================

    if not os.path.exists(
        IMAGE_PATH
    ):
        raise FileNotFoundError(
            f"Imagem não encontrada:\n"
            f"{IMAGE_PATH}"
        )

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Modelo não encontrado:\n"
            f"{MODEL_PATH}"
        )

    # ========================================================
    # 1. MODELO
    # ========================================================
    #
    # Esta é exatamente a forma utilizada pelo
    # Model.py do repositório.
    #
    # Model(
    #     model_path=...,
    #     input_size=...,
    #     conf=...
    # )
    #
    # ========================================================

    print("\n" + "=" * 70)
    print("CARREGANDO MODELO")
    print("=" * 70)

    model = Model(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf=CONF_THRESHOLD
    )

    # ========================================================
    # 2. IMAGEM
    # ========================================================

    print("\n" + "=" * 70)
    print("CARREGANDO IMAGEM")
    print("=" * 70)

    inp = load_img(
        IMAGE_PATH,
        INPUT_SIZE
    )

    print(
        f"Input shape: {inp.shape}"
    )

    # ========================================================
    # 3. PREDIÇÃO ORIGINAL
    # ========================================================
    #
    # O Model.py retorna:
    #
    # [x1, y1, x2, y2, confidence, class_id]
    #
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDIÇÕES ORIGINAIS")
    print("=" * 70)

    original_predictions = (
        model.run_on_batch(
            inp
        )
    )

    detections = (
        original_predictions[0]
    )

    if len(detections) == 0:

        raise RuntimeError(
            "Nenhuma detecção foi encontrada."
        )

    # ========================================================
    # 4. FILTRO DE CONFIANÇA
    # ========================================================
    #
    # O Model já utiliza CONF_THRESHOLD.
    # Fazemos também o filtro explicitamente para garantir
    # que somente essas detecções serão utilizadas no EBPG.
    #
    # ========================================================

    detections = detections[
        detections[:, 4]
        >= CONF_THRESHOLD
    ]

    if len(detections) == 0:

        raise RuntimeError(
            f"Nenhuma detecção possui "
            f"confidence >= {CONF_THRESHOLD}."
        )

    print(
        f"\nDetecções utilizadas: "
        f"{len(detections)}"
    )

    # ========================================================
    # 5. MOSTRAR DETECÇÕES
    # ========================================================

    print("\nPredições:")

    for i, detection in enumerate(
        detections
    ):

        x1, y1, x2, y2, conf, cls = (
            detection
        )

        class_id = int(
            cls
        )

        print(
            f"Detection {i}: "
            f"class={class_id} | "
            f"confidence={conf:.4f} | "
            f"box=["
            f"{x1:.2f}, "
            f"{y1:.2f}, "
            f"{x2:.2f}, "
            f"{y2:.2f}"
            f"]"
        )

    # ========================================================
    # 6. CLASSES PRESENTES
    # ========================================================

    class_ids = sorted(
        set(
            detections[:, 5]
            .astype(int)
        )
    )

    print(
        "\nClasses encontradas:"
    )

    for class_id in class_ids:

        class_count = np.sum(
            detections[:, 5].astype(int)
            == class_id
        )

        print(
            f"Classe {class_id}: "
            f"{class_count} predições"
        )

    # ========================================================
    # 7. GERAR MÁSCARAS
    # ========================================================
    #
    # Utiliza diretamente generate_masks()
    # do preprocess.py do repositório.
    #
    # ========================================================

    print("\n" + "=" * 70)
    print("GERANDO MÁSCARAS")
    print("=" * 70)

    masks = generate_masks(
        N=N,
        s=S,
        p1=P1,
        input_size=INPUT_SIZE
    )

    print(
        f"Masks shape: {masks.shape}"
    )

    # Esperado:

    # (N, 1, 640, 640)

    # ========================================================
    # 8. EXPLICAÇÃO POR OCLUSÃO
    # ========================================================
    #
    # Utiliza diretamente explain() do preprocess.py.
    #
    # O método:
    #
    # 1. executa o modelo na imagem original;
    # 2. aplica as N máscaras;
    # 3. executa o modelo nas imagens mascaradas;
    # 4. calcula IoU × confidence;
    # 5. gera um mapa para cada detecção.
    #
    # ========================================================

    print("\n" + "=" * 70)
    print("EXECUTANDO OCLUSÃO")
    print("=" * 70)

    saliency_maps, explain_detections = explain(
        model=model,
        inp=inp,
        masks=masks,
        batch_size=BATCH_SIZE,
        confidence_threshold=CONF_THRESHOLD
    )

    print(
        f"\nSaliency maps shape: "
        f"{saliency_maps.shape}"
    )

    print(
        f"Detections retornadas pelo explain: "
        f"{explain_detections.shape}"
    )

    # ========================================================
    # 9. GARANTIR O MESMO FILTRO
    # ========================================================
    #
    # A implementação atual de explain() possui uma
    # particularidade: ela cria D usando o número de
    # original_detections, embora crie filtered_detections.
    #
    # Como o Model foi criado com conf=0.70, normalmente
    # original_detections já está filtrado.
    #
    # Ainda assim, verificamos explicitamente.
    #
    # ========================================================

    valid_indices = np.where(
        explain_detections[:, 4]
        >= CONF_THRESHOLD
    )[0]

    if len(valid_indices) == 0:

        raise RuntimeError(
            "Nenhuma detecção válida após o explain()."
        )

    # Se necessário, selecionar somente os mapas
    # correspondentes às detecções >= threshold.

    if len(valid_indices) != len(
        explain_detections
    ):

        saliency_maps = (
            saliency_maps[
                valid_indices
            ]
        )

        explain_detections = (
            explain_detections[
                valid_indices
            ]
        )

    # ========================================================
    # 10. EBPG POR CLASSE
    # ========================================================
    #
    # Aqui ocorre a principal diferença em relação
    # ao EBPG original do repositório.
    #
    # O repositório calcula EBPG por bounding box.
    #
    # Nós fazemos:
    #
    # classe 0:
    #
    #   mapa bbox 1
    #   + mapa bbox 2
    #   + mapa bbox 3
    #
    # e:
    #
    #   máscara =
    #       bbox1 UNION bbox2 UNION bbox3
    #
    # Então:
    #
    #   EBPG_classe =
    #       energia dentro da região
    #       /
    #       energia total
    #
    # ========================================================

    print("\n" + "=" * 70)
    print("CALCULANDO EBPG POR CLASSE")
    print("=" * 70)

    (
        ebpg_results,
        class_maps,
        class_masks
    ) = calculate_ebpg_by_class(
        saliency_maps=saliency_maps,
        detections=explain_detections,
        image_shape=INPUT_SIZE
    )

    # ========================================================
    # 11. RESULTADOS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTADO — EBPG POR CLASSE")
    print("=" * 70)

    for result in ebpg_results:

        class_id = result[
            "class_id"
        ]

        num_predictions = result[
            "num_predictions"
        ]

        ebpg = result[
            "ebpg"
        ]

        print(
            f"\nClasse {class_id}"
        )

        print(
            f"  Predições: "
            f"{num_predictions}"
        )

        print(
            f"  EBPG: "
            f"{ebpg:.6f}"
        )

        print(
            f"  Energia dentro: "
            f"{result['energy_inside']:.6f}"
        )

        print(
            f"  Energia fora: "
            f"{result['energy_outside']:.6f}"
        )

        print(
            f"  Energia total: "
            f"{result['total_energy']:.6f}"
        )

    # ========================================================
    # 12. VISUALIZAÇÃO
    # ========================================================

    print("\n" + "=" * 70)
    print("VISUALIZAÇÃO")
    print("=" * 70)

    # --------------------------------------------------------
    # A imagem carregada pelo load_img() já está no formato
    # usado pelo repositório.
    # --------------------------------------------------------

    for result in ebpg_results:

        class_id = result[
            "class_id"
        ]

        ebpg = result[
            "ebpg"
        ]

        # ----------------------------------------------------
        # Nome da classe
        # ----------------------------------------------------

        try:

            class_name = model.model.names[
                class_id
            ]

        except Exception:

            class_name = f"class_{class_id}"

        # ----------------------------------------------------
        # Caminho
        # ----------------------------------------------------

        save_path = os.path.join(
            RESULTS_DIR,
            f"occlusion_ebpg_class_"
            f"{class_id}.png"
        )

        # ----------------------------------------------------
        # Visualização
        # ----------------------------------------------------

        visualize_class_ebpg(
            image=inp,
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
            save_path=save_path
        )

    # ========================================================
    # 13. RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    for result in ebpg_results:

        class_id = result[
            "class_id"
        ]

        print(
            f"Classe {class_id}: "
            f"EBPG = "
            f"{result['ebpg']:.6f}"
        )

    print(
        f"\nResultados salvos em:"
        f"\n{RESULTS_DIR}/"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()

