def insertion_test(
    model,
    inp,
    saliency,
    target,
    grid_size=16,
    steps=20,
    iou_threshold=0.5,
    baseline_value=0.0
):
    """
    Insertion Test para Grad-CAM.

    Começa com uma imagem baseline e adiciona
    progressivamente as regiões mais importantes
    indicadas pelo mapa Grad-CAM.

    Parâmetros
    ----------
    model:
        Objeto contendo o modelo YOLO.

    inp:
        Tensor da imagem original.
        Formato:
            [1, 3, H, W]

    saliency:
        Mapa Grad-CAM.
        Formato:
            [H, W]

    target:
        Detecção original:
            [x1, y1, x2, y2, confidence]

    grid_size:
        Número de regiões por dimensão.

        Exemplo:
            16 -> 16x16 = 256 regiões.

    steps:
        Número de etapas do experimento.

    iou_threshold:
        IoU mínimo para considerar que a detecção
        encontrada corresponde à detecção original.

    baseline_value:
        Valor inicial das regiões.

        Para imagem normalizada [0,1]:
            0.0 = preto.

    Retorna
    -------
    fractions:
        Fração da imagem original inserida.

    confidences:
        Confiança da detecção correspondente.

    ious:
        IoU da detecção encontrada com a detecção original.

    auc:
        Área sob a curva de Insertion.
    """

    # =========================================================
    # 1. Verificações
    # =========================================================

    if inp.ndim != 4:
        raise ValueError(
            f"'inp' deve possuir 4 dimensões [B,C,H,W]. "
            f"Recebido: {inp.shape}"
        )

    if inp.shape[0] != 1:
        raise ValueError(
            "O Insertion Test atual espera uma única imagem."
        )

    if inp.shape[1] != 3:
        raise ValueError(
            f"A imagem deve possuir 3 canais. "
            f"Recebido: {inp.shape[1]} canais."
        )

    if len(target) < 5:
        raise ValueError(
            "target deve possuir pelo menos "
            "[x1, y1, x2, y2, confidence]."
        )

    # =========================================================
    # 2. Normalizar Grad-CAM
    # =========================================================

    saliency = np.asarray(
        saliency,
        dtype=np.float32
    )

    if saliency.ndim != 2:
        raise ValueError(
            f"'saliency' deve possuir formato [H,W]. "
            f"Recebido: {saliency.shape}"
        )

    saliency_min = saliency.min()
    saliency_max = saliency.max()

    saliency = (
        saliency - saliency_min
    ) / (
        saliency_max -
        saliency_min +
        1e-8
    )

    # =========================================================
    # 3. Dimensões
    # =========================================================

    _, channels, height, width = inp.shape

    if saliency.shape != (
        height,
        width
    ):
        raise ValueError(
            "O tamanho do mapa Grad-CAM não corresponde "
            "ao tamanho da imagem.\n"
            f"Imagem: {height}x{width}\n"
            f"Saliency: {saliency.shape}"
        )

    # =========================================================
    # 4. Criar regiões grid_size × grid_size
    # =========================================================

    regions = []

    cell_height = height / grid_size
    cell_width = width / grid_size

    for grid_y in range(grid_size):

        for grid_x in range(grid_size):

            y1 = int(
                grid_y * cell_height
            )

            y2 = int(
                (grid_y + 1) * cell_height
            )

            x1 = int(
                grid_x * cell_width
            )

            x2 = int(
                (grid_x + 1) * cell_width
            )

            region_saliency = saliency[
                y1:y2,
                x1:x2
            ]

            # Energia da região
            importance = float(
                region_saliency.sum()
            )

            regions.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "importance": importance
            })

    # =========================================================
    # 5. Ordenar regiões
    # =========================================================

    regions.sort(
        key=lambda region: region["importance"],
        reverse=True
    )

    total_regions = len(regions)

    # =========================================================
    # 6. Imagem original
    # =========================================================

    original = (
        inp[0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    # =========================================================
    # 7. Imagem baseline
    # =========================================================

    baseline = np.full_like(
        original,
        baseline_value
    )

    # =========================================================
    # 8. Valores iniciais
    # =========================================================

    fractions = []
    confidences = []
    ious = []

    # =========================================================
    # 9. Insertion progressivo
    # =========================================================

    for step in range(
        steps + 1
    ):

        fraction = step / steps

        number_regions = int(
            fraction * total_regions
        )

        # ---------------------------------------------
        # Criar imagem a partir do baseline
        # ---------------------------------------------

        modified = baseline.copy()

        # ---------------------------------------------
        # Inserir regiões mais importantes
        # ---------------------------------------------

        for region in regions[
            :number_regions
        ]:

            x1 = region["x1"]
            y1 = region["y1"]
            x2 = region["x2"]
            y2 = region["y2"]

            modified[
                :,
                y1:y2,
                x1:x2
            ] = original[
                :,
                y1:y2,
                x1:x2
            ]

        # =================================================
        # 10. Converter para tensor
        # =================================================

        modified_tensor = torch.from_numpy(
            modified
        ).unsqueeze(0).float()

        modified_tensor = modified_tensor.to(
            inp.device
        )

        # =================================================
        # 11. YOLO
        # =================================================

        confidence, best_iou = get_detection_confidence(
            model.model,
            modified_tensor,
            target,
            iou_threshold=iou_threshold,
            imgsz=(height, width),
            conf=0.001
        )

        # =================================================
        # 12. Armazenar
        # =================================================

        fractions.append(
            fraction
        )

        confidences.append(
            confidence
        )

        ious.append(
            best_iou
        )

        # =================================================
        # 13. Progresso
        # =================================================

        print(
            f"Insertion "
            f"{fraction:6.1%} "
            f"| regiões inseridas: "
            f"{number_regions:3d}/{total_regions} "
            f"| confidence: "
            f"{confidence:.4f} "
            f"| IoU: "
            f"{best_iou:.4f}"
        )

    # =========================================================
    # 14. Converter para numpy
    # =========================================================

    fractions = np.asarray(
        fractions,
        dtype=np.float32
    )

    confidences = np.asarray(
        confidences,
        dtype=np.float32
    )

    ious = np.asarray(
        ious,
        dtype=np.float32
    )

    # =========================================================
    # 15. AUC
    # =========================================================

    auc = np.trapezoid(
        confidences,
        fractions
    )

    print()
    print(
        f"Insertion AUC: {auc:.6f}"
    )

    # =========================================================
    # 16. Retorno
    # =========================================================

    return (
        fractions,
        confidences,
        ious,
        float(auc)
    )