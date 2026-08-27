def deletion_test(
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
    Deletion Test para Grad-CAM.

    Remove progressivamente as regiões mais importantes
    indicadas pelo mapa Grad-CAM.

    Parâmetros
    ----------
    model:
        Objeto yolov8_heatmap.

    inp:
        Tensor da imagem original.
        Formato esperado: [1, 3, H, W].

    saliency:
        Mapa Grad-CAM.
        Formato esperado: [H, W].

    target:
        Detecção original:
        [x1, y1, x2, y2, confidence]

    grid_size:
        Número de regiões por dimensão.
        Exemplo:
            16 -> 16x16 = 256 regiões.

    steps:
        Número de etapas do Deletion Test.

    iou_threshold:
        IoU mínimo para considerar que a detecção
        encontrada corresponde à detecção original.

    baseline_value:
        Valor utilizado para apagar as regiões.
        Para tensor normalizado [0,1]:
            0.0 = preto.

    Retorna
    -------
    fractions:
        Fração das regiões removidas.

    confidences:
        Confiança da detecção correspondente
        em cada etapa.

    ious:
        IoU da detecção encontrada em relação à
        bounding box original.

    auc:
        Área sob a curva de Deletion.
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
            "O Deletion Test atual espera uma única imagem."
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
    # 2. Normalizar mapa Grad-CAM
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
    # 4. Criar regiões do mapa Grad-CAM
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

            importance = float(
                region_saliency.mean()
            )

            regions.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "importance": importance
            })

    # =========================================================
    # 5. Ordenar regiões pela importância
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
    # 7. Valores iniciais
    # =========================================================

    original_confidence = float(
        target[4]
    )

    fractions = [
        0.0
    ]

    confidences = [
        original_confidence
    ]

    ious = [
        1.0
    ]

    # =========================================================
    # 8. Deletion progressivo
    # =========================================================

    for step in range(
        1,
        steps + 1
    ):

        fraction = step / steps

        # Número de regiões que serão removidas
        number_regions = int(
            fraction * total_regions
        )

        # ---------------------------------------------
        # Criar cópia da imagem original
        # ---------------------------------------------

        modified = original.copy()

        # ---------------------------------------------
        # Remover as regiões mais importantes
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
            ] = baseline_value

        # =================================================
        # 9. Converter para tensor
        # =================================================

        modified_tensor = torch.from_numpy(
            modified
        ).unsqueeze(0).float()

        # Utilizar o mesmo device da imagem original
        modified_tensor = modified_tensor.to(
            inp.device
        )

        # =================================================
        # 10. Executar YOLO
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
        # 11. Armazenar resultados
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
        # 12. Mostrar progresso
        # =================================================

        print(
            f"Deletion "
            f"{fraction:6.1%} "
            f"| regiões removidas: "
            f"{number_regions:3d}/{total_regions} "
            f"| confidence: "
            f"{confidence:.4f} "
            f"| IoU: "
            f"{best_iou:.4f}"
        )

    # =========================================================
    # 13. Calcular AUC
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

    auc = np.trapezoid(
        confidences,
        fractions
    )

    # =========================================================
    # 14. Resultado
    # =========================================================

    print()
    print(
        f"Deletion AUC: {auc:.6f}"
    )

    return (
        fractions,
        confidences,
        ious,
        float(auc)
    )