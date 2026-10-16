from app.services.abcde.evolution import compute_evolution


def test_evolution_no_previous_image():
    current = {
        "diameter_mm": 9.0,
        "asymmetry": 0.28,
        "border_irregularity": 0.56,
        "color_index": 0.57,
    }

    result = compute_evolution(
        previous_features=None,
        current_features=current,
    )

    assert result["available"] is False
    assert result["change_detected"] is False


def test_evolution_no_meaningful_change():
    previous = {
        "diameter_mm": 9.0,
        "asymmetry": 0.28,
        "border_irregularity": 0.56,
        "color_index": 0.57,
    }

    current = {
        "diameter_mm": 9.1,
        "asymmetry": 0.29,
        "border_irregularity": 0.57,
        "color_index": 0.58,
    }

    result = compute_evolution(
        previous_features=previous,
        current_features=current,
    )

    assert result["available"] is True
    assert result["change_detected"] is False


def test_evolution_meaningful_change():
    previous = {
        "diameter_mm": 8.0,
        "asymmetry": 0.20,
        "border_irregularity": 0.40,
        "color_index": 0.40,
    }

    current = {
        "diameter_mm": 10.0,
        "asymmetry": 0.35,
        "border_irregularity": 0.60,
        "color_index": 0.58,
    }

    result = compute_evolution(
        previous_features=previous,
        current_features=current,
    )

    assert result["available"] is True
    assert result["change_detected"] is True

    assert result["diameter_change"] == 2.0
    assert result["asymmetry_change"] == 0.15
    assert result["border_change"] == 0.20
    assert result["color_change"] == 0.18