"""Fixed, structured evaluation set: representative grower questions + the key facts a good answer contains.

`expected_facts` feed the deterministic KeyFacts metric; `expected_output` is the reference answer
DeepEval's ContextualPrecision metric uses to judge whether relevant context was ranked first.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_facts: tuple[str, ...]
    expected_output: str


CASES: list[EvalCase] = [
    EvalCase(
        id="pm_izmir_white_spots",
        query="White spots showed up on my tomato leaves in İzmir",
        expected_facts=("Powdery Mildew", "FungiStop-X"),
        expected_output=(
            "The white spots are most likely Powdery Mildew, a fungal disease that develops in humid conditions. "
            "Treat it with the fungicide FungiStop-X at the first signs, remove affected leaves and improve air "
            "circulation. Growers in İzmir report the same white spots and suspect humidity."
        ),
    ),
    EvalCase(
        id="pm_manisa_spray",
        query="My tomato plants in Manisa have a white powdery coating on the leaves after humid nights. What should I spray?",
        expected_facts=("Powdery Mildew", "FungiStop-X"),
        expected_output=(
            "A white powdery coating after humid nights points to Powdery Mildew. The recommended product is the "
            "fungicide FungiStop-X; also lower humidity and ventilate. Several Manisa growers report the same "
            "problem this month and blame excess humidity."
        ),
    ),
    EvalCase(
        id="early_blight_manisa",
        query="Dark brown spots with rings are appearing on the lower leaves of my tomatoes in Manisa after the rain. What is it and how do I treat it?",
        expected_facts=("Early Blight", "BlightGuard-C"),
        expected_output=(
            "Dark brown spots with concentric rings on lower leaves after rain indicate Early Blight, a fungal "
            "disease favoured by warm, wet weather. Treat it with the copper-based fungicide BlightGuard-C, remove "
            "infected lower leaves and keep the foliage dry. A Manisa grower reported the same rings after rain."
        ),
    ),
    EvalCase(
        id="tuta_izmir_greenhouse",
        query="There are winding tunnels inside my tomato leaves and small moths flying in my İzmir greenhouse.",
        expected_facts=("Tomato Leaf Miner", "TutaTrap-P"),
        expected_output=(
            "Winding tunnels in the leaves plus small moths are signs of the Tomato Leaf Miner (Tuta absoluta). "
            "Use TutaTrap-P pheromone traps to monitor and mass-trap the moths, add insect screens and remove "
            "infested leaves. Another İzmir greenhouse grower reported the same moths and tunnels."
        ),
    ),
    EvalCase(
        id="fusarium_no_region",
        query="The lower leaves of my tomato plants are turning yellow and wilting on one side even though I water regularly. What could it be?",
        expected_facts=("Fusarium Wilt", "TrichoShield-Bio"),
        expected_output=(
            "Yellowing and wilting of lower leaves on one side of the plant suggests Fusarium Wilt, a soil-borne "
            "fungal disease. Infected plants cannot be cured; use the biological soil treatment TrichoShield-Bio "
            "(Trichoderma), resistant varieties and soil solarization."
        ),
    ),
    EvalCase(
        id="manisa_trends",
        query="What problems are tomato growers in Manisa reporting lately?",
        expected_facts=("Powdery Mildew", "Early Blight"),
        expected_output=(
            "Tomato growers in Manisa mostly report Powdery Mildew (white powder on leaves, linked to humid nights), "
            "treatable with FungiStop-X, and Early Blight (dark rings on lower leaves after rain), treatable with "
            "BlightGuard-C."
        ),
    ),
]


def select_cases(limit: int | None = None, case_ids: list[str] | None = None) -> list[EvalCase]:
    cases = [c for c in CASES if not case_ids or c.id in case_ids]
    if case_ids and len(cases) != len(set(case_ids)):
        unknown = set(case_ids) - {c.id for c in cases}
        raise SystemExit(f"Unknown case id(s): {', '.join(sorted(unknown))}. Known: {', '.join(c.id for c in CASES)}")
    return cases[:limit] if limit else cases
