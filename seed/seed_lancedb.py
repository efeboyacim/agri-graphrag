"""Embed a handful of short agronomy notes (all-MiniLM-L6-v2, local) and write them to LanceDB.

Run from the repo root:  python seed/seed_lancedb.py
Product names are deliberately left out of the notes: treatments come from the graph.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import vector_client  # noqa: E402
from app.config import LANCEDB_PATH, LANCEDB_TABLE, configure_console  # noqa: E402

NOTES = [
    {
        "id": "note-pm-humidity",
        "title": "Powdery mildew and humidity",
        "topic": "powdery mildew",
        "text": "Powdery mildew spreads quickly in high humidity; early detection matters. On tomato the "
        "first sign is small white, powdery spots on older leaves, which can merge until the leaf yellows and dries.",
    },
    {
        "id": "note-pm-management",
        "title": "Managing powdery mildew",
        "topic": "powdery mildew",
        "text": "To manage powdery mildew on tomatoes, remove the most affected leaves, improve air circulation "
        "between plants, and apply a registered fungicide at the first signs, following the label. "
        "Alternate fungicide modes of action to avoid resistance.",
    },
    {
        "id": "note-early-blight",
        "title": "Early blight",
        "topic": "early blight",
        "text": "Early blight (Alternaria solani) appears first on older, lower leaves as dark brown spots with "
        "concentric rings, often with a yellow halo. Warm weather and leaf wetness after rain or overhead "
        "irrigation favour it. Remove infected lower leaves, mulch the soil and keep the foliage dry.",
    },
    {
        "id": "note-fusarium",
        "title": "Fusarium wilt",
        "topic": "fusarium wilt",
        "text": "Fusarium wilt is soil-borne: lower leaves yellow and wilt, often on one side of the plant, and the "
        "stem shows brown streaks inside. Infected plants cannot be cured; use resistant varieties, summer soil "
        "solarization and biological soil treatments based on Trichoderma.",
    },
    {
        "id": "note-tuta",
        "title": "Tomato leaf miner (Tuta absoluta)",
        "topic": "tomato leaf miner",
        "text": "Tomato leaf miner (Tuta absoluta) larvae dig winding tunnels and blotch mines inside leaves and can "
        "bore into fruit. Pheromone traps monitor and mass-trap the small grey-brown moths; insect screens on "
        "greenhouse vents and removing infested leaves reduce the population.",
    },
    {
        "id": "note-greenhouse-humidity",
        "title": "Greenhouse humidity control",
        "topic": "prevention",
        "text": "In tomato greenhouses, keeping relative humidity below about 85% reduces most fungal diseases. "
        "Ventilate in the morning, prefer drip irrigation over overhead sprinklers, and avoid overcrowding plants.",
    },
    {
        "id": "note-aegean-season",
        "title": "Aegean late-summer disease pressure",
        "topic": "regional",
        "text": "In the Aegean region, including İzmir and Manisa, late summer brings warm days and humid nights. "
        "September is a high-risk period for tomato fungal diseases such as powdery mildew and early blight, "
        "so scouting should be increased.",
    },
    {
        "id": "note-scouting",
        "title": "Weekly scouting routine",
        "topic": "prevention",
        "text": "Scout tomato fields at least once a week: check both sides of the leaves on plants across the field, "
        "record what you find, and act early, because treatments work best at the first signs of disease or pests.",
    },
]


def main() -> None:
    configure_console()
    print(f"Embedding {len(NOTES)} agronomy notes into LanceDB at {LANCEDB_PATH} (table {LANCEDB_TABLE}) ...")
    rows = [{**note, "source": "agronomy-notes"} for note in NOTES]
    count = vector_client.write_chunks(rows)
    print(f"Loaded {count} chunks.")


if __name__ == "__main__":
    main()
