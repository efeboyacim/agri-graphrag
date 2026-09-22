"""Load the small agriculture knowledge graph into Neo4j (idempotent: wipes the demo labels first).

Run from the repo root:  python seed/seed_neo4j.py [--force]

Safety: the wipe only runs on an empty database or one this script seeded before (it leaves an
:AgriGraphRAGSeed marker node). Pointed at some other non-empty Neo4j it refuses unless --force.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import graph_client  # noqa: E402
from app.config import NEO4J_URI, configure_console  # noqa: E402

WIPE = """
MATCH (n) WHERE n:Crop OR n:Region OR n:Disease OR n:Symptom OR n:Product OR n:SocialPost
DETACH DELETE n
"""

# The spec's seed set verbatim, followed by an extension (three more tomato problems) so the
# evaluation set can cover different symptom/disease combinations. One statement, so the
# extension can reuse the spec's variables.
SEED = """
CREATE (tomato:Crop {name: "tomato"})
CREATE (izmir:Region {name: "İzmir"})
CREATE (manisa:Region {name: "Manisa"})
CREATE (mildew:Disease {name: "Powdery Mildew", description: "Fungal disease that develops in humid conditions"})
CREATE (symptom:Symptom {description: "white spots"})
CREATE (product:Product {name: "FungiStop-X", type: "fungicide"})

CREATE (tomato)-[:GROWN_IN_REGION]->(izmir)
CREATE (tomato)-[:GROWN_IN_REGION]->(manisa)
CREATE (tomato)-[:AFFECTED_BY]->(mildew)
CREATE (mildew)-[:HAS_SYMPTOM]->(symptom)
CREATE (mildew)-[:TREATED_BY]->(product)

// Mock social media posts (read by the Social Listening Agent)
CREATE (p1:SocialPost {text: "Seeing white spots on my tomatoes in İzmir, could it be humidity?", region: "İzmir", date: "2026-09-10"})
CREATE (p2:SocialPost {text: "Same thing happened in Manisa, people online say it's excess humidity", region: "Manisa", date: "2026-09-12"})
CREATE (p1)-[:RELATED_TO]->(mildew)
CREATE (p2)-[:RELATED_TO]->(mildew)

// --- Extension: three more tomato problems ---
CREATE (blight:Disease {name: "Early Blight", description: "Fungal disease (Alternaria solani) favoured by warm, wet weather; it starts on older, lower leaves"})
CREATE (ringSpots:Symptom {description: "dark brown spots with concentric rings on lower leaves"})
CREATE (blightGuard:Product {name: "BlightGuard-C", type: "copper-based fungicide"})
CREATE (fusarium:Disease {name: "Fusarium Wilt", description: "Soil-borne fungal disease that blocks the plant's water-conducting tissue; it persists in the soil for years"})
CREATE (wilting:Symptom {description: "yellowing and wilting of lower leaves, often on one side of the plant"})
CREATE (trichoShield:Product {name: "TrichoShield-Bio", type: "biological soil treatment (Trichoderma)"})
CREATE (leafMiner:Disease {name: "Tomato Leaf Miner", description: "Insect pest (Tuta absoluta) whose larvae tunnel through leaves and fruit; common in greenhouses"})
CREATE (mines:Symptom {description: "winding tunnels and blotch mines inside leaves, small moths in the greenhouse"})
CREATE (tutaTrap:Product {name: "TutaTrap-P", type: "pheromone trap"})

CREATE (tomato)-[:AFFECTED_BY]->(blight)
CREATE (blight)-[:HAS_SYMPTOM]->(ringSpots)
CREATE (blight)-[:TREATED_BY]->(blightGuard)
CREATE (tomato)-[:AFFECTED_BY]->(fusarium)
CREATE (fusarium)-[:HAS_SYMPTOM]->(wilting)
CREATE (fusarium)-[:TREATED_BY]->(trichoShield)
CREATE (tomato)-[:AFFECTED_BY]->(leafMiner)
CREATE (leafMiner)-[:HAS_SYMPTOM]->(mines)
CREATE (leafMiner)-[:TREATED_BY]->(tutaTrap)

CREATE (p3:SocialPost {text: "Dark rings on the lower tomato leaves after last week's rain here in Manisa, anyone else?", region: "Manisa", date: "2026-09-15"})
CREATE (p4:SocialPost {text: "Found tiny moths and tunnels in the leaves of my İzmir greenhouse tomatoes", region: "İzmir", date: "2026-09-14"})
CREATE (p5:SocialPost {text: "Third farm in Manisa this week with white powder on tomato leaves, the humid nights are not helping", region: "Manisa", date: "2026-09-18"})
CREATE (p3)-[:RELATED_TO]->(blight)
CREATE (p4)-[:RELATED_TO]->(leafMiner)
CREATE (p5)-[:RELATED_TO]->(mildew)
"""

MARKER = "MERGE (m:AgriGraphRAGSeed {name: 'agrigraphrag-demo'}) SET m.seeded_at = datetime()"

COUNTS = """
MATCH (n) WHERE n:Crop OR n:Region OR n:Disease OR n:Symptom OR n:Product OR n:SocialPost
RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label
"""


def main() -> None:
    configure_console()
    print(f"Seeding Neo4j at {NEO4J_URI} ...")
    existing = graph_client.run("MATCH (n) RETURN count(n) AS n")[0]["n"]
    ours = graph_client.run("MATCH (m:AgriGraphRAGSeed) RETURN count(m) AS n")[0]["n"]
    if existing and not ours and "--force" not in sys.argv:
        graph_client.close()
        raise SystemExit(
            f"Refusing to wipe: {NEO4J_URI} already holds {existing} nodes that this script did not create "
            "(no :AgriGraphRAGSeed marker). Check NEO4J_URI points at the AgriGraphRAG container, "
            "or re-run with --force."
        )
    graph_client.run(WIPE)
    graph_client.run(SEED)
    graph_client.run(MARKER)
    rows = graph_client.run(COUNTS)
    relationships = graph_client.run("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    for row in rows:
        print(f"  {row['label']:<11} {row['count']}")
    print(f"Loaded {sum(r['count'] for r in rows)} nodes and {relationships} relationships.")
    graph_client.close()


if __name__ == "__main__":
    main()
