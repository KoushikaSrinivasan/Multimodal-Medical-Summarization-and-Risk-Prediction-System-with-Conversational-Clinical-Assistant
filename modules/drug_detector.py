"""
Drug interaction and allergy conflict detection.
Uses a curated drug_interactions.json database derived from DrugBank/SIDER.
"""

import json
from itertools import combinations
from pathlib import Path


_DATA_PATH = Path(__file__).parent.parent / "data" / "drug_interactions.json"
_db: dict | None = None


def _get_db() -> dict:
    global _db
    if _db is None:
        with open(_DATA_PATH, "r") as f:
            _db = json.load(f)
    return _db


def detect_interactions(medications: list[str], patient_allergies: list[str] | None = None) -> dict:
    """
    Check all medication pairs for known interactions and allergy conflicts.

    Returns:
        {
          "interactions": [{"drugs": [...], "severity": ..., "effect": ..., "recommendation": ...}],
          "allergy_conflicts": [...],
          "highest_severity": "MAJOR" | "MODERATE" | "CONTRAINDICATED" | "NONE",
          "safe": bool,
        }
    """
    db = _get_db()
    meds_lower = [m.lower().strip() for m in medications]

    interactions = _check_interactions(db["interactions"], meds_lower)
    allergy_conflicts = _check_allergies(
        db["allergy_alerts"], meds_lower, patient_allergies or []
    )

    all_severities = [i["severity"] for i in interactions] + [
        a["severity"] for a in allergy_conflicts
    ]
    highest = _highest_severity(all_severities)

    return {
        "interactions": interactions,
        "allergy_conflicts": allergy_conflicts,
        "highest_severity": highest,
        "safe": highest == "NONE",
        "total_medications": len(medications),
    }


def _check_interactions(interaction_db: list[dict], meds: list[str]) -> list[dict]:
    found = []
    for record in interaction_db:
        drug_a = record["drug_a"].lower()
        drug_b = record["drug_b"].lower()

        a_present = any(drug_a in m or m in drug_a for m in meds)
        b_present = any(drug_b in m or m in drug_b for m in meds)

        if a_present and b_present:
            found.append({
                "drugs": [record["drug_a"], record["drug_b"]],
                "severity": record["severity"],
                "effect": record["effect"],
                "recommendation": record["recommendation"],
            })

    return found


def _check_allergies(allergy_db: list[dict], meds: list[str], known_allergies: list[str]) -> list[dict]:
    conflicts = []
    allergies_lower = [a.lower() for a in known_allergies]

    for record in allergy_db:
        drug_class = record["drug_class"].lower()
        cross_reactive = [d.lower() for d in record["cross_reactive"]]

        # Check if patient is allergic to the drug class or any cross-reactive member
        allergic = any(drug_class in a or a in drug_class for a in allergies_lower)
        allergic = allergic or any(cr in a for cr in cross_reactive for a in allergies_lower)

        if not allergic:
            continue

        # Check if any prescribed medication is in this class
        for med in meds:
            if any(cr in med or med in cr for cr in cross_reactive):
                conflicts.append({
                    "medication": med,
                    "conflict_class": record["drug_class"],
                    "severity": "MAJOR",
                    "note": record["note"],
                })

    return conflicts


def _highest_severity(severities: list[str]) -> str:
    order = {"CONTRAINDICATED": 4, "MAJOR": 3, "MODERATE": 2, "MINOR": 1, "NONE": 0}
    if not severities:
        return "NONE"
    return max(severities, key=lambda s: order.get(s, 0))
