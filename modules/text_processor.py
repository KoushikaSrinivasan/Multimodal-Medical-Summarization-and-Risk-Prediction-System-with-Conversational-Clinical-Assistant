"""
Medical NER using BioBERT (d4data/biomedical-ner-all).
Extracts diseases, medications, symptoms, and lab values from clinical text.
"""

import re
from typing import Any
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from config import NER_MODEL


_ner_pipeline = None


def _get_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Run BioBERT NER on clinical text and return categorized entities.
    Returns dict with keys: diseases, medications, symptoms, labs, other.
    """
    ner = _get_pipeline()

    # BioBERT NER works best on shorter chunks; split on paragraphs
    chunks = _chunk_text(text, max_chars=512)
    raw_entities: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_entities.extend(ner(chunk))

    entities: dict[str, list[str]] = {
        "diseases": [],
        "medications": [],
        "symptoms": [],
        "labs": [],
        "other": [],
    }

    for ent in raw_entities:
        word = ent["word"].strip()
        label = ent.get("entity_group", ent.get("entity", "")).upper()
        if not word or len(word) < 2:
            continue

        if any(k in label for k in ["DIS", "DISEASE", "CONDITION", "DISO"]):
            entities["diseases"].append(word)
        elif any(k in label for k in ["CHEM", "DRUG", "MED", "PHARM"]):
            entities["medications"].append(word)
        elif any(k in label for k in ["SIGN", "SYMP", "FINDING"]):
            entities["symptoms"].append(word)
        elif any(k in label for k in ["LAB", "TEST", "PROC"]):
            entities["labs"].append(word)
        else:
            entities["other"].append(word)

    # Deduplicate while preserving order
    for key in entities:
        seen: set[str] = set()
        deduped = []
        for item in entities[key]:
            normalized = item.lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(item)
        entities[key] = deduped

    # Also run regex-based medication extraction as backup
    entities["medications"] = _merge_unique(
        entities["medications"], _regex_medications(text)
    )
    entities["labs"] = _merge_unique(
        entities["labs"], _regex_labs(text)
    )

    return entities


def _chunk_text(text: str, max_chars: int = 512) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += " " + para
        else:
            if current:
                chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]


def _regex_medications(text: str) -> list[str]:
    pattern = re.compile(
        r"\b([A-Z][a-z]+(?:mab|nib|pril|sartan|olol|statin|mycin|cillin|oxacin|azole|prazole|dipine|tidine|vir|mide|lone|pine|zine|ine|ene))\b"
    )
    return list({m.group(1) for m in pattern.finditer(text)})


def _regex_labs(text: str) -> list[str]:
    lab_keywords = [
        "HbA1c", "hemoglobin", "WBC", "RBC", "platelet", "creatinine",
        "glucose", "sodium", "potassium", "cholesterol", "triglyceride",
        "ALT", "AST", "bilirubin", "albumin", "eGFR", "BUN", "INR", "TSH",
    ]
    found = []
    lower = text.lower()
    for lab in lab_keywords:
        if lab.lower() in lower:
            found.append(lab)
    return found


def _merge_unique(base: list[str], additions: list[str]) -> list[str]:
    seen = {x.lower() for x in base}
    result = list(base)
    for item in additions:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def extract_patient_meta(text: str) -> dict:
    """
    Auto-extract patient age, hospital stay duration, and prior admissions
    from clinical text using regex patterns.
    Returns only keys that were successfully detected.
    """
    meta = {}

    # Age: "68-year-old", "68 year old", "age: 68", "aged 68"
    age_match = re.search(r"\b(\d{1,3})[- ]year[- ]old", text, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r"\bage[d]?\s*[:\-]?\s*(\d{1,3})\b", text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
        if 0 < age < 120:
            meta["age"] = age

    # Time in hospital: difference between admission and discharge dates
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if len(dates) >= 2:
        try:
            from datetime import datetime
            admission_dt = datetime.strptime(dates[0], "%Y-%m-%d")
            discharge_dt = datetime.strptime(dates[1], "%Y-%m-%d")
            days = (discharge_dt - admission_dt).days
            if 0 < days < 180:
                meta["time_in_hospital"] = days
        except ValueError:
            pass

    # Prior admissions: count explicit mentions
    prior_patterns = [
        r"(\d+)\s*(?:previous|prior|past)\s*(?:hospital|admission|hospitalization)",
        r"(?:previous|prior|past)\s*(?:hospital|admission|hospitalization)[^.]*?(\d+)\s*(?:time|occasion)",
    ]
    for pat in prior_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["num_prior_admissions"] = int(m.group(1))
            break
    else:
        # Count generic mentions as a rough proxy
        count = len(re.findall(
            r"previous hospitalization|prior admission|past admission|prior hospitalization",
            text, re.IGNORECASE
        ))
        if count > 0:
            meta["num_prior_admissions"] = count

    return meta
