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
        word = _clean_token(ent["word"])
        if not word:
            continue
        label = ent.get("entity_group", ent.get("entity", "")).upper()

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

    # Deduplicate and clean each category
    for key in entities:
        entities[key] = _dedup_clean(entities[key])

    # Regex-based backup extraction for medications and labs
    entities["medications"] = _merge_unique(
        entities["medications"], _regex_medications(text)
    )
    entities["labs"] = _merge_unique(
        entities["labs"], _regex_labs(text)
    )

    # Final clean pass on all categories
    for key in entities:
        entities[key] = _dedup_clean(entities[key])

    return entities


# Generic noise words that are never valid medical entities
_NOISE_WORDS = {
    "medications", "multiple", "no", "none", "not", "yes", "also",
    "the", "and", "or", "of", "in", "on", "for", "with", "by",
    "was", "were", "has", "had", "have", "is", "are", "be",
    "per", "mg", "ml", "dl", "iv", "po", "prn", "bid", "tid",
    "once", "twice", "daily", "weekly", "reduced", "elevated",
    "normal", "abnormal", "negative", "positive", "mild", "severe",
    "moderate", "bilateral", "right", "left", "upper", "lower",
    "history", "review", "control", "monitoring", "scale", "dose",
    "unit", "units", "injection", "infusion", "therapy", "treatment",
    "admission", "discharge", "follow", "continued", "initiated",
    "ray", "scan", "test", "result", "culture", "level", "output",
    "resuscitation", "vegetation", "sliding", "fluid", "blood",
}


def _clean_token(word: str) -> str:
    """
    Remove BioBERT WordPiece artifacts and invalid tokens.
    Returns empty string if the token should be discarded.
    """
    word = word.strip()

    # Drop subword fragments (WordPiece artifacts)
    if word.startswith("##") or word.startswith(" ##"):
        return ""

    # Remove any remaining ## inside the word
    word = word.replace("##", "").strip()

    # Drop very short tokens (single chars, 2-char abbreviations that aren't labs)
    if len(word) <= 2:
        return ""

    # Drop pure numbers or tokens that are mostly numeric
    if re.match(r"^[\d\.\,\/\-\+\^\s]+$", word):
        return ""

    # Drop noise words (case-insensitive)
    if word.lower() in _NOISE_WORDS:
        return ""

    # Drop tokens with too many special characters (garbage)
    special_ratio = sum(1 for c in word if not c.isalnum() and c not in " -") / max(len(word), 1)
    if special_ratio > 0.4:
        return ""

    # Title-case for consistency if all lower
    if word.islower() and len(word) > 3:
        word = word.title()

    return word


def _dedup_clean(items: list[str]) -> list[str]:
    """Deduplicate while preserving order, case-insensitively."""
    seen: set[str] = set()
    result = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


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
    Auto-extract patient age, hospital stay duration, and prior admissions.
    Supports multiple date formats commonly found in clinical reports.
    Returns only keys that were successfully detected.
    """
    from datetime import datetime

    meta = {}

    # ── Age ──────────────────────────────────────────────────────────────────
    # Patterns: "68-year-old", "68 year old", "age: 68", "aged 68", "Age/Sex: 68/M"
    age_patterns = [
        r"\b(\d{1,3})[- ]year[- ]old",
        r"\bage[d]?\s*[:/\-]?\s*(\d{1,3})\b",
        r"\bage\s*/\s*sex\s*[:/]\s*(\d{1,3})",
        r"\b(\d{1,3})\s*(?:yr|yrs|y\.o)\b",
    ]
    for pat in age_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                meta["age"] = age
                break

    # ── Date parsing (multi-format) ──────────────────────────────────────────
    MONTHS = (
        "january|february|march|april|may|june|july|august|september|"
        "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    )

    DATE_PATTERNS = [
        # YYYY-MM-DD  e.g. 2026-05-01
        (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d", lambda m: m.group(0)),
        # DD/MM/YYYY  e.g. 01/05/2026
        (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "%d/%m/%Y", lambda m: m.group(0)),
        # DD-MM-YYYY  e.g. 01-05-2026
        (r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b", "%d-%m-%Y", lambda m: m.group(0)),
        # DD Month YYYY  e.g. 10 April 2026 or 10th April 2026
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\s+(\d{{4}})\b",
         "%d %B %Y", lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
        # Month DD, YYYY  e.g. April 10, 2026
        (rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
         "%B %d %Y", lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
        # Month YYYY  e.g. April 2026 (day assumed 1st — less precise)
        (rf"\b({MONTHS})\s+(\d{{4}})\b",
         "%B %Y", lambda m: f"{m.group(1)} {m.group(2)}"),
    ]

    def _parse_dates(text: str) -> list[datetime]:
        """Extract all recognisable dates from text, preserving document order."""
        found: list[tuple[int, datetime]] = []  # (position, datetime)
        for pattern, fmt, extractor in DATE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw = extractor(m)
                try:
                    dt = datetime.strptime(raw, fmt)
                    found.append((m.start(), dt))
                except ValueError:
                    continue
        # Sort by position, deduplicate nearby dates (within 5 chars)
        found.sort(key=lambda x: x[0])
        deduped: list[datetime] = []
        last_pos = -10
        for pos, dt in found:
            if pos - last_pos > 5:
                deduped.append(dt)
                last_pos = pos
        return deduped

    # Try to find admission/discharge dates explicitly by label
    def _labeled_date(label_pattern: str, text: str) -> datetime | None:
        m = re.search(label_pattern, text, re.IGNORECASE)
        if not m:
            return None
        snippet = text[m.end(): m.end() + 40]
        dates = _parse_dates(snippet)
        return dates[0] if dates else None

    admission_dt = _labeled_date(r"date\s+of\s+admission\s*[:/]?", text)
    discharge_dt = _labeled_date(r"date\s+of\s+discharge\s*[:/]?", text)

    # Fallback: use first two dates found in the document
    if not admission_dt or not discharge_dt:
        all_dates = _parse_dates(text)
        if len(all_dates) >= 2 and not admission_dt:
            admission_dt = all_dates[0]
        if len(all_dates) >= 2 and not discharge_dt:
            discharge_dt = all_dates[1]

    if admission_dt and discharge_dt and discharge_dt > admission_dt:
        days = (discharge_dt - admission_dt).days
        if 0 < days < 180:
            meta["time_in_hospital"] = days

    # ── Prior admissions ─────────────────────────────────────────────────────
    prior_patterns = [
        r"(\d+)\s*(?:previous|prior|past)\s*(?:hospital|admission|hospitalization)",
        r"(?:previous|prior|past)\s*(?:hospital|admission|hospitalization)[^.]{0,40}?(\d+)\s*(?:time|occasion)",
        r"admitted\s+(\d+)\s*(?:time|occasion)",
    ]
    for pat in prior_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["num_prior_admissions"] = int(m.group(1))
            break
    else:
        count = len(re.findall(
            r"previous hospitalization|prior admission|past admission|prior hospitalization",
            text, re.IGNORECASE,
        ))
        if count > 0:
            meta["num_prior_admissions"] = count

    return meta
