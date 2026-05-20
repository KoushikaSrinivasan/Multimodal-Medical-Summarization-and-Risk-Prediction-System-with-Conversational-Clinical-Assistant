import os
from dotenv import load_dotenv

load_dotenv()

NER_MODEL = "d4data/biomedical-ner-all"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "google/flan-t5-base"
XRAY_MODEL = "densenet121-res224-nih"  # torchxrayvision model tag

XRAY_CONFIDENCE_THRESHOLD = 0.3  # findings above this are considered positive
RISK_MODEL_PATH = "train/risk_model.json"

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

# Mapping from clinical text terms → NIH Chest X-ray label
# Used by consistency_scorer.py for cross-modal alignment
TEXT_TO_XRAY_LABEL = {
    "atelectasis": "Atelectasis",
    "collapsed lung": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    "consolidation": "Consolidation",
    "pulmonary consolidation": "Consolidation",
    "edema": "Edema",
    "pulmonary edema": "Edema",
    "fluid overload": "Edema",
    "pleural effusion": "Effusion",
    "effusion": "Effusion",
    "emphysema": "Emphysema",
    "copd": "Emphysema",
    "hyperinflation": "Emphysema",
    "fibrosis": "Fibrosis",
    "pulmonary fibrosis": "Fibrosis",
    "interstitial fibrosis": "Fibrosis",
    "hernia": "Hernia",
    "hiatal hernia": "Hernia",
    "diaphragmatic hernia": "Hernia",
    "infiltrate": "Infiltration",
    "infiltration": "Infiltration",
    "lung infiltrate": "Infiltration",
    "mass": "Mass",
    "lung mass": "Mass",
    "tumor": "Mass",
    "pulmonary mass": "Mass",
    "nodule": "Nodule",
    "pulmonary nodule": "Nodule",
    "lung nodule": "Nodule",
    "pleural thickening": "Pleural_Thickening",
    "pleural scarring": "Pleural_Thickening",
    "pneumonia": "Pneumonia",
    "lobar pneumonia": "Pneumonia",
    "community acquired pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "collapsed lung": "Pneumothorax",
    "air in pleural space": "Pneumothorax",
}
