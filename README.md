# Multimodal Medical Summarization and Risk Prediction System with Conversational Clinical Assistant

An end-to-end clinical AI system that combines NLP, medical imaging, risk prediction, and conversational AI into a unified pipeline for patient report analysis.

---

## Novel Contribution

**Cross-Modal Clinical Consistency Scoring** — automatically detects mismatches between findings documented in clinical text and findings detected in chest X-ray images. Radiological findings present in the X-ray but absent from clinical notes are flagged as potential missed diagnoses or documentation errors. This is a patient-safety contribution not addressed by prior systems.

---

## Architecture

```
Clinical Text + X-Ray Image
        ↓
┌─────────────────────────────────┐
│  1. NER  (BioBERT)              │  → diseases, meds, symptoms, labs
│  2. X-Ray CNN (torchxrayvision) │  → 14 pathology confidence scores
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  3. Risk Prediction  (XGBoost)  │  → readmission risk + severity
│  4. Drug Interaction Detection  │  → drug-drug conflicts, allergy alerts
│  5. Consistency Scoring (NOVEL) │  → text vs image agreement + missed findings
│  6. Summarization (Claude API)  │  → doctor summary + patient summary
│  7. SHAP Explainability         │  → why this risk score?
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  8. RAG Chatbot (Claude API)    │  → Q&A grounded in patient record
└─────────────────────────────────┘
        ↓
   FastAPI Backend + Streamlit UI
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Medical NER | BioBERT (`d4data/biomedical-ner-all`) |
| Summarization | Claude API (`claude-sonnet-4-6`) |
| X-Ray Analysis | torchxrayvision DenseNet (NIH pre-trained) |
| Risk Prediction | XGBoost |
| Explainability | SHAP |
| Drug Interactions | Curated JSON (DrugBank-derived) |
| Conversational AI | Claude API + RAG |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## Datasets

| Purpose | Dataset | Access |
|---------|---------|--------|
| Clinical text | MTSamples | [mtsamples.com](https://www.mtsamples.com) |
| X-ray training | NIH Chest X-ray (14 labels) | [NIH via Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| Risk prediction | Diabetes 130-US Hospitals | [UCI ML Repository](https://archive.ics.uci.edu/dataset/296) |
| Drug interactions | DrugBank + SIDER | [drugbank.com](https://go.drugbank.com) |
| Chatbot KB | PubMed Abstracts | [HuggingFace](https://huggingface.co/datasets/pubmed) |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/KoushikaSrinivasan/Multimodal-Medical-Summarization-and-Risk-Prediction-System-with-Conversational-Clinical-Assistant.git
cd Multimodal-Medical-Summarization-and-Risk-Prediction-System-with-Conversational-Clinical-Assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up API key
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the Streamlit app
```bash
streamlit run ui/app.py
```

### 5. (Optional) Run the FastAPI backend separately
```bash
python api/main.py
# API docs available at http://localhost:8000/docs
```

### 6. (Optional) Train the XGBoost risk model
```bash
# Download diabetic_data.csv from UCI and place in data/
python train/train_risk_model.py
```

---

## Project Structure

```
├── modules/
│   ├── text_processor.py      # BioBERT NER — extract clinical entities
│   ├── summarizer.py          # Claude API — doctor + patient summaries
│   ├── xray_analyzer.py       # torchxrayvision — 14-class X-ray classification
│   ├── risk_predictor.py      # XGBoost — readmission and emergency risk
│   ├── drug_detector.py       # Drug-drug interaction and allergy detection
│   ├── consistency_scorer.py  # NOVEL — cross-modal text vs X-ray consistency
│   ├── chatbot.py             # Claude API RAG — patient/doctor Q&A
│   └── explainer.py           # SHAP — explain risk predictions
├── api/
│   └── main.py                # FastAPI REST backend
├── ui/
│   └── app.py                 # Streamlit frontend
├── train/
│   └── train_risk_model.py    # XGBoost training on Diabetes 130-US dataset
├── data/
│   ├── drug_interactions.json # Drug interaction knowledge base
│   └── sample/
│       └── sample_report.txt  # Example discharge summary for testing
├── config.py                  # Model configs, NIH labels, entity→label mapping
├── requirements.txt
└── .env.example
```

---

## Research Contributions

1. **Cross-modal clinical consistency scoring** — novel Jaccard-based alignment between NER entities and radiological findings, with automated missed-finding alerts
2. **Integrated multimodal pipeline** — text + image + tabular features in a unified system
3. **Dual-audience summarization** — separate summaries for clinician and patient
4. **Explainable risk prediction** — SHAP values alongside each risk score
5. **RAG-grounded conversational assistant** — answers grounded strictly in the patient's own record

### Suggested Paper Titles
- *"Cross-Modal Clinical Consistency Scoring for Automated Missed Diagnosis Detection in Multimodal Medical AI"*
- *"Multimodal Medical Summarization and Risk Prediction using Transformer Networks and Conversational AI"*
- *"An Explainable Clinical Decision Support System using NLP, Medical Imaging, and Conversational Intelligence"*

### Target Venues
- IEEE Journal of Biomedical and Health Informatics (JBHI)
- Springer Medical Informatics
- ACL BioNLP Workshop
- AAAI Health Intelligence Workshop

---

## Running a Demo

1. Start the app: `streamlit run ui/app.py`
2. Open the **Input** tab
3. Paste or upload a clinical report (use `data/sample/sample_report.txt` to test)
4. Optionally upload a chest X-ray image
5. Click **Analyze**
6. Explore results across the Summary, Risk, X-Ray, Drugs, Consistency, and Assistant tabs

---

## License

LGPL-2.1 — see [LICENSE](LICENSE)