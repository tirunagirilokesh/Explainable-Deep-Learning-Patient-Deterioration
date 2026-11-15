# Explainable Deep Learning for Early Prediction of Patient Deterioration

### An Explainable Deep Learning Approach for Early Prediction of Intraoperative Hypotension

## Project Overview

This project develops an Explainable Artificial Intelligence (XAI)-based early warning system for predicting patient deterioration during surgery.

The system analyzes historical intraoperative physiological signals and predicts the risk of future hypotension across multiple prediction horizons. It uses deep learning techniques, particularly Long Short-Term Memory (LSTM) networks, to learn temporal patterns in patient vital signs.

Explainability techniques are incorporated to help interpret individual model predictions.

**This project is an academic research prototype and is not intended for direct clinical decision-making.**

## Objectives

- Develop an early warning system for predicting intraoperative patient deterioration.
- Predict hypotension before its occurrence.
- Analyze multiple physiological signals over a 30-minute historical window.
- Generate predictions across multiple future time horizons.
- Evaluate ANN and LSTM-based models.
- Apply explainable AI techniques to interpret model predictions.

## Dataset

The project uses the VitalDB intraoperative dataset.

Five physiological signals are selected:

| Signal | Description |
|---|---|
| ART_MBP | Mean Arterial Blood Pressure (MAP) |
| ART_SBP | Systolic Blood Pressure |
| ART_DBP | Diastolic Blood Pressure |
| HR | Heart Rate |
| PLETH_SPO2 | Peripheral Oxygen Saturation |

### Hypotension Definition

Intraoperative hypotension is defined as:

**Mean Arterial Pressure (MAP) below 65 mmHg sustained for at least one minute.**

### Input Configuration

- Sampling interval: 2 seconds
- Historical input duration: 30 minutes
- Input sequence: 900 observations × 5 signals
- Prediction horizons: 5, 10, 15 and 30 minutes

The dataset is divided at patient level to prevent patient overlap between training, validation and testing.

| Dataset | Patients | Windows |
|---|---:|---:|
| Training | 187 | 19,441 |
| Validation | 41 | 4,151 |
| Testing | 39 | 4,012 |
| **Total** | **267** | **27,604** |

Raw patient recordings and processed patient-level datasets are not included in this repository.

## Methodology

The project workflow consists of:

1. VitalDB data extraction
2. Physiological signal inspection and validation
3. Hypotension event identification
4. Patient selection and splitting
5. Time-series window generation
6. Training-only Z-score normalization
7. ANN baseline training
8. Multi-horizon LSTM training
9. Model evaluation
10. Explainability analysis using LIME and counterfactual explanations

## Deep Learning Models

### 1. Artificial Neural Network (ANN)

ANN models are used as baseline approaches for comparison.

### 2. Long Short-Term Memory (LSTM)

The primary model uses an LSTM architecture to learn temporal patterns in physiological signals.

The model produces four outputs corresponding to:

- 5-minute prediction horizon
- 10-minute prediction horizon
- 15-minute prediction horizon
- 30-minute prediction horizon

## Independent Test Results

The following results were obtained from the independent test set using raw model probabilities.

| Metric | 5 min | 10 min | 15 min | 30 min |
|---|---:|---:|---:|---:|
| AUROC | 0.8395 | 0.7954 | 0.7572 | 0.7449 |
| AUPRC | 0.1156 | 0.1656 | 0.1830 | 0.2818 |
| Brier Score | 0.1453 | 0.1748 | 0.1867 | 0.1967 |
| Positive Prevalence | 2.19% | 4.11% | 5.83% | 11.99% |

**Evaluation note:** The raw Brier scores are higher than the corresponding prevalence-baseline Brier scores. This indicates that probability calibration remains an important limitation. The results should not be interpreted as evidence of clinically reliable risk probabilities.

## Explainable Artificial Intelligence

The project incorporates two explainability approaches:

- **LIME (Local Interpretable Model-Agnostic Explanations):** Investigates which physiological signal segments contribute to individual predictions.
- **Counterfactual Explanations:** Examines how changes to selected physiological inputs affect model predictions.

These explanations describe model behavior and should not be interpreted as clinical recommendations or treatment interventions.

## Repository Structure

Patient-Deterioration-Prediction/
│
├── notebooks/
│   └── final_lstm_workflow_clean.ipynb
│
├── src/
│   ├── data_preprocessing/
│   │   └── preprocessing and validation scripts
│   │
│   └── models/
│       ├── train_ann.py
│       ├── train_ann_v2.py
│       ├── calibrate_ann_models.py
│       ├── compare_ann_models.py
│       └── train_lstm.py
│
├── inspect_vitaldb_records.py
├── requirements.txt
├── requirements-notebook.txt
├── .gitignore
└── README.md