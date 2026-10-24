# Explainable Multimodal **Brain Health** Classification Using MRI and Clinical Data

## Project Overview

This project presents an explainable multimodal machine learning approach for Alzheimer’s-related cognitive impairment classification using brain MRI images and clinical/demographic data from the OASIS-1 dataset.

The system combines deep learning-based MRI analysis with a clinical machine learning model to generate a fused prediction. It also includes model performance evaluation, Grad-CAM-based visual explainability, and an interactive dashboard for exploring model outputs.

## Objectives

- Analyze brain MRI images using CNN-based deep learning models.
- Use clinical and demographic features for cognitive impairment classification.
- Combine MRI and clinical predictions using weighted late fusion.
- Evaluate model performance using standard classification metrics.
- Explore model attention using Grad-CAM.
- Present the results through an interactive dashboard.

## Dataset

**Dataset:** OASIS-1 (Open Access Series of Imaging Studies)

The project uses MRI images and associated clinical/demographic information.

The classification target is derived from the Clinical Dementia Rating (CDR):

- **Class 0:** CDR = 0
- **Class 1:** CDR > 0

The project workflow includes data preparation, label checking, subject-level splitting, clinical data alignment, and MRI preprocessing.

> Note: OASIS data may have access, licensing, and usage conditions. Obtain the dataset from its authorized source and follow the applicable terms. Do not commit restricted or sensitive patient-level data to a public repository.

## Methodology

### 1. Data Preparation
- Prepare MRI images and metadata.
- Check labels and subject-level data consistency.
- Create training, validation, and test splits.
- Align clinical records and MRI predictions by subject.

### 2. MRI-Based Classification
The project explores the following CNN architectures:

- ResNet50
- DenseNet121
- EfficientNetB0
- EfficientNetV2B0

Slice-level predictions are aggregated for subject-level evaluation.

### 3. Clinical Classification
A Random Forest classifier is used with clinical and demographic predictors, including:

- Age
- Education
- Socioeconomic status (SES)
- Estimated total intracranial volume (eTIV)
- Normalized whole-brain volume (nWBV)
- Atlas scaling factor (ASF)
- Sex

The primary clinical workflow excludes MMSE to reduce the risk of target leakage. An MMSE-inclusive version may be used for comparison.

### 4. Multimodal Fusion
MRI and clinical predictions are combined using weighted late fusion.

Reported configuration:

- MRI weight: 0.40
- Clinical weight: 0.60
- Classification threshold: 0.43

These values should be checked against the final implementation before reproducing results.

### 5. Explainability
Grad-CAM is used to visualize regions influencing CNN predictions. These visualizations are exploratory and do not establish clinical validity or prove that a prediction is based on disease-specific brain regions.

### 6. Dashboard
The project includes an interactive dashboard for examining project information, MRI analysis, clinical analysis, multimodal fusion, and model performance.

## Reported Results

The project report/code output includes the following multimodal test metrics:

| Metric | Reported value |
|---|---:|
| Accuracy | 80.61% |
| Precision | 70.59% |
| Recall | 80.00% |
| F1-score | 0.75 |
| ROC-AUC | 0.8032 |

Reported confusion matrix:

```text
[[16, 5],
 [ 3, 12]]
```

**Result verification note:** The confusion matrix totals 36 subjects and gives 29 correct predictions, corresponding to approximately 80.56% accuracy. This differs slightly from the printed accuracy of 80.61%; verify the final evaluation outputs before presenting these figures as definitive results.

These results are experimental and should not be interpreted as evidence of clinical diagnostic performance.

## Project Structure

The repository may contain the following components, depending on which files are included in the final upload:

```text
HA_DATASET/
├── code/
│   ├── fusion/
│   │   ├── fusion.py
│   │   ├── train_clinical_aligned.py
│   │   └── generate_clinical_predictions.py
│   ├── train_cnn.py
│   ├── train_finetune.py
│   ├── train_improved.py
│   ├── evaluate_ensemble.py
│   ├── subject_level_evaluation.py
│   ├── prepare_splits.py
│   ├── label_mri_dataset.py
│   ├── merge_mri_datasets.py
│   ├── gradcam_analysis.py
│   └── analyze_results.py
├── dashboard/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

This is an illustrative structure. Adjust it to match the files actually committed to the repository.

## Installation

1. Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Create `requirements.txt` based on the libraries used by the final code. Depending on the implementation, these may include Python packages for data processing, machine learning, deep learning, visualization, and the dashboard.

## Running the Project

Run commands from the repository root and adjust paths according to your final folder structure.

### Run multimodal fusion

```bash
python code/fusion/fusion.py
```

### Run the dashboard

If the dashboard uses Streamlit:

```bash
streamlit run dashboard/app.py
```

The dashboard may require trained model files, prediction CSVs, and configuration files. Ensure these dependencies are available at the paths expected by the application.

## Key Features

- Multimodal MRI and clinical data analysis
- CNN-based MRI classification
- Random Forest clinical classification
- Subject-level prediction evaluation
- Weighted late fusion
- Grad-CAM visualization
- Performance metrics and confusion matrix
- Interactive dashboard

## Limitations

- The project is based on the OASIS-1 dataset and may not generalize to other populations or clinical settings.
- Dataset size, class balance, image preprocessing, and split strategy can influence reported performance.
- Grad-CAM visualizations are not clinical explanations or proof of causality.
- The system is a research prototype, not a substitute for professional medical assessment.
- Results must be reproduced and independently validated before any clinical use.

## Future Scope

- External validation on independent datasets.
- Improved MRI preprocessing and subject-level aggregation.
- More robust multimodal fusion strategies.
- Better calibration and uncertainty estimation.
- Further explainability and bias analysis.
- Clinical collaboration and prospective validation.

## Responsible Use

This project is intended for educational and research purposes only. It is not designed or approved to diagnose Alzheimer’s disease or make clinical decisions.

## Acknowledgements

The project uses the OASIS-1 dataset. Please cite the appropriate dataset publication and follow the dataset provider’s attribution and usage requirements.

## License

Add a license only after confirming that you have the rights to distribute the project code and that the license is compatible with all included components and dependencies.
