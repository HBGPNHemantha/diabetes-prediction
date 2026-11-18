# Diabetes Prediction Using Machine Learning in R

A machine learning project that predicts diabetes outcomes using the **Pima Indians Diabetes Dataset**.

## Project Structure

```
Diabetes-prediction/
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes Dataset
├── code/
│   ├── 01_import_check.R     # Import, cleaning, EDA, visualization, correlation
│   └── 02_model.R            # Full pipeline: preprocessing → models → evaluation
├── figures/                  # Auto-generated plots
│   ├── 01_outcome_distribution.png
│   ├── 02_glucose_boxplot.png
│   ├── 03_bmi_boxplot.png
│   ├── 04_age_distribution.png
│   ├── 05_correlation_heatmap.png
│   └── 06_feature_importance.png
├── Diabetes-prediction.Rproj
└── README.md
```

## How to Run

1. Open `Diabetes-prediction.Rproj` in **RStudio**
2. Install required packages (once):
   ```r
   install.packages(c("caret", "randomForest", "ggplot2", "corrplot", "dplyr", "e1071"))
   ```
3. Run `code/02_model.R` — it covers the full pipeline end-to-end

> `01_import_check.R` contains the original step-by-step exploration code.

## Dataset

- **Source:** [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Target:** `Outcome` — 0 (No Diabetes) / 1 (Diabetes)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Patient age (years) |
| Outcome | 0 = No Diabetes, 1 = Diabetes |

## Methodology

| Step | Description |
|---|---|
| 1. Import | Load CSV, assign column names |
| 2. Cleaning | Replace invalid zeros with NA → median imputation |
| 3. EDA | Summary stats, outcome distribution, group means |
| 4. Visualization | Bar chart, boxplots, histogram, correlation heatmap |
| 5. Split | 80% train / 20% test (stratified) |
| 6. Models | Logistic Regression + Random Forest (5-fold CV) |
| 7. Evaluation | Accuracy, Sensitivity, Specificity, F1, AUC-ROC |
| 8. Importance | Feature importance from Random Forest |

## Models Used

- **Logistic Regression** — Baseline linear classifier
- **Random Forest** — Ensemble tree-based classifier

## Key Findings

- **Glucose** and **BMI** are the strongest predictors of diabetes
- Random Forest typically achieves ~**77–80% accuracy** on this dataset
- 5-fold cross-validation is used to avoid overfitting

## Requirements

- R ≥ 4.0
- Packages: `caret`, `randomForest`, `ggplot2`, `corrplot`, `dplyr`, `e1071`

---

*Academic project — not for clinical diagnosis.*

