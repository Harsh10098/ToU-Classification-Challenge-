# Diabetes Risk Prediction: Project Report

This repository contains an exploratory and modelling study on early diabetes risk identification, supported by a Jupyter notebook workflow and a small Flask application for interactive inference. The project is positioned as a preventive screening aid rather than a diagnostic system.

## Executive Summary

The central question is whether routinely collected clinical indicators can support earlier identification of diabetes risk in settings where formal diagnosis often happens late. The analysis uses a diabetes dataset with `768` records and `8` predictor variables, compares several classical machine learning models, and exports a tuned `RandomForestClassifier` for use in a web app.

The evidence in the notebooks shows three clear conclusions:

- diabetes risk is not cleanly separable by any single feature; the strongest signal comes from multivariate patterns, especially involving `Glucose`, `BMI`, and `Age`
- the positive class is materially harder to recover than the negative class, so accuracy alone is not sufficient for model selection
- the final exported model is usable as a screening or triage aid, but not as a diagnostic decision-maker, and it should be threshold-tuned for recall if used in preventive settings

A supplementary evaluation of the exported model artifact shows:

- hold-out accuracy: `72.92%`
- precision: `64.71%`
- recall: `49.25%`
- F1: `55.93%`
- ROC-AUC: `0.814`

That default operating point is too conservative for a screening workflow. Lowering the decision threshold improves recall substantially with only a modest drop in accuracy:

- threshold `0.50`: recall `49.25%`, accuracy `72.92%`
- threshold `0.40`: recall `67.16%`, accuracy `73.96%`
- threshold `0.35`: recall `80.60%`, accuracy `72.40%`

For real-world screening, the model is therefore more defensible as a high-recall triage tool than as a binary yes/no classifier at the default threshold.

## 1. Problem Framing

### Target

The target variable is `Outcome`, where:

- `1` indicates diabetes present
- `0` indicates diabetes absent

### Stakeholders

The project is most relevant to:

- patients who may benefit from earlier identification of elevated risk
- primary-care clinicians and community health workers who need a quick triage aid
- public-health planners who must prioritize follow-up, outreach, and confirmatory testing

### Decision Scenarios

The most realistic decision settings for this model are:

- deciding which patients should be referred for confirmatory laboratory testing
- prioritizing lifestyle counselling or closer monitoring for higher-risk individuals
- supporting low-cost early-warning workflows when healthcare capacity is limited

The model should not be treated as a standalone diagnostic system, because false negatives are still material and the dataset is too narrow to justify broad clinical automation.

### Societal Relevance

The challenge framing in [report.ipynb](/Users/harshkyada/Desktop/ToU%20CC/report.ipynb) positions diabetes as a major public-health burden in India, where delayed diagnosis contributes to preventable complications and higher long-term system cost. That makes the problem directly relevant to:

- `SDG 3.4`: reducing premature mortality from non-communicable diseases
- `SDG 3.8`: improving access to quality healthcare
- `SDG 3.d`: strengthening early warning systems for health risks

## 2. Data Audit and Quality

The core dataset is [diabetes.csv](/Users/harshkyada/Desktop/ToU%20CC/Diabetes%20EDA/diabetes.csv).

### Dataset Profile

| Measure | Value |
|---|---:|
| Rows | 768 |
| Columns | 9 |
| Predictors | 8 |
| Negative class (`Outcome = 0`) | 500 |
| Positive class (`Outcome = 1`) | 268 |
| Positive rate | 34.9% |

The predictors are:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

### Data Quality Findings

The notebook correctly identifies that several variables contain clinically implausible zeros that are better interpreted as missing or placeholder measurements:

| Column | Zero count |
|---|---:|
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |
| `BMI` | 11 |

This is one of the most important analytical findings in the project: the data contains no literal null values, but it still has meaningful quality issues that would distort modelling if left untreated.

### Cleaning Decisions

The notebook intends to address these issues by:

- replacing implausible zeros in `Insulin` with the median
- replacing implausible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, and `BMI` with the mean
- standardizing features
- using a stratified train/test split

That logic is sensible in spirit, but the current implementation in [DSS_SDGs.ipynb](/Users/harshkyada/Desktop/ToU%20CC/Diabetes%20EDA/DSS_SDGs.ipynb) is not yet fully rigorous:

- the scaler is fit separately on train and test data instead of fitting on train once and only transforming test data
- zero replacement is performed after scaling, when it should happen before scaling

Those details matter for reproducibility and leakage control. A submission-quality production workflow should apply:

1. placeholder-value handling on raw training features
2. fitting any imputer on training data only
3. fitting the scaler on training data only
4. transforming validation and test sets using the frozen training-fitted objects
5. bundling preprocessing and model logic in a single pipeline artifact

## 3. Visual and Analytical Findings

The notebook provides broad exploratory coverage through:

- class balance plots
- histograms and boxplots for each feature
- density plots
- outcome-conditioned distributions
- a correlation heatmap

The most useful insights are not cosmetic; they directly inform modelling and deployment.

### Key Insights

`Glucose` is the strongest single signal. The notebook reports the highest correlation with the outcome at approximately `0.47`, substantially above the other variables.

`BMI`, `Age`, and `Pregnancies` provide secondary but still meaningful signal. The notebook reports approximate correlations of `0.29`, `0.24`, and `0.22` respectively.

`BloodPressure` and `SkinThickness` overlap heavily across classes. That overlap is important because it shows why threshold-style rules are too weak for this problem.

Spikes at zero in several variables reveal data-quality artifacts rather than true physiological states. This is a non-obvious but crucial insight, because the dataset would otherwise appear cleaner than it is.

The distributions suggest that many positive and negative cases occupy overlapping regions of feature space. This means the problem is structurally multivariate: diabetes status is better inferred from combinations of indicators than from any single cutoff.

### Fairness-Relevant Insight

The dataset does not contain strong protected-attribute coverage for a full fairness audit. There is no direct socioeconomic, geographic, or explicit sex variable, and the feature set is clinically narrow. However, an age-banded subgroup check on the exported model is still informative:

| Age band | Test samples | Positive rate | Accuracy | Positive-class recall |
|---|---:|---:|---:|---:|
| `18-34` | 130 | 26.9% | 75.38% | 34.29% |
| `35-49` | 41 | 56.1% | 75.61% | 73.91% |
| `50+` | 21 | 42.9% | 52.38% | 44.44% |

This suggests the current model is much better at recovering positives in the middle age band than among younger adults, even though screening value is often highest when risk is detected early. That makes subgroup monitoring a deployment requirement, not an optional extra.

## 4. Feature Strategy

The project uses a restrained feature strategy: it keeps the original eight clinically interpretable variables instead of layering on aggressive feature engineering.

That restraint is justified for three reasons:

- the dataset is small, so heavy engineered interactions risk instability and overfitting
- the predictors already map cleanly to clinical concepts, which helps stakeholder communication
- the deployed Flask form in [app.py](/Users/harshkyada/Desktop/ToU%20CC/Diabetes%20EDA/flask_app/app.py) expects these exact eight inputs

This is a good example of domain-aware restraint. The most valuable next features would not be arbitrary polynomial expansions; they would be targeted additions such as:

- missingness indicators for zero-placeholder fields
- calibrated risk scores rather than only hard class labels
- carefully selected interactions such as `Glucose x BMI` or `Age x Pregnancies`

Those should only be introduced after the preprocessing pipeline is corrected and frozen in a leakage-safe workflow.

## 5. Model Development and Comparison

The notebook compares five model families:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- XGBoost

### Reported Performance From The Notebook

| Model | Train accuracy | Test accuracy | 10-fold CV mean accuracy | Interpretation |
|---|---:|---:|---:|---|
| Logistic Regression | 79.51% | 73.44% | 78.82% | strongest generalization baseline |
| Decision Tree | 100.00% | 71.88% | 68.60% | severe overfitting |
| Random Forest | 100.00% | 74.48% | 75.88% | better than tree, but still overfit |
| KNN | 81.25% | 73.44% | 74.67% | competitive but less flexible |
| XGBoost | 100.00% | 75.52% | 73.81% | strong hold-out score, overfit in train |

### Hyperparameter Search

The notebook then runs grid search for Random Forest and XGBoost:

- tuned Random Forest best cross-validation accuracy: `77.61%`
- tuned XGBoost best cross-validation accuracy: `76.58%`

The exported model in [diabetes_model.pkl](/Users/harshkyada/Desktop/ToU%20CC/Diabetes%20EDA/diabetes_model.pkl) is a tuned `RandomForestClassifier` with:

- `n_estimators = 300`
- `max_features = 1`
- `min_samples_leaf = 10`
- `min_samples_split = 3`
- `bootstrap = False`
- `criterion = gini`

### Final Model Choice

There are two defensible narratives in this project:

- `LogisticRegression` is the strongest pure generalization baseline on cross-validation
- the tuned `RandomForestClassifier` is the chosen deployed model because it captures nonlinear structure, remains lightweight enough for Flask deployment, and offers straightforward feature importance analysis

The final Random Forest choice is therefore reasonable, but the evidence also shows the margin is not overwhelming. In a stricter production review, Logistic Regression should remain a live benchmark rather than being treated as conclusively inferior.

## 6. Evaluation and Real-World Behaviour

### Why Accuracy Alone Is Not Enough

Because the positive class is the operationally important class in screening, the model must be judged on:

- recall for `Outcome = 1`
- precision, to avoid excessive unnecessary follow-up
- F1, to summarize the trade-off
- ROC-AUC, to evaluate ranking quality
- subgroup behaviour, where possible

### Exported Model Evaluation

To understand the behaviour of the actual deployed artifact rather than only notebook cells, the saved model and scaler were re-evaluated on the recreated hold-out split used by the project:

| Metric | Exported Random Forest |
|---|---:|
| Accuracy | 72.92% |
| Precision | 64.71% |
| Recall | 49.25% |
| F1 | 55.93% |
| ROC-AUC | 0.814 |

Confusion matrix:

```text
[[107, 18],
 [ 34, 33]]
```

The ROC-AUC is encouraging, which means the model ranks risk reasonably well. The default threshold, however, is too conservative for early-warning use because it misses roughly half of true positives.

### Threshold Sensitivity

This is the most deployment-relevant behaviour in the project:

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `0.50` | 72.92% | 64.71% | 49.25% | 55.93% |
| `0.45` | 73.96% | 64.41% | 56.72% | 60.32% |
| `0.40` | 73.96% | 61.64% | 67.16% | 64.29% |
| `0.35` | 72.40% | 57.45% | 80.60% | 67.08% |
| `0.30` | 68.75% | 53.40% | 82.09% | 64.71% |

This strongly suggests:

- if the use case is preventive screening, a threshold near `0.35-0.40` is more appropriate than the default `0.50`
- if the use case is minimizing unnecessary referrals, the default threshold is more conservative but also riskier because of missed positives

In other words, the model is most useful as a risk-ranking or triage tool, not as a fixed-threshold diagnostic classifier.

### Feature Importance

The exported Random Forest ranks features approximately as follows:

| Feature | Importance |
|---|---:|
| `Glucose` | 0.324 |
| `BMI` | 0.155 |
| `Age` | 0.141 |
| `DiabetesPedigreeFunction` | 0.088 |
| `Pregnancies` | 0.088 |
| `Insulin` | 0.077 |
| `SkinThickness` | 0.070 |
| `BloodPressure` | 0.058 |

This aligns well with the notebook’s visual and correlation analysis, which is a good sign that the modelling story is internally coherent.

## 7. Reproducibility and Deployment Readiness

### What Is Good

- the repository contains the notebooks, dataset, exported model, scaler, and Flask app in one place
- the app in [app.py](/Users/harshkyada/Desktop/ToU%20CC/Diabetes%20EDA/flask_app/app.py) is simple enough for a reviewer to run locally
- the exported model is fast and lightweight

### What Still Needs Improvement

- the current project logic is notebook-first rather than pipeline-first
- preprocessing is not yet implemented in a fully leakage-safe, production-ready order
- the app performs very limited input validation
- the model artifact and scaler are stored separately instead of as one immutable pipeline
- the checked-in virtual environment appears to have been moved, so its `pip` wrapper points to an outdated absolute path

To support cleaner reproducibility, the repository now includes a root [requirements.txt](/Users/harshkyada/Desktop/ToU%20CC/requirements.txt) so a fresh environment can be created without relying on the broken checked-in venv.

## 8. How To Run The Project

### Create a fresh environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Open the analysis notebook

```bash
cd "Diabetes EDA"
jupyter notebook DSS_SDGs.ipynb
```

### Run the Flask app

Run the app from inside the Flask directory so the relative paths to the saved artifacts resolve correctly.

```bash
cd "Diabetes EDA/flask_app"
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## 9. Limitations

This project is strong as an educational and prototyping exercise, but it should not be oversold.

The main limitations are:

- small dataset size
- class imbalance
- placeholder-zero quality issues in several clinical variables
- narrow feature coverage with limited fairness variables
- preprocessing order that should be corrected before making stronger methodological claims
- lack of calibration analysis and external validation

## 10. Recommended Next Steps

The most valuable next improvements are:

1. replace the notebook preprocessing with a single `Pipeline` that handles imputation, scaling, and modelling correctly
2. expose risk probability and adjustable thresholding in the Flask app instead of only fixed labels
3. add calibration and ROC/PR analysis to support threshold selection
4. perform subgroup analysis wherever ethically relevant attributes are available
5. compare the tuned Random Forest directly against a calibrated Logistic Regression baseline
6. add tests and a small training script so the model can be rebuilt reproducibly

## Conclusion

The project already tells a meaningful and coherent story: diabetes risk in this dataset is multivariate, early warning is more realistic than hard diagnosis, and model value depends heavily on operating threshold and deployment context.

Its most important strengths are the public-health framing, the clear evidence that `Glucose`, `BMI`, and `Age` are the strongest drivers, and the fact that the modelling results are close enough to raise real selection trade-offs instead of producing a trivial winner.

Its most important weakness is not lack of ambition; it is that the current preprocessing and reproducibility choices are not yet as rigorous as the narrative deserves. Once the cleaning pipeline is formalized and thresholding is made explicit, the project becomes much closer to a strong submission-standard decision-support prototype.

## Project Structure

```text
ToU CC/
├── README.md
├── requirements.txt
├── report.ipynb
└── Diabetes EDA/
    ├── README.md
    ├── DSS_SDGs.ipynb
    ├── diabetes.csv
    ├── diabetes_model.pkl
    ├── scaler.pkl
    └── flask_app/
        ├── app.py
        └── templates/
            └── index.html
```

## Disclaimer

This project is for educational and research use. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
