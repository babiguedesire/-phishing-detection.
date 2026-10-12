# Machine Learning-Based Phishing Website Detection and Its Robustness to Feature Forgery

Submission to the competition **"Artificial Intelligence in Information Security Tasks"**
(Format: Git repository). Author: BABIGUE DESIRE, Bachelor's student, 3rd year,
Moscow Technical University of Communications and Informatics.

Version: 1.0 - October 2026

## 1. Purpose of the project
Phishing remains one of the most common attack vectors, including in African countries
where mobile and online services grow quickly. This project builds and compares three
machine learning classifiers that decide whether a website is phishing or legitimate from
30 pre-extracted website features. It also evaluates a **safety aspect of the AI model**:
how quickly detection fails when an attacker forges the most important features.

## 2. Directory structure
```
.
├── data/phishing.csv        # dataset (UCI Phishing Websites, CC BY 4.0)
├── src/train.py             # data cleaning, training, evaluation, robustness test
├── results/                 # generated outputs (metrics, figures, model)
├── requirements.txt
├── LICENSE                  # MIT (code)
└── README.md
```

## 3. Requirements
Python 3.9+ and the packages in `requirements.txt` (numpy, pandas, scikit-learn,
matplotlib, joblib). No GPU needed; runs in under 2 minutes on a laptop.

## 4. Installation
```bash
git clone <this repository>
cd phishing-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Usage (reproducing the results)
```bash
python src/train.py
```
All random seeds are fixed (`SEED = 42`), so the results below are reproducible.
Outputs are written to `results/`.

## 6. Main modules
`src/train.py` contains:
- `load_data`: loads the CSV, **removes duplicate rows**, sets phishing = positive class;
- `evaluate`: accuracy, precision, recall, F1;
- `robustness_test`: simulates an attacker who forges the top-k features;
- `main`: stratified 80/20 split, 5-fold cross-validation, training of three models,
  figures and saved model.

## 7. Dataset, models and libraries
- **Dataset:** UCI "Phishing Websites" (Mohammad, Thabtah, McCluskey, 2012/2014),
  11,055 websites, 30 features, licence CC BY 4.0.
  https://archive.ics.uci.edu/dataset/327/phishing+websites
- **Data cleaning:** the file contains 5,206 duplicated rows; keeping them would put
  identical samples in both train and test sets and inflate scores. After removal:
  **5,849 unique websites** (51.6 % phishing).
- **Models:** Logistic Regression, Random Forest (300 trees), Gradient Boosting (scikit-learn).
- **Libraries:** numpy, pandas, scikit-learn, matplotlib, joblib (all open source).

## 8. Results (test set = 20 % of unique data)
| Model | Accuracy | Precision | Recall | F1 | CV F1 (5-fold) |
|---|---|---|---|---|---|
| Logistic Regression | 0.918 | 0.925 | 0.916 | 0.920 | 0.923 ± 0.003 |
| Random Forest | 0.942 | 0.957 | 0.929 | 0.943 | 0.949 ± 0.005 |
| Gradient Boosting | 0.944 | 0.956 | 0.935 | 0.946 | 0.946 ± 0.005 |

Most important features (Random Forest): `SSLfinal_State` (0.32), `URL_of_Anchor` (0.24),
`having_Sub_Domain`, `Prefix_Suffix`, `web_traffic`.

**Robustness test.** If an attacker sets the k most important features of phishing pages
to the "legitimate" value, phishing recall of the Random Forest drops from 0.929 (k=0)
to 0.801 (k=1), 0.121 (k=2), 0.017 (k=3) and 0.000 for k>=4 (`results/robustness.png`).

## 9. Risks, limitations and safe use
- **Old dataset (2015).** Phishing techniques evolve; the model may not detect modern attacks.
- **Pre-extracted features.** The model does not read raw URLs or pages; a real deployment
  needs a reliable feature extractor.
- **Adversarial fragility.** The robustness test is a worst-case simulation (the attacker knows
  the important features and can change them freely), but it shows that the detector relies
  heavily on a few features. Do not use this model as the only line of defence.
- **Errors.** About 7 % of phishing sites are missed (recall 0.93) and about 4 % of sites
  flagged as phishing are legitimate (precision 0.96).
- **Dual use.** The repository contains no attack tool: the forgery test only modifies
  numeric feature values and cannot be used to build real phishing pages.
- **Recommended use:** education, research and as one signal among others (blocklists,
  user training, human review). Human verification is advised before blocking a website.
- **Personal data:** the dataset contains only numeric features, no personal data.

## 10. License
Code: MIT License (see `LICENSE`). Dataset: CC BY 4.0, attribution to the original authors
(see section 7).
