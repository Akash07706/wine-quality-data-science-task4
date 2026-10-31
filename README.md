# Wine Quality Analysis – Data Science Tool Mastery

## Overview
A practical data-science workflow using red and white wine quality datasets.

## Dataset
- Red: 1599 rows, 12 columns before cleaning.
- White: 4898 rows, 12 columns before cleaning.
- CSV source files are semicolon-delimited and include physicochemical measurements and a quality score.

## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn and Jupyter Notebook.

## Workflow
1. Load and inspect the two CSV datasets.
2. Check missing cells and duplicate rows.
3. Remove duplicates in analysis copies while preserving raw source files.
4. Explore quality scores and chemical measurements.
5. Create visualizations.
6. Train Random Forest regression models and evaluate using MAE, RMSE and R².

## Data checks
- Red: 0 missing cells; 240 duplicate rows; 1359 rows after duplicate removal.
- White: 0 missing cells; 937 duplicate rows; 3961 rows after duplicate removal.

## Model results
80/20 train-test split, random_state=42. These are educational baseline results.

| Dataset | Clean rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Red | 1359 | 0.467 | 0.617 | 0.462 |
| White | 3961 | 0.564 | 0.739 | 0.335 |

## Run
```bash
pip install -r requirements.txt
python main.py
```

Dataset source: UCI Machine Learning Repository, Wine Quality dataset.
