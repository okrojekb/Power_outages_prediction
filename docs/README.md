# Dynamic Rhythms

Welcome to the **Dynamic Rhythms** project, submitted as part of the competition. This repository contains all the necessary notebooks, scripts, and model files related to our work.

## Project Overview

Our goal was to analyze and model rhythmic patterns based on dynamic data streams. The project followed a structured workflow, including exploratory data analysis (EDA), feature engineering, model development, and interpretability analysis.

Due to memory limitations, the complete datasets are not included directly in this repository. However, all datasets used throughout the project can be accessed here:  
🔗 [Download Data](https://1drv.ms/u/c/26e35e72c256bf0a/EYj-bgeSRgxAohqgwVkLEGwBeNcyrU6AfPk4paxdbPUbQg?e=bGIxCX)

---

## Repository Structure

### `/notebooks/`
This folder contains a series of Jupyter Notebooks documenting each stage of the project:

- **01_data_preprocessing.ipynb**  
  - Exploratory Data Analysis (EDA) and visualization.
  - Initial analysis of the datasets we contributed.
  - Early conclusions to guide our modeling strategy.

- **02_feature_engineering.ipynb**  
  - Data transformation and preparation for modeling.
  - Creation of the final `modelingDataFrame.csv` used as the primary modeling dataset.
  - (Download link for modelingDataFrame.csv provided upper).

- **03_model_development.ipynb**  
  - Detailed training processes.
  - Various modeling approaches and experiments.
  - Comparison and evaluation of different models.

- **04_interpretability.ipynb**  
  - Summary of results.
  - Analysis of feature importance.
  - Interpretation of model outputs and insights.

Additionally, the folder includes Python scripts with reusable functions:

- `data_preprocessing.py`
- `feature_engineering.py`
- `interpretability.py`

---

### `/models/`
Pre-trained model files:

- **xgb_model.pkl** – Best-performing XGBoost model (after grid search/random search), not used in the interpretability phase.
- **tabnet_model.zip** – Initial TabNet model trained on the complete dataset (even though better model was trained it was not much better so weused for interpretability this one).
- **model_TabNet_florida.zip** – TabNet model trained on Florida-specific data (showed significant improvements in localized modeling).
- **model_MLP.pkl** – Best-trained Multi-Layer Perceptron (MLP) model.
- **suffolk_sarimax_model.pkl** – Best SARIMAX model for Suffolk County (no peak data).
- **waynesboro_sarimax_model.pkl** – Best SARIMAX model for Waynesboro County (with peak data).

---

### `/docs/`
Supporting documents:

- **README.md** – This document.
- **requirements.txt** – List of required Python packages to reproduce our results.


## Additional Notes

- For full functionality, please download the required datasets from the provided link.
- Pre-trained models are available in `/models/` and can be loaded directly for evaluation or inference.
- Not all models developed are used in the final interpretability stage; selections were based on model performance and interpretability trade-offs.

