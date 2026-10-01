# Fire Radiative Power Prediction with Machine Learning and Explainable AI

This project investigates the prediction of **Fire Radiative Power (FRP)** from satellite-based active fire observations using multiple machine learning models.

The goal is to compare traditional ensemble methods and deep learning approaches for FRP regression while also using **SHAP (SHapley Additive exPlanations)** to interpret how individual features influence model predictions.

## Overview

Fire Radiative Power (FRP) is a satellite-derived measure related to the intensity of actively burning fires.

In this project, I trained and evaluated three regression models:

- Random Forest Regressor
- XGBoost Regressor
- TabNet Regressor

The models were trained using satellite fire-detection features such as geographic location, thermal measurements, confidence level, and day/night information.

Hyperparameter tuning and cross-validation were used to improve model performance, while SHAP analysis was used to examine feature importance and individual predictions.

## Dataset

The main dataset used in the research notebook is:

`fire_nrt_J2V-C2_538155(data1).csv`

The dataset contains satellite-based active fire observations obtained from **NASA FIRMS (Fire Information for Resource Management System)**.

The notebook contains approximately:

- 103,070 observations
- 14 original columns

After preprocessing, six variables were selected as model inputs.

### Features

The following features were used:

- `latitude` — geographic latitude of the detected fire
- `longitude` — geographic longitude of the detected fire
- `brightness` — satellite-measured brightness temperature
- `bright_t31` — brightness temperature from the T31 thermal channel
- `confidence` — confidence level associated with the fire detection
- `daynight` — whether the observation was acquired during the day or night

### Target Variable

The prediction target is:

- `frp` — Fire Radiative Power

## Data Preprocessing

Several preprocessing steps were applied before model training.

Categorical variables were converted into numerical representations.

For confidence:

```python
confidence = {
    "l": 1,
    "n": 2,
    "h": 3
}
```

For day/night:

```python
daynight = {
    "D": 1,
    "N": 2
}
```

Rows containing missing values in the target or required encoded features were removed.

The dataset was then divided into:

- 80% training data
- 20% testing data

This resulted in approximately:

- 82,456 training observations
- 20,614 testing observations

Feature scaling was performed using `MinMaxScaler`.

Because FRP is highly skewed, the target variable was transformed using a **Yeo-Johnson Power Transformation**.

The feature scaler and target transformation were fit using the training data in order to reduce information leakage from the test set.

## Models

### Random Forest

A `RandomForestRegressor` was optimized using `RandomizedSearchCV`.

The hyperparameter search evaluated multiple parameter combinations using 5-fold cross-validation.

Best parameters:

```python
{
    "n_estimators": 300,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 30,
    "bootstrap": True
}
```

### TabNet

TabNet was used as a deep-learning model designed specifically for tabular data.

Several network architectures were evaluated using cross-validation.

Best configuration:

```python
{
    "n_d": 16,
    "n_a": 16,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 0.0001
}
```

GPU acceleration was used when available.

### XGBoost

An `XGBRegressor` was also optimized using randomized hyperparameter search and cross-validation.

Best parameters:

```python
{
    "subsample": 0.85,
    "n_estimators": 400,
    "min_child_weight": 3,
    "max_depth": 8,
    "learning_rate": 0.1,
    "colsample_bytree": 1.0
}
```

## Results

The three models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.3972 | 0.2774 | 0.7216 |
| TabNet | 0.4390 | 0.3310 | 0.6679 |
| XGBoost | **0.3932** | **0.2728** | **0.7262** |

Among the tested models, **XGBoost achieved the highest R² score and the lowest MAE and MSE**.

Random Forest performed very similarly, while TabNet achieved somewhat lower predictive performance on this dataset.

> **Note:** These metrics were calculated on the transformed FRP target rather than directly in the original FRP physical units.

## Explainable AI

To better understand model behavior, the project uses **SHAP** to analyze feature contributions.

SHAP explanations were generated for:

- Random Forest
- TabNet
- XGBoost

For the tree-based models, `TreeExplainer` was used.

For TabNet, `KernelExplainer` was used.

The explainability analysis includes:

- Global feature importance
- SHAP summary plots
- Dependence plots
- Individual waterfall plots
- Decision plots

This allows the project to examine not only predictive accuracy, but also how different variables influence individual FRP predictions.

## Project Structure

```text
.
├── FRP_Prediction_Refined.ipynb
├── fire_nrt_J2V-C2_538155(data1).csv
├── fire_archive_M-C61_604376.csv
├── interactive_dashboard.py
└── README.md
```

### Main Files

`FRP_Prediction_Refined.ipynb`

Contains the primary data preprocessing, model training, hyperparameter tuning, evaluation, and SHAP explainability analysis.

`interactive_dashboard.py`

Contains the interactive visualization/dashboard component of the project.

`fire_nrt_J2V-C2_538155(data1).csv`

Main dataset used by the research notebook.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch
- PyTorch TabNet
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch pytorch-tabnet shap
```

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd <repository-name>
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
FRP_Prediction_Refined.ipynb
```

Make sure the dataset:

```text
fire_nrt_J2V-C2_538155(data1).csv
```

is located in the same project directory or update the dataset path inside the notebook.

## Future Improvements

Possible extensions of the project include:

- Evaluating predictions after transforming FRP values back into their original units
- Testing additional gradient boosting and neural network architectures
- Incorporating acquisition date and time information
- Adding meteorological variables such as temperature, humidity, wind, and precipitation
- Testing geographic generalization using spatial train/test splits
- Testing temporal generalization using chronological train/test splits
- Investigating performance across different regions and fire conditions
- Improving model calibration and uncertainty estimation
- Deploying the trained model through an interactive web application
- Comparing satellite-derived FRP predictions across multiple satellite products

## Limitations

Several limitations should be considered when interpreting the results.

The model performance depends on the available satellite-derived features and may not generalize equally well across different geographic regions, fire types, seasons, or satellite sensors.

The current train/test approach may also allow geographically or temporally similar observations to appear in both sets. Future experiments could use spatial or chronological splitting strategies to provide a stronger test of model generalization.

Additionally, the reported evaluation metrics are calculated using the transformed FRP target rather than FRP values converted back into their original physical scale.

## Data Source

The satellite fire data used in this project was obtained from:

**NASA Fire Information for Resource Management System (FIRMS)**

https://firms.modaps.eosdis.nasa.gov/

## Author

**Nijat Taghizada**