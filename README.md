# Sprint 11 - Project: Oil Well Selection (ML) - OilyGiant

This individual project tests machine learning skills applied to a realistic business case.

---

## Project Description

**OilyGiant**, an oil extraction company, needs to identify the best locations to open **200 new oil wells** in one of three available regions, maximizing profit and minimizing risk.

---

## Objectives

- Predict the volume of reserves in new wells.
- Select the 200 wells with the highest estimated volume.
- Choose the region with the highest total profit.
- Evaluate risks using bootstrapping.

---

## Datasets

The data is contained in three CSV files:

- `geo_data_0.csv`
- `geo_data_1.csv`
- `geo_data_2.csv`

Each file contains:

- `id`: Unique well identifier.
- `f0`, `f1`, `f2`: Point characteristics.
- `product`: Reserve volume (in thousands of barrels).

---

## Project Instructions

### 1. Data Preparation
- Initial loading and exploration.
- Division into training set (75%) and validation set (25%).

### 2. Model Training
- Linear regression model by region.
- Prediction of reserve volume.
- Evaluation with RMSE and average volume.

### 3. Well Selection
- Selection of the 200 wells with the highest estimated volume.
- Calculation of potential profit by region.

### 4. Risk Assessment
- Bootstrapping with 1,000 samples.
- Calculation of:
  - Average profit.
  - 95% confidence interval.
  - Risk of losses (<2.5%).

---

## Business Conditions

- Budget: **$100 million**.
- Revenue per unit: **$4,500**.
- Minimum threshold per well: **111.1 units**.
- Only **linear regression** is allowed.

---

## Checklist

- Proper data preparation.
- Compliance with instructions and conditions.
- Correct application of bootstrapping.
- Clear justification of the selected region.
- Clean, modular code without duplications.

---

## Tools

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Pyplot
- Sklearn

