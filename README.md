# Crop Feature Selection for Agricultural Decision Support

**Which soil measurement is most informative for crop classification—and what do we gain by measuring all four?**

This machine-learning project compares Logistic Regression models using individual soil measurements with a model using all four available measurements. It extends DataCamp's *Predictive Modeling for Agriculture* single-feature exercise with a reproducible train/test split, scaling pipelines, five-fold stratified cross-validation, and a held-out performance comparison.

## Key findings

| Evaluation | Weighted F1 |
| --- | ---: |
| Best single measurement: **Potassium (K)** | **0.202** |
| All four measurements: held-out test | **0.677** |
| All four measurements: 5-fold stratified CV (mean ± SD) | **0.658 ± 0.028** |
| Absolute test F1 gain from adding measurements | **0.475** |

**All-feature test accuracy:** 0.682.

Potassium was the strongest individual predictor in this experiment, but its performance was substantially lower than the model combining nitrogen, phosphorus, potassium, and pH. These are predictive associations within this dataset—not evidence that potassium alone determines crop suitability.

## Dataset

The educational dataset contains **2,200 records**, four numeric predictors (`N`, `P`, `K`, `ph`), and the multiclass label `crop` (**22 crop classes**).

**Source:** DataCamp, *Predictive Modeling for Agriculture* project (`soil_measures.csv`). The dataset is **not included** in this repository because redistribution permission has not been established. Obtain it through the original course/project access you are authorized to use.

## Method

1. Inspect dataset structure, missing values, and class counts.
2. Create a **stratified 80/20 train/test split** (`random_state=42`).
3. Train four independent single-feature baselines, each using `StandardScaler` and `LogisticRegression(max_iter=2000)`.
4. Train the four-feature model with the same preprocessing and classifier pipeline.
5. Evaluate the four-feature pipeline with **five-fold stratified cross-validation on the training set only**.
6. Fit the four-feature pipeline on the training set, evaluate on the held-out test set, and visualize the weighted F1 comparison.

**Primary metric:** weighted F1, a multiclass metric that aggregates class-level F1 scores by class support. Test accuracy is also reported.

## Run the notebook

### Google Colab

1. Open `Crop_Feature_Selection.ipynb` in Colab.
2. Upload your `soil_measures.csv` to the Colab session (normally `/content/`).
3. Select **Runtime → Run all**.

### Local Jupyter

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Create a `data/` folder and place `soil_measures.csv` inside it. Then open `Crop_Feature_Selection.ipynb` and run all cells. The `data/` directory is ignored by Git to avoid accidentally publishing the dataset.

## Interpretation and limitations

The absolute weighted F1 improvement of **0.475** measures the difference between the all-feature and best single-feature held-out test scores. It does **not** measure monetary savings or establish a causal effect.

The analysis is limited to one educational dataset and four soil variables. Actual crop suitability depends on climate, water availability, soil type, local growing conditions, and other factors. Logistic Regression is a baseline classifier; future work could examine nonlinear models and the real cost of collecting additional measurements.

## Project scope and attribution

- **Original exercise:** Identify the single most predictive soil measurement (DataCamp's *Predictive Modeling for Agriculture*).
- **Portfolio extension:** Compare the one-measurement constraint against all four features; add leakage-safe preprocessing, stratified cross-validation, held-out testing, a performance chart, and interpretation of the trade-off.

This is a learning and research-portfolio project, **not** a validated agricultural recommendation system.
