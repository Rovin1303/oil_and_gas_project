# Demand Prediction ML Project

An end-to-end machine learning pipeline for predicting product demand across a supply chain network. The project follows a **medallion architecture** (Bronze → Silver → Gold) for data engineering, uses **Databricks Feature Store** for feature management, and implements a **Champion/Challenger** model lifecycle with MLflow.

---

## Project Structure

```
ML_Project/
├── bronze/
│   └── bronze_dim                 # Raw CSV ingestion → Bronze Delta table
├── silver/
│   └── silver_dim                 # Clean & deduplicate Bronze → Silver Delta table
├── gold/
│   ├── gold_fact                   # Aggregate Silver → Gold fact table
│   └── gold_features               # Feature engineering (lag, rolling avg) → Gold features table
├── feature_store/
│   └── features                    # Register features & labels in UC Feature Store
├── machine_learning/
│   ├── 1_model_training            # Train DecisionTree model, log to MLflow
│   ├── 2_model_selection           # Register best model to Unity Catalog
│   ├── 3_testing_model             # Test model on held-out data, assign Challenger alias
│   └── 4_model_predictions          # Champion vs Challenger comparison, write predictions
├── pipeline/
│   ├── 1_data_engineering          # Orchestrate: bronze → silver → gold → feature store
│   ├── 2_model_training            # Orchestrate: model training
│   ├── 3_predictions               # Orchestrate: selection + testing + predictions
│   └── models/
│       └── onehot_encoder.pkl      # Saved OneHotEncoder artifact
├── models/
│   └── onehot_encoder.pkl          # Saved OneHotEncoder artifact
└── README.md
```

---

## Architecture Overview

### Data Pipeline (Medallion Architecture)

| Layer | Source | Target Table | Description |
|-------|--------|-------------|-------------|
| **Bronze** | `/Volumes/oag/bronze/raw/supply_chain_dataset.csv` | `oag.bronze.bronze_table` | Raw CSV ingested with explicit schema, 74,900 rows |
| **Silver** | `oag.bronze.bronze_table` | `oag.silver.silver_table` | Drop metadata columns, remove duplicates by `transaction_id` |
| **Gold (Fact)** | `oag.silver.silver_table` | `oag.gold.gold_table` | Aggregate by `(transaction_date, product_name, destination_city)` — sum demand, avg price, sum inventory, count transactions |
| **Gold (Features)** | `oag.gold.gold_table` | `oag.gold.gold_features` | Lag features (`demand_1`, `demand_7`), 7-day rolling average, month, day of week |
| **Feature Store** | `oag.gold.gold_features` | `oag.gold.demand_input_features` | UC Feature Store table with composite primary key (`yyyyMMdd-product-city`) |
| **Feature Store (Labels)** | `oag.gold.gold_features` | `oag.gold.demand_output` | Label table with `primary_key` and `total_demand` |

### ML Pipeline

| Step | Notebook | Description |
|------|----------|-------------|
| **1. Training** | `1_model_training` | Train `DecisionTreeRegressor` (max_depth=20) on 64% train split. OneHot encode categorical columns. Log model + encoder to MLflow. |
| **2. Selection** | `2_model_selection` | Search MLflow for best run by `validation_r2`. Register model to UC as `oag.ml.demand_prediction_model`. |
| **3. Testing** | `3_testing_model` | Load latest registered model, evaluate R² on 20% test split. If R² > 0.90, assign `Challenger` alias. |
| **4. Predictions** | `4_model_predictions` | Compare `Challenger` vs `Champion` R² on test data. Promote Challenger if it wins. Predict with Champion and write results to `oag.predictions.prediction_table`. |

### Job Orchestration

A Databricks Job orchestrates the full pipeline as 3 sequential tasks:

| Task | Notebook | Depends On |
|------|----------|------------|
| `data_engineering` | `pipeline/1_data_engineering` | — |
| `2_model_training` | `pipeline/2_model_training` | `data_engineering` |
| `3_model_prediction` | `pipeline/3_predictions` | `2_model_training` |

---

## Features & Target

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `product_name` | categorical | Product identifier (OneHot encoded) |
| `destination_city` | categorical | Delivery destination (OneHot encoded) |
| `avg_unit_price` | numeric | Average unit price (USD) |
| `total_inventory` | numeric | Total available inventory |
| `transaction_count` | numeric | Number of transactions for that day/product/city |
| `demand_1` | numeric | Demand lagged by 1 day |
| `demand_7` | numeric | Demand lagged by 7 days |
| `rolling_7_day_avg` | numeric | 7-day rolling average of demand |
| `month` | numeric | Calendar month (1–12) |
| `day_of_week` | numeric | Day of week (1–7) |

### Target

- `total_demand` — aggregate demand quantity per transaction date, product, and destination city

### Data Splits

Data is split **chronologically** (`shuffle=False`), ordered by `transaction_date`:

| Split | Rows | Percentage | Purpose |
|-------|------|-----------|---------|
| Train | 47,936 | 64% | Model training |
| Validation | 11,984 | 16% | Hyperparameter evaluation |
| Test | 14,980 | 20% | Final model evaluation & predictions |

---

## Unity Catalog Tables

| Table | Layer | Description |
|-------|-------|-------------|
| `oag.bronze.bronze_table` | Bronze | Raw supply chain data |
| `oag.silver.silver_table` | Silver | Cleaned & deduplicated data |
| `oag.gold.gold_table` | Gold | Aggregated demand facts |
| `oag.gold.gold_features` | Gold | Feature-engineered data with lag/rolling features |
| `oag.gold.demand_input_features` | Feature Store | UC Feature Store feature table (PK: `primary_key`) |
| `oag.gold.demand_output` | Feature Store | Label table (`primary_key`, `total_demand`) |
| `oag.predictions.prediction_table` | Predictions | Model predictions on test data only |

### MLflow Registered Model

- **Name**: `oag.ml.demand_prediction_model`
- **Registry**: Unity Catalog (`databricks-uc`)
- **Aliases**: `Champion` (production model), `Challenger` (candidate model)
- **Algorithm**: `DecisionTreeRegressor` (criterion=squared_error, max_depth=20)

### Prediction Table Schema

| Column | Type |
|--------|------|
| `product_name` | string |
| `destination_city` | string |
| `avg_unit_price` | double |
| `total_inventory` | double |
| `transaction_count` | long |
| `demand_1` | double |
| `demand_7` | double |
| `rolling_7_day_avg` | double |
| `month` | integer |
| `day_of_week` | integer |
| `actual_demand` | double |
| `predicted_demand` | double |
| `timestamp` | timestamp |

---

## Champion / Challenger Workflow

The `4_model_predictions` notebook implements an automated model lifecycle:

1. **Load models** — Fetch `Champion` and `Challenger` models by alias from MLflow
2. **Evaluate** — Compute R² score for both models on the 20% test split
3. **Promote** — If the Challenger's R² exceeds the Champion's, reassign the `Champion` alias to the Challenger version
4. **Predict** — Run inference with the (potentially updated) Champion model on test data
5. **Write** — Overwrite `oag.predictions.prediction_table` with test data predictions (features + `actual_demand` + `predicted_demand` + `timestamp`)

---

## Technologies

- **Databricks** — Serverless compute, notebooks, Jobs
- **Unity Catalog** — Data governance, table & model registry
- **Delta Lake** — Storage format for all tables
- **Databricks Feature Store** — Feature management via `FeatureEngineeringClient`
- **MLflow** — Experiment tracking, model registry, artifact storage
- **scikit-learn** — `DecisionTreeRegressor`, `OneHotEncoder`, `train_test_split`
- **PySpark** — Data processing & transformations

---

## How to Run

### Option 1: Run the orchestrated Job

Trigger the Databricks Job (ID: `1051307128364436`) which runs the full pipeline end-to-end:

1. **Data Engineering** — Bronze → Silver → Gold → Feature Store
2. **Model Training** — Train & log DecisionTree model to MLflow
3. **Predictions** — Model selection, testing, Champion/Challenger comparison, write predictions

### Option 2: Run notebooks individually

Execute notebooks in the following order:

1. `pipeline/1_data_engineering` (or run `bronze_dim` → `silver_dim` → `gold_fact` → `gold_features` → `features`)
2. `pipeline/2_model_training` (or `machine_learning/1_model_training`)
3. `pipeline/3_predictions` (or `machine_learning/2_model_selection` → `3_testing_model` → `4_model_predictions`)

---

## Key Design Decisions

- **Chronological splits** — Data is split by `transaction_date` without shuffling, ensuring the test set represents the most recent real-world data
- **Test data only in predictions** — The prediction table contains only the 20% held-out test split, not the full dataset
- **Overwrite mode** — All tables use `mode("overwrite")`, so each run replaces the previous data
- **Feature Store integration** — Features are managed through Databricks Feature Store with a composite primary key for reproducible training sets
- **Model lifecycle** — The Champion/Challenger system enables continuous model improvement with automatic promotion based on test performance