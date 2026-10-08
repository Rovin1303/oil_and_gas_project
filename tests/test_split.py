import pandas as pd
import pytest
import sys

def test_split_method(spark,model_training):
    data = [
        (1, "2026-01-01", 100),
        (2, "2026-01-02", 110),
        (3, "2026-01-03", 120),
        (4, "2026-01-04", 130),
        (5, "2026-01-05", 140),
        (6, "2026-01-06", 150),
        (7, "2026-01-07", 160),
        (8, "2026-01-08", 170),
        (9, "2026-01-09", 180),
        (10, "2026-01-10", 190)
    ]
    df = spark.createDataFrame(data,["transaction_id","transaction_date","total_demand"])

    train_df, validate_df, test_df = model_training.split_data(df)

    assert len(train_df) == 6
    assert len(validate_df) == 2
    assert len(test_df) == 2
    assert train_df["transaction_id"].tolist() == [1,2,3,4,5,6]
    assert validate_df["transaction_id"].tolist() == [7,8]
    assert test_df["transaction_id"].tolist() == [9,10]

def test_encode_categorical_data(encoding_model_training):

    X_train = pd.DataFrame({
        "product_name": ["Product_A","Product_B","Product_C","Product_A"],
        "destination_city": ["Mumbai","Delhi","Chennai","Mumbai"],
        "ordered_quantity": [100,200,150,300],
        "unit_price": [10.0,20.0,15.0,30.0],
    })

    X_train_final, encoder = encoding_model_training.encode_categorical_data(X_train)

    assert len(X_train_final) == len(X_train)
    assert "product_name_Product_A" in X_train_final.columns
    assert "product_name_Product_B" in X_train_final.columns
    assert "product_name_Product_C" in X_train_final.columns
    assert "destination_city_Chennai" in X_train_final.columns
    assert "destination_city_Delhi" in X_train_final.columns
    assert "destination_city_Mumbai" in X_train_final.columns

def test_invalid_column(encoding_model_training):

    X_train = pd.DataFrame({
        "product_name": ["Product_A", "Product_B"],
        "destination_city": ["Mumbai", "Delhi"],
        "ordered_quantity": [100, 200],
        "unit_price": [10.0, 20.0]
    })

    encoding_model_training.config["ml"]["encoding"]["categorical_columns"] = ["invalid_column"]

    with pytest.raises(KeyError):
        encoding_model_training.encode_categorical_data(X_train)
