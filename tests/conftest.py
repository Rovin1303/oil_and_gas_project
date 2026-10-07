import os
import sys
from unittest.mock import MagicMock
import pytest
from pyspark.sql import SparkSession

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.modules["databricks.feature_engineering"] = MagicMock()


# Import after mocking Databricks dependency
from src.model_train.model_training import MLModelTraining


@pytest.fixture(scope="session")
def spark():
    """Provide a SparkSession for tests."""

    spark = SparkSession.builder.getOrCreate()

    yield spark
    spark.stop()

@pytest.fixture
def model_training(tmp_path):
    """Provide MLModelTraining instance for tests."""

    config = {
        "ml": {
            "encoding": {
                "categorical_columns": [
                    "product_name",
                    "destination_city"
                ],
                "numeric_columns": [
                    "ordered_quantity",
                    "unit_price"
                ]
            },
            "artifacts": {
                "directory": str(tmp_path),
                "encoder": "encoder.pkl"
            }
        }
    }

    return MLModelTraining(
        spark=None,
        feature_client=None,
        feature_table="dummy_feature_table",
        output_table="dummy_output_table",
        config=config
    )