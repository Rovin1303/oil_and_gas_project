#!/usr/bin/env python
import yaml
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk.runtime import spark
from model_training import MLModelTraining


def main():
    """Execute the full ML model training pipeline using configuration from config.yml."""
    with open("../../config/config.yml", "r") as file:
        config = yaml.safe_load(file)

    feature_client = FeatureEngineeringClient()
    feature_table = config["tables"]["input_feature"]
    output_table = config["tables"]["output_feature"]
    ml_training = MLModelTraining(
        spark=spark,
        feature_client=feature_client,
        feature_table=feature_table,
        output_table=output_table,
        config=config
    )
    ml_training.run()


if __name__ == "__main__":
    main()