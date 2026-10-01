import yaml

from databricks.feature_engineering import FeatureEngineeringClient
from bronze import BronzeIngestion
from silver import SilverTransformation
from gold import GoldTransformation
from gold_feature import Features
from feature_store import FeatureStore


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def bronze(config):
    data_path = config["paths"]["data_path"]
    catalog_name = config["catalog"]["name"]
    bronze_table = config["tables"]["bronze"]

    bronze_ingestion = BronzeIngestion(
        spark=spark,
        data_path=data_path,
        catalog_name=catalog_name,
        table_name=bronze_table
    )

    bronze_ingestion.run()


def silver(config):
    catalog_name = config["catalog"]["name"]
    bronze_table = config["tables"]["bronze"]
    silver_table = config["tables"]["silver"]

    silver = SilverTransformation(
        spark=spark,
        catalog_name=catalog_name,
        bronze_table=bronze_table,
        silver_table=silver_table
    )

    silver.run()


def gold(config):
    catalog_name = config["catalog"]["name"]
    silver_table = config["tables"]["silver"]
    gold_table = config["tables"]["gold"]

    gold = GoldTransformation(
        spark=spark,
        catalog_name=catalog_name,
        silver_table=silver_table,
        gold_table=gold_table
    )

    gold.run()


def gold_feature(config):
    catalog_name = config["catalog"]["name"]
    gold_table = config["tables"]["gold"]
    gold_feature_table = config["tables"]["gold_feature"]

    features = Features(
        spark=spark,
        catalog_name=catalog_name,
        gold_table=gold_table,
        gold_feature=gold_feature_table
    )
    features.run()

def feature_store(config):
    gold_feature_table = config["tables"]["gold_feature"]
    input_features = config["tables"]["input_feature"]
    output_features = config["tables"]["output_feature"]

    feature_client = FeatureEngineeringClient()

    features_obj = FeatureStore(
        spark=spark,
        feature_client=feature_client,
        gold_table=gold_feature_table,
        feature_table=input_features,
        label_table=output_features
    )

    features_obj.run()


def main():

    config_path = "../../config/config.yml"

    config = load_config(config_path)

    bronze(config)
    silver(config)
    gold(config)
    gold_feature(config)
    feature_store(config)


if __name__ == "__main__":
    main()