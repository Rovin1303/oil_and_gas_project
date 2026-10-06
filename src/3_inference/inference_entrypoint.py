import yaml
from databricks.feature_engineering import FeatureEngineeringClient
from model_selection import ModelSelection
from model_predictions import ChampionChallenger


def main():
    """Execute the full inference pipeline: model selection and champion-challenger evaluation."""
    with open("../../config/config.yml", "r") as file:
        config = yaml.safe_load(file)

    fe = FeatureEngineeringClient()
    feature_table = config["tables"]["input_feature"]
    output_table = config["tables"]["output_feature"]
    prediction_table = config["tables"]["prediction"]
    model_selection = ModelSelection(
        spark=spark,
        feature_client=fe,
        feature_table=feature_table,
        output_table=output_table,
        config=config
    )
    model_selection.run()

    champion_challenger = ChampionChallenger(
        spark=spark,
        feature_client=fe,
        feature_table=feature_table,
        output_table=output_table,
        prediction_table=prediction_table,
        config=config
    )
    champion_challenger.run()

if __name__ == "__main__":
    main()