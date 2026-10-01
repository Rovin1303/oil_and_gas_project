import logging
import pickle
import mlflow
import pandas as pd
from mlflow import MlflowClient
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureLookup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

logger = logging.getLogger("DemandPrediction")
logger.setLevel(logging.INFO)

class ChampionChallenger:

    def __init__(self,spark,feature_client,feature_table,output_table,prediction_table,config):

        self.spark = spark
        self.feature_client = feature_client
        self.feature_table = feature_table
        self.output_table = output_table
        self.prediction_table = prediction_table
        self.config = config
        self.model_name = self.config["ml"]["model_name"]["registered_name"]
        registry_uri = self.config["ml"]["registry"]["uri"] 
        mlflow.set_registry_uri(registry_uri)

        self.client = MlflowClient()

        logger.info("ChampionChallenger initialized")

    def get_test_data(self):

        logger.info("Creating test dataset")
        target_column = self.config["ml"]["target"]["column"]

        training_set = self.feature_client.create_training_set(
            df=self.spark.table(self.output_table),
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table,
                    lookup_key="primary_key"
                )
            ],
            label=target_column,
            exclude_columns=["primary_key"]
        )
        df = training_set.load_df()
        df = df.orderBy("transaction_date")
        df = df.drop("transaction_date")
        df = df.toPandas()
        train_validate_df,test_df = train_test_split(df,test_size=0.20,shuffle=False)
        train_df,validate_df = train_test_split(train_validate_df,test_size=0.20,shuffle=False)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        return X_test,y_test

    def load_encoder(self,run_id):

        logger.info("Loading encoder")

        encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id,
            artifact_path=self.config["ml"]["artifacts"]["artifact_path"])

        with open(encoder_path,"rb") as file:
            encoder = pickle.load(file)

        return encoder
    
    def encode_test_data(self,X_test,encoder):

        category_columns = (self.config["ml"]["encoding"]["categorical_columns"])
        numeric_columns = (self.config["ml"]["encoding"]["numeric_columns"])

        encoded_data = encoder.transform(X_test[category_columns])

        encoded_columns = (
            encoder.get_feature_names_out(
                category_columns
            )
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_columns,
            index=X_test.index
        )

        X_test_final = pd.concat(
            [
                X_test[numeric_columns],
                encoded_df
            ],
            axis=1
        )

        return X_test_final
    
    def evaluate_model(self,model_alias,X_test,y_test):

        logger.info("Evaluating model using alias: %s",model_alias)

        model_details = (self.client.get_model_version_by_alias(self.model_name,model_alias))

        model_uri = (f"models:/{self.model_name}@{model_alias}")
        model = mlflow.sklearn.load_model(model_uri)

        encoder = self.load_encoder(model_details.run_id)

        X_test_final = self.encode_test_data(X_test,encoder)

        predictions = model.predict(X_test_final)

        test_r2 = r2_score(y_test,predictions)

        logger.info(
            "%s Test R²: %.4f",
            model_alias,
            test_r2
        )
        return {
            "version": model_details.version,
            "run_id": model_details.run_id,
            "test_r2": test_r2,
            "predictions": predictions
        }
    
    def compare_models(self,X_test,y_test):

        logger.info("Starting Champion-Challenger comparison")
        challenger_alias = self.config["ml"]["registry"]["challenger_alias"]
        champion_alias = self.config["ml"]["registry"]["champion_alias"]
        challenger = self.evaluate_model(challenger_alias,X_test,y_test)

        challenger_r2 = challenger["test_r2"]
        try:

            champion_model = (
                self.client.get_model_version_by_alias(
                    self.model_name,
                    champion_alias
                )
            )

            logger.info(
                "Champion model exists. Version: %s",
                champion_model.version
            )

            champion = self.evaluate_model(
                champion_alias,
                X_test,
                y_test
            )

            champion_r2 = champion["test_r2"]

            logger.info(
                "Champion R²: %.4f | Challenger R²: %.4f",
                champion_r2,
                challenger_r2
            )

            metric_r2_passed = (challenger_r2 > champion_r2)
        except Exception:

            logger.info(
                "No Champion found. "
                "Accepting Challenger as first Champion."
            )

            champion_r2 = None

            metric_r2_passed = True

        self.client.set_model_version_tag(
            name=self.model_name,
            version=challenger["version"],
            key="test_r2",
            value=str(challenger_r2)
        )

        self.client.set_model_version_tag(
            name=self.model_name,
            version=challenger["version"],
            key="metric_r2_passed",
            value=str(metric_r2_passed)
        )

        if champion_r2 is not None:

            self.client.set_model_version_tag(
                name=self.model_name,
                version=challenger["version"],
                key="champion_r2",
                value=str(champion_r2)
            )

        if metric_r2_passed:

            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=champion_alias,
                version=challenger["version"]
            )

            logger.info(
                "Challenger version %s promoted to Champion",
                challenger["version"]
            )

        else:

            logger.info(
                "No model promotion. "
                "Champion remains unchanged."
            )

        return {
            "challenger_version": challenger["version"],
            "challenger_r2": challenger_r2,
            "champion_r2": champion_r2,
            "metric_r2_passed": metric_r2_passed
        }
    def predict_with_champion_model(self, X_test):
        logger.info("Loading Champion model for final prediction")
        champion_alias = self.config["ml"]["registry"]["champion_alias"]
        champion_details = (self.client.get_model_version_by_alias(self.model_name,champion_alias))
        model_uri = (f"models:/{self.model_name}@{champion_alias}")

        logger.info("Champion model URI: %s",model_uri)

        logger.info(
            "Champion version: %s",
            champion_details.version
        )

        logger.info(
            "Champion run ID: %s",
            champion_details.run_id
        )

        model = mlflow.sklearn.load_model(model_uri)

        encoder = self.load_encoder(champion_details.run_id)

        X_test_final = self.encode_test_data(X_test,encoder)

        predictions = model.predict(X_test_final)

        logger.info("Predictions generated using Champion version %s",champion_details.version)

        return predictions
    
    def write_to_table(self,X_test,y_test,predictions):
        logger.info(
            "Creating Champion prediction dataframe"
        )

        prediction_df = X_test.copy()

        prediction_df["actual_demand"] = y_test.values

        prediction_df["predicted_demand"] = predictions

        prediction_df["timestamp"] = (
            pd.Timestamp.now()
        )

        prediction_spark_df = (
            self.spark.createDataFrame(
                prediction_df
            )
        )

        prediction_spark_df.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(
                self.prediction_table
            )

        logger.info("Champion predictions saved successfully")
    def run(self):

        logger.info(
            "Champion-Challenger pipeline started"
        )
        X_test, y_test = self.get_test_data()
        result = self.compare_models(X_test,y_test)
        predictions = self.predict_with_champion_model(X_test)
        self.write_to_table(X_test,y_test,predictions)
        logger.info("Champion-Challenger pipeline completed")
        return result