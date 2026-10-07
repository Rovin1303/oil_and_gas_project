#!/usr/bin/env python
import logging
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from databricks.feature_engineering import FeatureLookup
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ModelSelection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

class ModelSelection:

    def __init__(
        self,
        spark,
        feature_client,
        feature_table,
        output_table,
        config
    ):
        """Initialize the ModelSelection instance.

        Args:
            spark: SparkSession object
            feature_client: FeatureEngineeringClient instance
            feature_table: Name of the feature table in Feature Store
            output_table: Name of the label/output table
            config: Dictionary containing ML configuration
        """
        self.spark = spark
        self.feature_client = feature_client
        self.feature_table = feature_table
        self.output_table = output_table
        self.config = config

        self.model_version = None
        self.run_id = None

        logger.info("Model Selection initialized")

    def get_best_run(self):
        """Search MLflow for the best completed run based on validation R2.

        Returns:
            Pandas Series containing the best run's metadata and metrics.
        """
        logger.info(
            "Searching for best model based on validation R2"
        )

        experiment = mlflow.get_experiment_by_name(
            self.config["ml"]["experiment_name"]
        )

        if experiment is None:
            raise ValueError(
                f"Experiment not found: {self.config['ml']['experiment_name']}"
            )
        selection_metric = self.config["ml"]["mlflow"]["selection_metric"]
        run_name = self.config["ml"]["mlflow"]["run_name"]
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "status = 'FINISHED' "
                f"and run_name = '{run_name}'"
            ),
            order_by=[f"metrics.{selection_metric} DESC"],
            max_results=1
        )

        if runs.empty:
            raise ValueError(
                "No completed runs with validation R2 found"
            )

        best_run = runs.iloc[0]

        logger.info(
            "Best run ID: %s",
            best_run["run_id"]
        )

        logger.info(
            "Best validation R2: %.4f",
            best_run[f"metrics.{selection_metric}"]
        )

        return best_run

    def create_training_dataset(self):
        """Create a training dataset by joining the output table with Feature Store features."""
        logger.info("Fetching dataset from Feature Store")
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

        logger.info(
            "Feature Store dataset loaded successfully"
        )

        return df

    def get_test_data(self):
        """Create a test dataset from the Feature Store and split it into X and y."""
        logger.info("Creating test dataset")
        target_column = self.config["ml"]["target"]["column"]

        df = self.create_training_dataset()

        df = df.orderBy("transaction_date")
        df = df.drop("transaction_date")
        df = df.toPandas()

        train_validate_df, test_df = train_test_split(
            df,
            test_size=0.20,
            shuffle=False
        )

        train_df, validate_df = train_test_split(
            train_validate_df,
            test_size=0.20,
            shuffle=False
        )

        X_test = test_df.drop(
            columns=[target_column]
        )

        y_test = test_df[target_column]

        logger.info(
            "Train shape: %s",
            train_df.shape
        )

        logger.info(
            "Validation shape: %s",
            validate_df.shape
        )

        logger.info(
            "Test shape: %s",
            test_df.shape
        )

        logger.info(
            "X_test shape: %s | y_test shape: %s",
            X_test.shape,
            y_test.shape
        )

        return X_test, y_test

    def load_model(self, run_id):
        """Load a candidate model from an MLflow run.

        Args:
            run_id: MLflow run ID of the model to load.

        Returns:
            Loaded sklearn model instance.
        """
        logger.info(
            "Loading candidate model from MLflow run"
        )
        model_artifact_name = self.config["ml"]["mlflow"]["model_artifact_name"]
        model_uri = (
            f"runs:/{run_id}/{model_artifact_name}"
        )

        logger.info(
            "Model URI: %s",
            model_uri
        )

        model = mlflow.sklearn.load_model(
            model_uri
        )

        logger.info(
            "Candidate model loaded successfully"
        )

        return model

    def load_encoder(self, run_id):
        """Load the OneHotEncoder artifact from an MLflow run.

        Args:
            run_id: MLflow run ID containing the encoder artifact.

        Returns:
            Loaded OneHotEncoder instance.
        """
        logger.info(
            "Loading encoder from MLflow run"
        )

        encoder_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=self.config["ml"]["artifacts"]["artifact_path"]
        )

        with open(encoder_path, "rb") as file:
            encoder = pickle.load(file)

        logger.info(
            "Encoder loaded successfully"
        )

        return encoder

    def encode_test_data(self, X_test, encoder):
        """One-hot encode the categorical columns of the test data.

        Args:
            X_test: Pandas DataFrame of test features.
            encoder: Fitted OneHotEncoder instance.

        Returns:
            Pandas DataFrame with encoded categorical and numeric columns.
        """
        logger.info("Encoding test data")

        category_columns = (self.config["ml"]["encoding"]["categorical_columns"])
        numeric_columns = (self.config["ml"]["encoding"]["numeric_columns"])

        encoded_data = encoder.transform(
            X_test[category_columns]
        )

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

        logger.info(
            "Encoded test data shape: %s",
            X_test_final.shape
        )

        return X_test_final

    def test_model(self, X_test, y_test, run_id):
        """Load and test a candidate model on the test dataset.

        Args:
            X_test: Pandas DataFrame of test features.
            y_test: Pandas Series of test labels.
            run_id: MLflow run ID of the model to test.

        Returns:
            R2 score of the model on the test data.
        """
        logger.info(
            "Testing candidate model"
        )

        model = self.load_model(run_id)

        encoder = self.load_encoder(run_id)

        X_test_final = self.encode_test_data(X_test,encoder)
        predictions = model.predict(X_test_final)

        test_r2 = r2_score(
            y_test,
            predictions
        )

        logger.info(
            "Test R2: %.4f",
            test_r2
        )

        return test_r2

    def register_model(self, run_id, test_r2):
        """Register the model in Unity Catalog if it passes the R2 threshold.

        Args:
            run_id: MLflow run ID of the model to register.
            test_r2: R2 score achieved on the test dataset.

        Returns:
            True if the model was registered, False otherwise.
        """
        threshold = self.config["ml"]["validation"]["test_r2_threshold"]

        if test_r2 <= threshold:

            logger.info(
                "Test R2 %.4f did not pass "
                "the 0.90 threshold",
                test_r2
            )

            logger.info(
                "Model will not be registered"
            )

            return False

        logger.info(
            "Test R2 %.4f passed threshold 0.90",
            test_r2
        )

        logger.info(
            "Registering model in Unity Catalog"
        )
        registry_uri = self.config["ml"]["registry"]["uri"]
        mlflow.set_registry_uri(registry_uri)

        model_artifact_name = self.config["ml"]["mlflow"]["model_artifact_name"]

        model_uri = (
            f"runs:/{run_id}/{model_artifact_name}"
        )
        model_name = self.config["ml"]["model_name"]["registered_name"]
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        self.model_version = (
            registered_model.version
        )

        self.run_id = run_id

        logger.info(
            "Model registered successfully. "
            "Version: %s",
            self.model_version
        )

        return True

    def assign_challenger(self, test_r2):
        """Assign the Challenger alias to the registered model version.

        Args:
            test_r2: R2 score to tag on the model version.
        """
        client = MlflowClient()
        challenger_alias = self.config["ml"]["registry"]["challenger_alias"]
        model_name = self.config["ml"]["model_name"]["registered_name"]
        client.set_model_version_tag(
            name=model_name,
            version=self.model_version,
            key="test_r2",
            value=str(test_r2)
        )

        client.set_model_version_tag(
            name=model_name,
            version=self.model_version,
            key="test_r2_status",
            value="cleared"
        )

        client.set_registered_model_alias(
            name=model_name,
            alias=challenger_alias,
            version=self.model_version
        )

        logger.info(
            "Model version %s assigned Challenger alias",
            self.model_version
        )

        return True

    def run(self):
        """Execute the full model selection, testing, and registration pipeline."""
        logger.info(
            "Model selection and testing pipeline started"
        )
        best_run = self.get_best_run()

        run_id = best_run["run_id"]

        X_test, y_test = self.get_test_data()

        test_r2 = self.test_model(
            X_test,
            y_test,
            run_id
        )
        registered = self.register_model(
            run_id,
            test_r2
        )

        if not registered:

            logger.info(
                "Model did not pass test R2. "
                "Challenger was not created."
            )

            return False

        self.assign_challenger(
            test_r2
        )

        logger.info(
            "Model selection and testing pipeline completed"
        )

        return True
