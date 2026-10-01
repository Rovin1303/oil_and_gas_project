import logging
from databricks.feature_engineering import FeatureEngineeringClient,FeatureLookup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from mlflow.models import infer_signature

logger = logging.getLogger("MLTraining")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False

class MLModelTraining:
    def __init__(self,spark,feature_client,feature_table,output_table,config):
        '''Initialize the MLModelTraining instance.

        Args:
            spark: SparkSession object
            feature_client: FeatureEngineeringClient instance
            feature_table: Name of the feature table in Feature Store
            output_table: Name of the label/output table
        '''
        self.spark = spark
        self.feature_client = feature_client
        self.feature_table_name = feature_table
        self.output_table_name = output_table
        self.config = config
        logger.info("ML Model Training instantiated")

    def create_training_dataset(self):
        '''Create a training dataset by joining the output table with features from the Feature Store.'''
        logger.info("Reading the %s",self.output_table_name)
        new_df = self.spark.table(self.output_table_name)
        training_df = self.feature_client.create_training_set(
            df = new_df,
            feature_lookups = [FeatureLookup(table_name=self.feature_table_name,lookup_key="primary_key")],
            label=self.config["ml"]["target"]["column"],
            exclude_columns = "primary_key"
        )
        logger.info("Joined the input and the target data")
        return training_df.load_df()
    
    def sort_data(self, df):
        '''Sort the DataFrame by transaction_date and drop the date column.'''
        logger.info("Sorting data by transaction_date")
        df = df.orderBy("transaction_date")
        df = df.drop("transaction_date")
        return df
    
    def split_data(self,df):
        '''Split the DataFrame into training, validation, and test sets chronologically.'''
        logger.info("Split data into training,validation and testing data")
        data_df = df.toPandas()
        training_df,test_df = train_test_split(data_df,test_size = 0.20,shuffle = False)
        train_df,validate_df = train_test_split(training_df,test_size = 0.20,shuffle = False)
        return train_df,validate_df,test_df
    
    def split_input_output(self, train_df, validation_df, test_df):
        '''Separate features (X) and target (y) for each dataset split.'''

        logger.info("Splitting data into X and y")
        X_train = train_df.drop(columns=["total_demand"])
        y_train = train_df["total_demand"]
        X_validate = validation_df.drop(columns=["total_demand"])
        y_validate = validation_df["total_demand"]
        X_test = test_df.drop(columns=["total_demand"])
        y_test = test_df["total_demand"]
        logger.info("X and y split completed")
        logger.info("Train X: %s | Train y: %s",X_train.shape,y_train.shape)
        logger.info("Validation X: %s | Validation y: %s",X_validate.shape,y_validate.shape)
        logger.info("Test X: %s | Test y: %s",X_test.shape,y_test.shape)
        print(X_train.isnull().sum())
        print(X_validate.isnull().sum())
        print(X_test.isnull().sum())
        return X_train,y_train,X_validate,y_validate,X_test,y_test
    
    def encode_categorical_data(self, X_train):
        '''One-hot encode categorical columns and save the encoder to disk.'''

        logger.info("Starting categorical encoding")
        category_columns = (self.config["ml"]["encoding"]["categorical_columns"])
        rest_columns = (self.config["ml"]["encoding"]["numeric_columns"])
        encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        X_train_encoded = encoder.fit_transform(X_train[category_columns])
        encoded_df = pd.DataFrame(X_train_encoded,columns=encoder.get_feature_names_out(category_columns),
                index=X_train.index)
        rest_df = X_train[rest_columns]
        X_train_final = pd.concat([rest_df, encoded_df],axis=1)
        logger.info("Categorical encoding completed")
        artifact_directory = (self.config["ml"]["artifacts"]["directory"])
        encoder_filename = (self.config["ml"]["artifacts"]["encoder"])
        os.makedirs(artifact_directory,exist_ok=True)
        encoder_path = os.path.join(artifact_directory,encoder_filename)
        with open(encoder_path,"wb") as file:
            pickle.dump(encoder,file)

        logger.info("OneHotEncoder saved successfully at %s",encoder_path)

        return X_train_final,encoder
    
    def load_encoder(self):
        '''Load the saved OneHotEncoder.'''

        logger.info("Loading OneHotEncoder")
        artifact_directory = (self.config["ml"]["artifacts"]["directory"])
        encoder_filename = (self.config["ml"]["artifacts"]["encoder"])
        encoder_path = os.path.join(artifact_directory,encoder_filename)
        with open(encoder_path, "rb") as file:
            encoder = pickle.load(file)
        logger.info("OneHotEncoder loaded successfully")
        return encoder

    def encode_validation_data(self, X_validate):
        '''Encode validation/test data using the saved OneHotEncoder.'''

        logger.info("Starting categorical encoding")
        category_columns = (self.config["ml"]["encoding"]["categorical_columns"])
        rest_columns = (self.config["ml"]["encoding"]["numeric_columns"])
        encoder = self.load_encoder()
        logger.info("OneHotEncoder fetched successfully")

        X_validate_encoded = encoder.transform(X_validate[category_columns])
        encoded_df = pd.DataFrame(X_validate_encoded,columns=encoder.get_feature_names_out(category_columns),
                                  index=X_validate.index)
        rest_df = X_validate[rest_columns]
        X_validate_final = pd.concat([rest_df,encoded_df], axis=1)

        logger.info("Validation data encoding completed")
        return X_validate_final
    
    def create_experiment(self):
        '''Set up the MLflow experiment for model training.'''

        experiment_name =  self.config["ml"]["experiment_name"]

        mlflow.set_experiment(experiment_name)

        logger.info("MLflow experiment created/set: %s",experiment_name)

    def train_model(self,X_train,y_train,X_validate,y_validate):
        '''Train a DecisionTreeRegressor and log it to MLflow with metrics and artifacts.'''

        self.create_experiment()
        logger.info("Starting Decision Tree model training")
        model_config = self.config["ml"]["model"]
        params = {
            "criterion": model_config["criterion"],
            "max_depth": model_config["max_depth"],
            "min_samples_split": model_config["min_samples_split"],
            "min_samples_leaf": model_config["min_samples_leaf"],
            "random_state": model_config["random_state"]
        }
        model = DecisionTreeRegressor(**params)
        with mlflow.start_run(run_name=self.config["ml"]["mlflow"]["run_name"]):
            mlflow.log_params(params)
            mlflow.set_tags({
                "run_name": "fifth run",
                "model_type": "DecisionTreeRegressor",
                "problem_type": "regression",
                "dataset": "demand_prediction",
                "target": self.config["ml"]["target"]["column"],
                "feature_store": self.feature_table_name
            })
            model.fit(X_train,y_train)
            logger.info("Decision Tree model trained successfully")
            y_pred = model.predict(X_validate)
            logger.info("X_validate shape: %s", X_validate.shape)
            logger.info("y_validate shape: %s", y_validate.shape)
            logger.info("y_pred shape: %s", y_pred.shape)
            signature = infer_signature(X_validate,y_pred)
            mae = mean_absolute_error(y_validate,y_pred)
            rmse = mean_squared_error(y_validate,y_pred) ** 0.5
            r2 = r2_score(y_validate,y_pred)
            mlflow.log_metrics({
                "validation_mae": mae,
                "validation_rmse": rmse,
                "validation_r2": r2
            })
            artifact_directory = (self.config["ml"]["artifacts"]["directory"])
            encoder_filename = (self.config["ml"]["artifacts"]["encoder"])
            encoder_path = os.path.join(artifact_directory,encoder_filename)
            mlflow.log_artifact(encoder_path,artifact_path="preprocessing")

            logger.info("OneHotEncoder logged as MLflow artifact")

            logger.info("Validation MAE: %.4f", mae)
            logger.info("Validation RMSE: %.4f", rmse)
            logger.info("Validation R2: %.4f", r2)
            model_name = self.config["ml"]["mlflow"]["run_name"]
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                signature=signature,
                input_example=X_validate.iloc[[0]],
                skops_trusted_types=["sklearn.tree._tree.Tree"]
                # registered_model_name="oag.ml.demand_prediction_model"
            )
            logger.info("Model logged successfully")
        #logger.info("MLflow run completed: %s",run_id)
        return model

    
    def run(self):
        '''Execute the full ML training pipeline from data loading to model logging.'''
        logger.info("ML pipeline started")
        df = self.create_training_dataset()
        df = self.sort_data(df)
        train_df, validate_df, test_df = self.split_data(df)
        X_train,y_train,X_validate,y_validate,X_test,y_test = self.split_input_output(train_df,validate_df,test_df)
        X_train_final, encoder = self.encode_categorical_data(X_train)
        X_validate_final = self.encode_validation_data(X_validate)
        model= self.train_model(X_train_final,y_train,X_validate_final,y_validate)
        logger.info("ML pipeline completed")
    



