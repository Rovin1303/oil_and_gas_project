import logging

from pyspark.sql import functions as F

logger = logging.getLogger("FeatureStore")
logger.setLevel(logging.INFO)
if not logger.handlers:
   handler = logging.StreamHandler()
   formatter = logging.Formatter(
       "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   )
   handler.setFormatter(formatter)
   logger.addHandler(handler)

class FeatureStore:
    def __init__( self,spark,feature_client,gold_table,feature_table,label_table):
        '''Initialize the Features instance.

        Args:
            spark: SparkSession object
            feature_client: FeatureEngineeringClient instance
            gold_table: Name of the gold source table
            feature_table: Name of the feature table to create
            label_table: Name of the label table to create
        '''

        self.spark = spark
        self.feature_client = feature_client
        self.gold_table = gold_table
        self.feature_table_name = feature_table
        self.label_table_name = label_table
        logger.info("Features class has instantiated")
        logger.info("Feature table Name: %s",self.feature_table_name)
        logger.info("Label table: %s",self.label_table_name)

    def read_gold_table(self):
        '''Read the gold table from Unity Catalog and return it as a DataFrame.'''
        logger.info("Reading Gold feature table")
        df = self.spark.table(self.gold_table)
        logger.info("Gold feature table loaded successfully with %d columns",len(df.columns))
        return df
    
    def convert_nulls(self,df):
        """Replace null values in lag and rolling average columns with zero."""
        df = (df.withColumn("demand_1",F.coalesce(F.col("demand_1"),F.lit(0)))
            .withColumn("demand_7",F.coalesce(F.col("demand_7"),F.lit(0)))
            .withColumn("rolling_7_day_avg",F.coalesce(F.col("rolling_7_day_avg"),F.lit(0))))
        return df

    def create_primary_key(self,df):
        '''Create a composite primary key from transaction_date, product_name, and destination_city.'''
        logger.info("Creating primary key for feature table")
        df = df.withColumn("primary_key",
            F.concat_ws("-",
                F.date_format(F.col("transaction_date"),"yyyyMMdd"),
                F.col("product_name"),
                F.col("destination_city")))
        logger.info("Primary key has been created")
        return df
    
    def split_features(self, df):
        '''Split the gold DataFrame into feature columns (X) and label columns (y).'''

        logger.info("Splitting features and target")
        feature_columns = [
            "primary_key","transaction_date","product_name","destination_city","avg_unit_price","total_inventory",
            "transaction_count","demand_1","demand_7","rolling_7_day_avg","month","day_of_week"]

        label_columns = ["primary_key","total_demand"]
        X = df.select(feature_columns)
        y = df.select(label_columns)
        logger.info("Features df created")
        logger.info("Label df is created")
        return X, y
    
    def load_features(self, X):
        '''Create and load the feature table into the Databricks Feature Store.'''
        logger.info("Loading features into Feature Store: %s",self.feature_table_name)
        self.feature_client.create_table(
            name=self.feature_table_name,
            primary_keys=["primary_key"],
            df=X,
            description="Feature table for demand prediction",
            tags={
            "domain": "time series data",
            "project": "demand prediction",
            "layer": "gold",
            "features":"lag,inventory,demand",
            "owner": "ml_team"
        }
        )
        logger.info("Feature Store table created successfully")
    
    def load_labels(self, y):
        '''Write the label DataFrame as a Delta table to Unity Catalog.'''
        logger.info("Loading labels into table: %s",self.label_table_name)
        (
            y.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(self.label_table_name)
        )
        logger.info("Label table created successfully")
    
    def run(self):
        '''Execute the full feature store creation pipeline.'''

        logger.info("Feature Store Process")
        gold_df = self.read_gold_table()
        gold_df = self.convert_nulls(gold_df)
        gold_df = self.create_primary_key(gold_df)
        X, y = self.split_features(gold_df)
        self.load_features(X)
        self.load_labels(y)
        logger.info("Process Complete Feature Store Created")
        return True
        

