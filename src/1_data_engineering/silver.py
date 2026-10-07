import logging


class SilverTransformation:

    def __init__(self, spark, catalog_name,bronze_table,silver_table):
        '''Initialize the SilverTransformation instance.

        Args:
            spark: SparkSession object
            catalog_name: Unity Catalog name
        '''
        self.spark = spark
        self.catalog_name = catalog_name
        self.bronze_table = bronze_table
        self.silver_table = silver_table
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def read_bronze(self):
        '''Read the bronze table from Unity Catalog and return it as a DataFrame.'''
        self.logger.info(
            "Reading Bronze table: %s", self.bronze_table
        )
        try:
            bronze_df = self.spark.table(self.bronze_table)
            self.logger.info(
                "Bronze table read successfully"
            )
            return bronze_df
        except Exception as e:
            self.logger.error(
                "Failed to read Bronze table: %s", str(e))
            raise
            
    def drop_columns(self, df):
        '''Drop metadata columns (ingestion_timestamp, source_system, operation_type) from the DataFrame.'''
        self.logger.info("Starting column removal")
        try:
            silver_df = df.drop("ingestion_timestamp", "source_system", "operation_type")
            self.logger.info("Dropped columns")
            return silver_df
        
        except Exception as e:
            self.logger.error("Failed to drop columns: %s", str(e))
            raise

    def drop_duplicates(self, df):
        '''Remove duplicate records based on transaction_id.'''
        self.logger.info("Starting duplicate removal")
        try:
            silver_df = df.dropDuplicates(["transaction_id"])
            self.logger.info("Duplicates removed using transaction_id")
            return silver_df

        except Exception as e:
            self.logger.error("Failed to remove duplicates: %s", str(e))
            raise

    def create_silver_table(self, df):
        '''Write the transformed DataFrame as a silver Delta table to Unity Catalog.'''
        self.logger.info("Writing Silver table: %s", self.silver_table)
        try:
            (
                df.write
                .format("delta")
                .mode("overwrite")
                .saveAsTable(self.silver_table)
            )
            self.logger.info(
                "Silver table created successfully: %s", self.silver_table
            )
            return True
        except Exception as e:
            self.logger.error("Failed to create Silver table: %s", str(e))
            raise

    def run(self):
        '''Execute the full silver transformation pipeline.'''

        self.logger.info("========== Silver transformation started ===========")
        try:
            bronze_df = self.read_bronze()
            silver_df = self.drop_columns(bronze_df)
            silver_df = self.drop_duplicates(silver_df)
            self.create_silver_table(silver_df)
            self.logger.info("========== Silver transformation completed successfully ==========")
            return True

        except Exception as e:
            self.logger.error("Silver transformation failed: %s", str(e))
            raise
