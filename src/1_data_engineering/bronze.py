import logging

import pyspark.sql.functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


class BronzeIngestion:

    def __init__(self, spark, data_path, catalog_name,table_name):
        '''
        Initialize the BronzeIngestion instance.

        Args:
            spark: SparkSession object
            data_path: Path to the source CSV file
            catalog_name: Unity Catalog name
        '''
        self.spark = spark
        self.data_path = data_path
        self.catalog_name = catalog_name
        self.table_name = table_name

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def define_schema(self):
        '''Define the schema for the bronze layer supply chain data.'''

        self.logger.info("Defining Bronze schema")

        bronze_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("transaction_date", DateType(), True),
            StructField("transaction_year", IntegerType(), True),
            StructField("transaction_quarter", IntegerType(), True),
            StructField("transaction_month", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("product_category", StringType(), True),
            StructField("quantity_unit", StringType(), True),
            StructField("supplier_name", StringType(), True),
            StructField("supplier_country", StringType(), True),
            StructField("supplier_reliability_score", DoubleType(), True),
            StructField("refinery_name", StringType(), True),
            StructField("destination_city", StringType(), True),
            StructField("transportation_mode", StringType(), True),
            StructField("ordered_quantity", DoubleType(), True),
            StructField("demand_quantity", DoubleType(), True),
            StructField("available_inventory", DoubleType(), True),
            StructField("unit_price_usd", DoubleType(), True),
            StructField("product_cost_usd", DoubleType(), True),
            StructField("transportation_cost_usd", DoubleType(), True),
            StructField("total_cost_usd", DoubleType(), True),
            StructField("expected_lead_time_days", IntegerType(), True),
            StructField("actual_lead_time_days", IntegerType(), True),
            StructField("delay_days", IntegerType(), True),
            StructField("is_delayed", IntegerType(), True),
            StructField("is_stockout", IntegerType(), True),
            StructField("quality_status", StringType(), True),
            StructField("quality_score", DoubleType(), True),
            StructField("disruption_type", StringType(), True),
            StructField("delivery_status", StringType(), True),
            StructField("ingestion_timestamp", StringType(), True),
            StructField("source_system", StringType(), True),
            StructField("operation_type", StringType(), True),
        ])

        self.logger.info("Bronze schema defined successfully")

        return bronze_schema
    
    def create_dataframe(self):
        '''Read the source CSV file and create a Spark DataFrame with the bronze schema.'''

        self.logger.info("Reading source file: %s", self.data_path)
        try:
            bronze_schema = self.define_schema()
            bronze_df = self.spark.read.csv(
                path=self.data_path,
                schema=bronze_schema,
                header=True,
                sep=","
            )

            bronze_df = bronze_df.withColumn("ingested_at",F.current_timestamp())

            self.logger.info("DataFrame created successfully")

            return bronze_df

        except Exception as e:
            self.logger.error(
                "Failed to create DataFrame: %s", str(e)
            )
            raise
    
    def create_table(self):
        '''Create the bronze Delta table by reading the source file and writing to Unity Catalog.'''
        self.logger.info("Starting Bronze table creation: %s", self.table_name)
        try:
            bronze_df = self.create_dataframe()
            record_count = bronze_df.count()
            self.logger.info("Records read from source: %s", record_count)
            (
                bronze_df.write
                .format("delta")
                .mode("overwrite")
                .saveAsTable(self.table_name)
            )

            self.logger.info("Bronze table created successfully: %s", self.table_name)
            return True
        except Exception as e:
            self.logger.error(
                "Bronze table creation failed: %s", str(e)
            )
            raise
    
    def run(self):
        '''Execute the full bronze ingestion pipeline.'''
        self.logger.info("========== Bronze ingestion started ==========")
        try:
            result = self.create_table()
            if result:
                self.logger.info("========== Bronze ingestion completed successfully ===========")
                return result
        except Exception:
                self.logger.error("========== Bronze ingestion failed ==========")
                raise