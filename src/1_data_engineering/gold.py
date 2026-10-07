import logging
import pyspark.sql.functions as F
from pyspark.sql.window import Window

class GoldTransformation:

    def __init__(self, spark, catalog_name,silver_table,gold_table):
        """Initialize the GoldTransformation instance.

        Args:
            spark: SparkSession object
            catalog_name: Unity Catalog name
            silver_table: Name of the source silver table
            gold_table: Name of the target gold table
        """
        self.spark = spark
        self.catalog_name = catalog_name
        self.silver_table = silver_table
        self.gold_table = gold_table

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def read_silver(self):
        """Read the silver table from Unity Catalog and return it as a DataFrame."""
        self.logger.info(
            "Reading Silver table: %s", self.silver_table
        )
        try:
            silver_df = self.spark.table(self.silver_table)
            self.logger.info(
                "Silver table read successfully"
            )
            return silver_df
        except Exception as e:
            self.logger.error(
                "Failed to read Silver table: %s", str(e)
            )
            raise

    def gold_table_df(self, silver_df):
        """Aggregate silver data by date, product, and city into gold-level metrics."""
        self.logger.info(
            "Starting Gold table aggregation")
        try:
            df = (silver_df.groupBy(
                    "transaction_date",
                    "product_name",
                    "destination_city")
                .agg(
                    F.round(F.sum("demand_quantity"), 2).alias("total_demand"),
                    F.round(F.avg("unit_price_usd"), 2).alias("avg_unit_price"),
                    F.round(F.sum("available_inventory"), 2).alias("total_inventory"),
                    F.count("*").alias("transaction_count")
                ))
            self.logger.info("Gold table aggregation completed successfully")
            return df
        except Exception as e:
            raise self.logger.error("Gold aggregation failed: %s", str(e))
    
    def create_gold_table(self, df):
        """Write the aggregated gold DataFrame as a Delta table to Unity Catalog."""
        self.logger.info("Writing Gold table: %s", self.gold_table)
        try:
            (
            df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(self.gold_table)
        )
            self.logger.info(
                "Gold table created successfully: %s", self.gold_table
            )
            return True
        except Exception as e:
            self.logger.error("Failed to create Gold table: %s", str(e))
            raise 
    
    def run(self):
        """Execute the full gold transformation pipeline."""
        self.logger.info("========== Gold transformation started ==========")
        try:
            silver_df = self.read_silver()
            gold_df = self.gold_table_df(silver_df)
            self.create_gold_table(gold_df)
            self.logger.info("========== Gold transformation completed successfully ==========")
            return True
        except Exception as e:
            raise self.logger.error("Gold transformation failed: %s", str(e))