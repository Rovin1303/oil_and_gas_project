import logging
import pyspark.sql.functions as F
from pyspark.sql.window import Window

class Features:

    def __init__(self, spark, catalog_name,gold_table,gold_feature):
        """Initialize the Features instance.

        Args:
            spark: SparkSession object
            catalog_name: Unity Catalog name
            gold_table: Name of the source gold table
            gold_feature: Name of the target gold feature table
        """
        self.spark = spark
        self.catalog_name = catalog_name
        self.gold_table = gold_table
        self.gold_feature = gold_feature
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:

            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

    def read_gold(self):
        """Read the gold table from Unity Catalog and return it as a DataFrame."""

        self.logger.info("Reading Gold table: %s", self.gold_table)
        try:
            gold_df = self.spark.table(self.gold_table)

            self.logger.info("Gold table read successfully")

            return gold_df

        except Exception as e:

            self.logger.error("Failed to read Gold table: %s", str(e))
            raise
            
    def create_features(self, gold_df):
        """Engineer lag, rolling average, and calendar features from the gold table.

        Args:
            gold_df: Spark DataFrame containing gold-level aggregated data.

        Returns:
            DataFrame with demand_1, demand_7, rolling_7_day_avg, month, and day_of_week columns.
        """

        self.logger.info("Starting feature engineering")
        w = (Window.partitionBy(
                    "product_name",
                    "destination_city"
                ).orderBy(
                    "transaction_date"
                ))

        rolling_w = (
                Window
                .partitionBy(
                    "product_name",
                    "destination_city"
                )
                .orderBy(
                    "transaction_date"
                )
                .rowsBetween(-7, -1)
            )

        feature_df = (
            gold_df.withColumn(
                    "demand_1",
                    F.lag(
                        "total_demand",
                        1
                    ).over(w)
                )
            .withColumn(
                    "demand_7",
                    F.lag(
                        "total_demand",
                        7
                    ).over(w)
                )
            .withColumn(
                    "rolling_7_day_avg",
                    F.round(
                        F.avg(
                            "total_demand"
                        ).over(rolling_w),
                        2
                    )
            )
            .withColumn(
                    "month",
                    F.month(
                        "transaction_date"
                    )
                )

            .withColumn(
                    "day_of_week",
                    F.dayofweek(
                        "transaction_date"
                    )
                )
            )

        self.logger.info(
                "All 5 features created successfully"
            )

        return feature_df


    def create_feature_table(self, feature_df):
        """Write the feature DataFrame as a Delta table to Unity Catalog."""
        self.logger.info(
            "Writing feature table: %s", self.gold_feature
        )

        try:

            (
                feature_df.write
                .format("delta")
                .mode("overwrite")
                .saveAsTable(self.gold_feature)
            )

            self.logger.info(
                "Feature table created successfully: %s", self.gold_feature
            )

            return True

        except Exception as e:

            self.logger.error("Failed to create feature table: %s", str(e))
            raise 

    def run(self):
        """Execute the full feature engineering pipeline."""

        self.logger.info(
            "========== Feature Engineering Started =========="
        )

        try:

            # Step 1
            gold_df = self.read_gold()

            # Step 2
            feature_df = self.create_features(
                gold_df
            )

            # Step 3
            self.create_feature_table(
                feature_df
            )

            self.logger.info(
                "========== Feature Engineering Completed Successfully =========="
            )

            return True

        except Exception as e:
            self.logger.error("Feature pipeline failed: %s", str(e))
            raise