from src.data_engineering.silver import SilverTransformation


def test_drop_duplicates(spark):
    """Test that drop_duplicates removes rows with duplicate transaction_id."""
    data = [
        (1, "2026-01-01", "Product_A", "Mumbai", 100),
        (1, "2026-01-01", "Product_A", "Mumbai", 100),
        (2, "2026-01-02", "Product_B", "Delhi", 200),
    ]

    df = spark.createDataFrame(
        data,
        [
            "transaction_id",
            "transaction_date",
            "product_name",
            "destination_city",
            "ordered_quantity",
        ],
    )
    silver_obj = SilverTransformation(
        spark=spark,
        catalog_name="oag",
        bronze_table="dummy_bronze",
        silver_table="dummy_silver",
    )
    result = silver_obj.drop_duplicates(df)
    assert result.count() == 2
