import sys
import os

# Add project root to sys.path so 'from src...' imports resolve
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Provide a SparkSession for tests."""
    spark = SparkSession.builder.getOrCreate()
    yield spark
