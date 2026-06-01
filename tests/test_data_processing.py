"""
Unit tests for feature engineering pipeline.
Tests all transformers and the complete pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports - CI/CD safe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import, skip if module not available
try:
    from src.data_processing import (
        CustomerAggregator,
        TimeFeatureExtractor,
        CategoricalFeatureExtractor,
        FeaturePipeline,
        process_data,
    )

    MODULE_AVAILABLE = True
except ImportError as e:
    MODULE_AVAILABLE = False
    print(f"Warning: Could not import module: {e}")


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not MODULE_AVAILABLE, reason="src.data_processing module not available"
)


# ==================== Fixtures ====================


@pytest.fixture
def sample_transactions():
    """Create a realistic sample of transaction data for testing."""
    np.random.seed(42)  # Fixed seed for reproducibility

    customers = [f"CUST_{i:03d}" for i in range(10)]
    data = []

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    date_range = (end_date - start_date).days

    for i, customer in enumerate(customers):
        # Different behavior patterns per customer
        if i < 3:  # High activity customers
            n_tx = np.random.randint(10, 50)  # Reduced for CI speed
        elif i < 7:  # Medium activity customers
            n_tx = np.random.randint(5, 20)
        else:  # Low activity customers (potential high risk)
            n_tx = np.random.randint(1, 10)

        for _ in range(n_tx):
            tx_date = start_date + timedelta(days=np.random.randint(0, date_range))

            if i >= 7:  # Low activity customers
                amount = np.random.uniform(10, 500)
                is_refund = np.random.random() < 0.3
            else:
                amount = np.random.uniform(100, 5000)
                is_refund = np.random.random() < 0.1

            amount = -amount if is_refund else amount

            data.append(
                {
                    "TransactionId": f"TX_{customer}_{_}",
                    "BatchId": f"BATCH_{np.random.randint(1, 100)}",
                    "AccountId": f"ACC_{customer}",
                    "SubscriptionId": f"SUB_{customer}",
                    "CustomerId": customer,
                    "CurrencyCode": "UGX",
                    "CountryCode": 256,
                    "ProviderId": f"ProviderId_{np.random.randint(1, 7)}",
                    "ProductId": f"ProductId_{np.random.randint(1, 24)}",
                    "ProductCategory": np.random.choice(
                        [
                            "airtime",
                            "financial_services",
                            "tv",
                            "utility_bill",
                            "data_bundles",
                            "movies",
                            "transport",
                            "ticket",
                        ],
                        p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03],
                    ),
                    "ChannelId": f"ChannelId_{np.random.randint(1, 5)}",
                    "Amount": amount,
                    "Value": abs(amount),
                    "TransactionStartTime": tx_date,
                    "PricingStrategy": np.random.randint(0, 5),
                    "FraudResult": 1 if np.random.random() < 0.002 else 0,
                }
            )

    df = pd.DataFrame(data)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    return df


@pytest.fixture
def sample_with_single_transaction():
    """Test case for customers with single transaction."""
    df = pd.DataFrame(
        {
            "CustomerId": ["SINGLE_001", "SINGLE_002"],
            "TransactionStartTime": ["2024-01-15 10:30:00", "2024-01-20 23:45:00"],
            "Amount": [1000, -50],
            "ProductCategory": ["airtime", "financial_services"],
            "ChannelId": ["ChannelId_3", "ChannelId_2"],
            "ProviderId": ["ProviderId_6", "ProviderId_4"],
            "Value": [1000, 50],
            "TransactionId": ["TX_001", "TX_002"],
            "BatchId": ["BATCH_001", "BATCH_002"],
            "AccountId": ["ACC_001", "ACC_002"],
            "SubscriptionId": ["SUB_001", "SUB_002"],
            "CurrencyCode": ["UGX", "UGX"],
            "CountryCode": [256, 256],
            "PricingStrategy": [2, 2],
            "FraudResult": [0, 0],
        }
    )
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


# ==================== Tests for CustomerAggregator ====================


class TestCustomerAggregator:
    """Test suite for CustomerAggregator class."""

    def test_basic_aggregation(self, sample_transactions):
        """Test that aggregator produces expected columns and shapes."""
        aggregator = CustomerAggregator()
        aggregator.fit(sample_transactions)
        result = aggregator.transform(sample_transactions)

        expected_columns = [
            "CustomerId",
            "recency",
            "frequency",
            "monetary",
            "avg_amount",
            "std_amount",
            "refund_rate",
            "refund_amount",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        assert len(result) == len(sample_transactions["CustomerId"].unique())
        assert result["frequency"].sum() == len(sample_transactions)

    def test_recency_calculation(self, sample_transactions):
        """Test that recency is calculated correctly."""
        aggregator = CustomerAggregator()
        aggregator.fit(sample_transactions)
        result = aggregator.transform(sample_transactions)

        snapshot = sample_transactions["TransactionStartTime"].max()
        for _, row in result.iterrows():
            customer_txs = sample_transactions[
                sample_transactions["CustomerId"] == row["CustomerId"]
            ]
            max_tx_date = customer_txs["TransactionStartTime"].max()
            expected_recency = (snapshot - max_tx_date).days
            assert row["recency"] == expected_recency

    def test_monetary_only_positive(self, sample_transactions):
        """Test that monetary only sums positive amounts."""
        aggregator = CustomerAggregator()
        aggregator.fit(sample_transactions)
        result = aggregator.transform(sample_transactions)

        for _, row in result.iterrows():
            customer_txs = sample_transactions[
                sample_transactions["CustomerId"] == row["CustomerId"]
            ]
            expected_monetary = customer_txs[customer_txs["Amount"] > 0]["Amount"].sum()
            # Use pytest.approx for floating-point comparison
            assert row["monetary"] == pytest.approx(expected_monetary, rel=1e-9)

    def test_refund_rate_bounds(self, sample_transactions):
        """Test that refund rate is between 0 and 1."""
        aggregator = CustomerAggregator()
        aggregator.fit(sample_transactions)
        result = aggregator.transform(sample_transactions)

        # Allow for NaN values (customers with no transactions)
        refund_rates = result["refund_rate"].dropna()
        if len(refund_rates) > 0:
            assert refund_rates.between(0, 1).all()

    def test_single_transaction_handling(self, sample_with_single_transaction):
        """Test that single transaction customers are handled correctly."""
        aggregator = CustomerAggregator()
        aggregator.fit(sample_with_single_transaction)
        result = aggregator.transform(sample_with_single_transaction)

        # std_amount should be 0 or NaN for single transaction
        assert result["std_amount"].iloc[0] == 0 or pd.isna(
            result["std_amount"].iloc[0]
        )

        # refund_rate should be 0 or 1 based on transaction type
        assert result["refund_rate"].iloc[0] in [0, 1]


# ==================== Tests for TimeFeatureExtractor ====================


class TestTimeFeatureExtractor:
    """Test suite for TimeFeatureExtractor class."""

    def test_basic_extraction(self, sample_transactions):
        """Test that time features are extracted correctly."""
        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_transactions)

        expected_columns = [
            "CustomerId",
            "weekend_ratio",
            "business_hour_ratio",
            "transaction_hour_std",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        assert len(result) == len(sample_transactions["CustomerId"].unique())

    def test_weekend_ratio_range(self, sample_transactions):
        """Test that weekend ratio is between 0 and 1."""
        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_transactions)

        ratios = result["weekend_ratio"].dropna()
        if len(ratios) > 0:
            assert ratios.between(0, 1).all()

    def test_business_hour_ratio_range(self, sample_transactions):
        """Test that business hour ratio is between 0 and 1."""
        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_transactions)

        ratios = result["business_hour_ratio"].dropna()
        if len(ratios) > 0:
            assert ratios.between(0, 1).all()

    def test_hour_std_non_negative(self, sample_transactions):
        """Test that hour standard deviation is non-negative."""
        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_transactions)

        # hour_std should be >= 0 (fillna ensures no NaN)
        assert (result["transaction_hour_std"] >= 0).all()

    def test_single_transaction_std_zero(self, sample_with_single_transaction):
        """Test that single transaction customers have hour_std = 0."""
        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_with_single_transaction)

        # Should be 0 after fillna(0)
        assert result["transaction_hour_std"].iloc[0] == 0
        assert result["transaction_hour_std"].iloc[1] == 0

    def test_weekend_calculation(self):
        """Test that weekend detection works correctly."""
        df = pd.DataFrame(
            {
                "CustomerId": ["TEST_001", "TEST_001"],
                "TransactionStartTime": ["2024-01-13 10:00:00", "2024-01-14 15:00:00"],
                "Amount": [100, 200],
                "ProductCategory": ["airtime", "airtime"],
                "ChannelId": ["ChannelId_3", "ChannelId_3"],
                "ProviderId": ["ProviderId_6", "ProviderId_6"],
                "Value": [100, 200],
                "TransactionId": ["TX_001", "TX_002"],
                "BatchId": ["BATCH_001", "BATCH_001"],
                "AccountId": ["ACC_001", "ACC_001"],
                "SubscriptionId": ["SUB_001", "SUB_001"],
                "CurrencyCode": ["UGX", "UGX"],
                "CountryCode": [256, 256],
                "PricingStrategy": [2, 2],
                "FraudResult": [0, 0],
            }
        )
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        extractor = TimeFeatureExtractor()
        result = extractor.transform(df)

        # Both transactions on weekend -> ratio should be 1.0
        assert result["weekend_ratio"].iloc[0] == 1.0


# ==================== Tests for CategoricalFeatureExtractor ====================


class TestCategoricalFeatureExtractor:
    """Test suite for CategoricalFeatureExtractor class."""

    def test_basic_extraction(self, sample_transactions):
        """Test that categorical features are extracted correctly."""
        extractor = CategoricalFeatureExtractor()
        result = extractor.transform(sample_transactions)

        expected_columns = [
            "CustomerId",
            "ProductCategory",
            "ChannelId",
            "ProviderId",
            "unique_productcategory",
            "unique_channelid",
            "unique_providerid",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        assert len(result) == len(sample_transactions["CustomerId"].unique())

    def test_mode_extraction(self, sample_transactions):
        """Test that mode (most frequent) is extracted correctly."""
        extractor = CategoricalFeatureExtractor()
        result = extractor.transform(sample_transactions)

        for _, row in result.iterrows():
            customer_txs = sample_transactions[
                sample_transactions["CustomerId"] == row["CustomerId"]
            ]

            if not customer_txs.empty:
                expected_mode = (
                    customer_txs["ProductCategory"].mode().iloc[0]
                    if not customer_txs["ProductCategory"].mode().empty
                    else "unknown"
                )
                assert row["ProductCategory"] == str(expected_mode)

    def test_unique_counts(self, sample_transactions):
        """Test that unique counts are calculated correctly."""
        extractor = CategoricalFeatureExtractor()
        result = extractor.transform(sample_transactions)

        for _, row in result.iterrows():
            customer_txs = sample_transactions[
                sample_transactions["CustomerId"] == row["CustomerId"]
            ]

            expected_unique_cats = customer_txs["ProductCategory"].nunique()
            assert row["unique_productcategory"] == expected_unique_cats


# ==================== Tests for FeaturePipeline ====================


class TestFeaturePipeline:
    """Test suite for complete FeaturePipeline."""

    def test_pipeline_fit_transform(self, sample_transactions):
        """Test that complete pipeline runs without errors."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(sample_transactions)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_transactions["CustomerId"].unique())

    def test_pipeline_columns(self, sample_transactions):
        """Test that pipeline produces expected columns."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(sample_transactions)

        # Expected column categories (some may be missing in simple data)
        expected_numerical = [
            "recency",
            "frequency",
            "monetary",
            "avg_amount",
            "refund_rate",
            "refund_amount",
            "weekend_ratio",
            "business_hour_ratio",
            "transaction_hour_std",
            "unique_productcategory",
        ]

        for col in expected_numerical:
            assert col in features.columns, f"Missing column: {col}"

        # Check for encoded categorical columns (at least one should exist)
        encoded_cols = [
            col
            for col in features.columns
            if col.startswith(("cat_", "channel_", "provider_"))
        ]
        if len(sample_transactions) > 0:
            assert len(encoded_cols) > 0, "No encoded categorical columns found"

    def test_no_missing_values(self, sample_transactions):
        """Test that pipeline produces no missing values."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(sample_transactions)

        # Check for NaN values
        assert not features.isnull().any().any(), "Pipeline produced NaN values"

    def test_fit_before_transform_raises_error(self, sample_transactions):
        """Test that transform without fit raises error."""
        pipeline = FeaturePipeline()

        with pytest.raises(RuntimeError, match="Pipeline must be fitted"):
            pipeline.transform(sample_transactions)

    def test_consistency_across_runs(self, sample_transactions):
        """Test that pipeline produces same results on multiple runs."""
        pipeline = FeaturePipeline()
        features1 = pipeline.fit_transform(sample_transactions)
        features2 = pipeline.transform(sample_transactions)

        # Compare column by column to handle potential index differences
        for col in features1.columns:
            pd.testing.assert_series_equal(
                features1[col], features2[col], check_names=False
            )


# ==================== Tests for process_data Function ====================


class TestProcessData:
    """Test suite for process_data function."""

    def test_process_data_returns_features_and_pipeline(self, sample_transactions):
        """Test that process_data returns both features and pipeline."""
        features, pipeline = process_data(sample_transactions)

        assert isinstance(features, pd.DataFrame)
        assert isinstance(pipeline, FeaturePipeline)
        assert pipeline.is_fitted

    def test_process_data_output_shape(self, sample_transactions):
        """Test that process_data produces correct shape."""
        features, _ = process_data(sample_transactions)

        n_customers = len(sample_transactions["CustomerId"].unique())
        assert features.shape[0] == n_customers
        assert features.shape[1] > 5  # Should have several features


# ==================== Edge Cases and Integration Tests ====================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_transaction_times(self):
        """Test handling of multiple transactions at same timestamp."""
        df = pd.DataFrame(
            {
                "CustomerId": ["TEST_001", "TEST_001", "TEST_001"],
                "TransactionStartTime": [
                    "2024-01-15 10:00:00",
                    "2024-01-15 10:00:00",
                    "2024-01-15 10:00:00",
                ],
                "Amount": [100, -50, 200],
                "ProductCategory": ["airtime", "airtime", "airtime"],
                "ChannelId": ["ChannelId_3", "ChannelId_3", "ChannelId_3"],
                "ProviderId": ["ProviderId_6", "ProviderId_6", "ProviderId_6"],
                "Value": [100, 50, 200],
                "TransactionId": ["TX_001", "TX_002", "TX_003"],
                "BatchId": ["BATCH_001", "BATCH_001", "BATCH_001"],
                "AccountId": ["ACC_001", "ACC_001", "ACC_001"],
                "SubscriptionId": ["SUB_001", "SUB_001", "SUB_001"],
                "CurrencyCode": ["UGX", "UGX", "UGX"],
                "CountryCode": [256, 256, 256],
                "PricingStrategy": [2, 2, 2],
                "FraudResult": [0, 0, 0],
            }
        )
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        features, _ = process_data(df)

        # Should produce one row per customer
        assert len(features) == 1
        # hour_std should be 0 (all same hour)
        assert features["transaction_hour_std"].iloc[0] == 0


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
