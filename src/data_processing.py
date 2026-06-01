"""
Feature engineering pipeline for credit risk modeling.
Transforms raw transaction data into model-ready customer-level features.
Corrected to eliminate invalid arithmetic means of circular time components.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import joblib


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregates transaction-level data to customer-level financial metrics."""

    def __init__(self):
        self.snapshot_date = None

    def fit(self, X, y=None):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        self.snapshot_date = df["TransactionStartTime"].max()
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        # Core Aggregations (RFM Foundations)
        last_transaction = df.groupby("CustomerId")["TransactionStartTime"].max()
        recency = (self.snapshot_date - last_transaction).dt.days
        frequency = df.groupby("CustomerId").size()
        monetary = df[df["Amount"] > 0].groupby("CustomerId")["Amount"].sum()
        avg_amount = df.groupby("CustomerId")["Amount"].mean()
        std_amount = df.groupby("CustomerId")["Amount"].std()

        # Refund Behavioral Risk Indicators
        refund_count = df[df["Amount"] < 0].groupby("CustomerId").size()
        refund_rate = refund_count / frequency
        negative_df = df[df["Amount"] < 0]
        refund_amount = (
            negative_df["Amount"].abs().groupby(negative_df["CustomerId"]).sum()
        )

        customer_features = pd.DataFrame(
            {
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
                "avg_amount": avg_amount,
                "std_amount": std_amount,
                "refund_rate": refund_rate,
                "refund_amount": refund_amount,
            }
        ).reset_index()

        return customer_features.fillna(0)


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts high-signal, mathematically sound behavioral time metrics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        df["hour"] = df["TransactionStartTime"].dt.hour
        df["day_of_week"] = df["TransactionStartTime"].dt.dayofweek

        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 17)).astype(int)

        # Aggregate using ratios and variances to capture behavioral stability
        time_features = (
            df.groupby("CustomerId")
            .agg(
                {
                    "is_weekend": "mean",  # Proportion of total activity on weekends
                    "is_business_hour": "mean",  # Proportion of total activity during business hours
                    "hour": "std",  # Stability/predictability of transaction time
                }
            )
            .reset_index()
        )

        time_features.columns = [
            "CustomerId",
            "weekend_ratio",
            "business_hour_ratio",
            "transaction_hour_std",
        ]

        # Single-transaction customers will have an hour std of NaN; fill with 0 (perfect stability)
        return time_features.fillna(0)


class CategoricalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts structural categorical attributes safely using modes and unique counts."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        categorical_cols = ["ProductCategory", "ChannelId", "ProviderId"]

        categorical_features = (
            df.groupby("CustomerId")[categorical_cols]
            .agg(lambda x: str(x.mode().iloc[0]) if not x.mode().empty else "unknown")
            .reset_index()
        )

        for col in categorical_cols:
            unique_counts = (
                df.groupby("CustomerId")[col].nunique().rename(f"unique_{col.lower()}")
            )
            categorical_features = categorical_features.merge(
                unique_counts, on="CustomerId"
            )

        return categorical_features


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Complete, stateful scikit-learn Feature Engineering Pipeline."""

    def __init__(self):
        self.customer_agg = CustomerAggregator()
        self.time_extractor = TimeFeatureExtractor()
        self.cat_extractor = CategoricalFeatureExtractor()

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.categorical_cols = ["ProductCategory", "ChannelId", "ProviderId"]
        self.numerical_cols = None
        self.is_fitted = False

    def fit(self, df, y=None):
        df_clean = df.drop(columns=["Value"], errors="ignore")

        self.customer_agg.fit(df_clean)
        self.time_extractor.fit(df_clean)
        self.cat_extractor.fit(df_clean)

        c_feat = self.customer_agg.transform(df_clean)
        t_feat = self.time_extractor.transform(df_clean)
        cat_feat = self.cat_extractor.transform(df_clean)

        features = c_feat.merge(t_feat, on="CustomerId", how="left")
        features = features.merge(cat_feat, on="CustomerId", how="left")

        self.encoder.fit(features[self.categorical_cols])

        self.numerical_cols = [
            "recency",
            "frequency",
            "monetary",
            "avg_amount",
            "std_amount",
            "refund_rate",
            "refund_amount",
            "weekend_ratio",
            "business_hour_ratio",
            "transaction_hour_std",
            "unique_productcategory",
            "unique_channelid",
            "unique_providerid",
        ]

        self.imputer.fit(features[self.numerical_cols])
        self.scaler.fit(features[self.numerical_cols])

        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before running transform phases."
            )

        df_clean = df.drop(columns=["Value"], errors="ignore")

        c_feat = self.customer_agg.transform(df_clean)
        t_feat = self.time_extractor.transform(df_clean)
        cat_feat = self.cat_extractor.transform(df_clean)

        features = c_feat.merge(t_feat, on="CustomerId", how="left")
        features = features.merge(cat_feat, on="CustomerId", how="left")

        features[self.numerical_cols] = self.imputer.transform(
            features[self.numerical_cols]
        )
        features[self.numerical_cols] = self.scaler.transform(
            features[self.numerical_cols]
        )

        # One-hot encode categoricals
        encoded_array = self.encoder.transform(features[self.categorical_cols])
        encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)

        # Clear, predictable naming translation block
        encoded_cols_clean = []
        for col in encoded_cols:
            col_lower = col.lower()
            if "productcategory_" in col_lower:
                clean_name = col_lower.replace("productcategory_", "cat_")
            elif "channelid_" in col_lower:
                clean_name = col_lower.replace("channelid_", "channel_")
            elif "providerid_" in col_lower:
                clean_name = col_lower.replace("providerid_", "provider_")
            else:
                clean_name = col_lower

            # Resolve potential internal duplicates (e.g. channel_channel_3 -> channel_3)
            if "provider_provider_" in clean_name:
                clean_name = clean_name.replace("provider_provider_", "provider_")
            if "channel_channel_" in clean_name:
                clean_name = clean_name.replace("channel_channel_", "channel_")

            encoded_cols_clean.append(clean_name)

        encoded_df = pd.DataFrame(
            encoded_array, columns=encoded_cols_clean, index=features.index
        )

        final_features = features.drop(columns=self.categorical_cols)
        final_features = pd.concat([final_features, encoded_df], axis=1)

        return final_features

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)


def process_data(df):
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(df)
    return features, pipeline


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "data.csv")

    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    print("\nLoading data...")
    df = pd.read_csv(data_path)

    print("\nRunning feature engineering pipeline...")
    features, pipeline = process_data(df)

    print(f"\n Pipeline complete: {features.shape[0]} unique customer vectors created.")

    processed_dir = os.path.join(script_dir, "..", "data", "processed")
    model_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "processed_data.csv")
    features.to_csv(output_path, index=False)
    print(f" Saved model-ready data: {output_path}")

    pipeline_path = os.path.join(model_dir, "feature_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f" Saved fitted pipeline: {pipeline_path}")
