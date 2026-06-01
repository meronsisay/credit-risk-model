"""
Feature engineering pipeline for credit risk modeling.
Transforms raw transaction data into model-ready customer-level features.
Includes proxy target variable engineering via RFM clustering and manual WoE/IV.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
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

        time_features = (
            df.groupby("CustomerId")
            .agg(
                {
                    "is_weekend": "mean",
                    "is_business_hour": "mean",
                    "hour": "std",
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


class HighRiskTargetCreator(BaseEstimator, TransformerMixin):
    """Creates proxy target variable using K-Means clustering on RFM features."""

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.internal_scaler = StandardScaler()
        self.kmeans = None
        self.high_risk_cluster = None
        self.cluster_profiles_ = None

    def fit(self, X, y=None):
        # X should have recency, frequency, monetary columns
        rfm = X[["recency", "frequency", "monetary"]].copy()

        rfm["frequency_log"] = np.log1p(rfm["frequency"])
        rfm["monetary_log"] = np.log1p(rfm["monetary"])

        features_to_scale = ["recency", "frequency_log", "monetary_log"]
        X_scaled = self.internal_scaler.fit_transform(rfm[features_to_scale])

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.kmeans.fit(X_scaled)

        cluster_labels = self.kmeans.labels_
        profiles = []
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            profiles.append(
                {
                    "cluster": i,
                    "size": mask.sum(),
                    "recency_mean": rfm.loc[mask, "recency"].mean(),
                    "frequency_mean": rfm.loc[mask, "frequency"].mean(),
                    "monetary_mean": rfm.loc[mask, "monetary"].mean(),
                }
            )

        profiles_df = pd.DataFrame(profiles)

        self.high_risk_cluster = int(
            profiles_df.loc[profiles_df["recency_mean"].idxmax(), "cluster"]
        )
        self.cluster_profiles_ = profiles_df

        print("\n--- Target Variable Cluster Breakdown ---")
        print(
            profiles_df[
                ["cluster", "size", "recency_mean", "frequency_mean", "monetary_mean"]
            ].to_string(index=False)
        )

        # Fixed: Break long line into multiple lines (E501 fix)
        high_risk_recency = profiles_df.loc[
            profiles_df["cluster"] == self.high_risk_cluster, "recency_mean"
        ].values[0]
        print(
            f"High-Risk Cluster: {self.high_risk_cluster} "
            f"(Highest recency: {high_risk_recency:.1f} days)"
        )
        return self

    def transform(self, X, y=None):
        if self.kmeans is None:
            raise RuntimeError(
                "Target engine state uninitialized. Run fit stage first."
            )

        rfm = X[["recency", "frequency", "monetary"]].copy()
        rfm["frequency_log"] = np.log1p(rfm["frequency"])
        rfm["monetary_log"] = np.log1p(rfm["monetary"])

        X_scaled = self.internal_scaler.transform(
            rfm[["recency", "frequency_log", "monetary_log"]]
        )
        predicted_clusters = self.kmeans.predict(X_scaled)

        output = pd.DataFrame(index=X.index)

        if "CustomerId" in X.columns:
            output["CustomerId"] = X["CustomerId"]
        else:
            output["CustomerId"] = X.index.astype(str)

        output["is_high_risk"] = (predicted_clusters == self.high_risk_cluster).astype(
            int
        )
        return output


class WoEIVCalculator:
    """Custom stateful Weight of Evidence and Information Value engine."""

    def __init__(self, target_col="is_high_risk"):
        self.target_col = target_col
        self.woe_maps = {}
        self.bin_edges = {}
        self.iv_scores = {}
        self.summary_df = None

    def _get_bin_labels(self, df_col, feature_name):
        """Helper to create distinct interval groups for continuous data."""
        if df_col.dtype in ["int64", "float64"] and df_col.nunique() > 10:
            try:
                _, edges = pd.qcut(df_col, q=10, retbins=True, duplicates="drop")
            except ValueError:
                _, edges = pd.qcut(df_col, q=5, retbins=True, duplicates="drop")
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges[feature_name] = edges
            return pd.cut(df_col, bins=edges).astype(str)
        else:
            self.bin_edges[feature_name] = None
            return df_col.astype(str)

    def calculate_woe_iv(self, df, feature, binned_series):
        """Calculate WoE and IV values on an explicitly grouped structural column."""
        temp_df = pd.DataFrame({"group": binned_series, "target": df[self.target_col]})

        grouped = temp_df.groupby("group")["target"].agg(["count", "sum"])
        grouped.columns = ["total", "events"]
        grouped["non_events"] = grouped["total"] - grouped["events"]

        total_events = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()

        epsilon = 0.5

        grouped["event_pct"] = (grouped["events"] + epsilon) / (total_events + epsilon)
        grouped["non_event_pct"] = (grouped["non_events"] + epsilon) / (
            total_non_events + epsilon
        )

        grouped["woe"] = np.log(grouped["event_pct"] / grouped["non_event_pct"])
        grouped["woe"] = grouped["woe"].clip(-5, 5)

        grouped["iv_component"] = (
            grouped["event_pct"] - grouped["non_event_pct"]
        ) * grouped["woe"]
        iv = grouped["iv_component"].sum()

        return grouped["woe"].to_dict(), abs(iv)

    def fit(self, df):
        exclude_cols = [self.target_col, "CustomerId"]
        features = [col for col in df.columns if col not in exclude_cols]

        print("\n" + "=" * 70)
        print("WEIGHT OF EVIDENCE (WoE) & INFORMATION VALUE (IV) ANALYSIS")
        print("=" * 70)

        results = []
        for feature in features:
            try:
                binned_series = self._get_bin_labels(df[feature], feature)
                woe_map, iv = self.calculate_woe_iv(df, feature, binned_series)

                self.woe_maps[feature] = woe_map
                self.iv_scores[feature] = iv

                if iv > 0.3:
                    strength = "STRONG"
                elif iv > 0.1:
                    strength = "MEDIUM"
                elif iv > 0.02:
                    strength = "WEAK"
                else:
                    strength = "USELESS"

                results.append(
                    {
                        "Feature": feature,
                        "IV": round(iv, 4),
                        "Predictive Power": strength,
                        "Categories": len(woe_map),
                    }
                )
                print(f"  {feature:25s} | IV = {iv:.4f} | {strength}")

            except Exception as e:
                print(f"  {feature:25s} | ERROR: {str(e)[:50]}")

        print("=" * 70)
        self.summary_df = pd.DataFrame(results).sort_values("IV", ascending=False)

        print("\n--- FEATURE SELECTION RECOMMENDATION (IV > 0.02) ---")
        keep_features = self.summary_df[self.summary_df["IV"] > 0.02]
        drop_features = self.summary_df[self.summary_df["IV"] <= 0.02]

        print(
            f"Features to KEEP ({len(keep_features)}): "
            f"{', '.join(keep_features['Feature'].tolist())}"
        )
        print(
            f"Features to DROP ({len(drop_features)}): "
            f"{', '.join(drop_features['Feature'].tolist())}"
        )

        return self

    def transform(self, df):
        df_woe = pd.DataFrame(index=df.index)

        for feature, woe_map in self.woe_maps.items():
            if feature in df.columns:
                edges = self.bin_edges.get(feature)
                if edges is not None:
                    binned = pd.cut(df[feature], bins=edges).astype(str)
                    df_woe[feature + "_woe"] = binned.map(woe_map).fillna(0)
                else:
                    df_woe[feature + "_woe"] = (
                        df[feature].astype(str).map(woe_map).fillna(0)
                    )

        return df_woe

    def get_feature_importance(self):
        return self.summary_df


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Complete Feature Engineering Pipeline with WoE/IV."""

    def __init__(
        self, create_target=False, apply_woe=False, n_clusters=3, random_state=42
    ):
        self.create_target = create_target
        self.apply_woe = apply_woe
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.customer_agg = CustomerAggregator()
        self.time_extractor = TimeFeatureExtractor()
        self.cat_extractor = CategoricalFeatureExtractor()

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.categorical_cols = ["ProductCategory", "ChannelId", "ProviderId"]

        self.numerical_cols = [
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

        self.is_fitted = False
        self.target_creator = None
        self.woe_calculator = None

    def fit(self, df, y=None):
        df_clean = df.drop(columns=["Value"], errors="ignore")

        self.customer_agg.fit(df_clean)
        self.time_extractor.fit(df_clean)
        self.cat_extractor.fit(df_clean)

        c_feat = self.customer_agg.transform(df_clean)
        t_feat = self.time_extractor.transform(df_clean)
        cat_feat = self.cat_extractor.transform(df_clean)

        features = c_feat.merge(t_feat, on="CustomerId", how="left")
        raw_features = features.merge(cat_feat, on="CustomerId", how="left")

        # Create target using ONLY RFM features
        if self.create_target:
            self.target_creator = HighRiskTargetCreator(
                n_clusters=self.n_clusters, random_state=self.random_state
            )
            # Pass the full dataframe with CustomerId
            rfm_data_with_id = raw_features[
                ["CustomerId", "recency", "frequency", "monetary"]
            ].copy()
            self.target_creator.fit(rfm_data_with_id)
            target_df = self.target_creator.transform(rfm_data_with_id)
            raw_features = raw_features.merge(target_df, on="CustomerId", how="left")

        # Fit preprocessing
        self.encoder.fit(raw_features[self.categorical_cols])
        self.imputer.fit(raw_features[self.numerical_cols])
        self.scaler.fit(raw_features[self.numerical_cols])

        # WoE analysis on NON-leakage features only
        if self.apply_woe and self.create_target and self.target_creator is not None:
            self.woe_calculator = WoEIVCalculator(target_col="is_high_risk")
            # Use only non-leakage features for WoE
            woe_features_df = raw_features[
                self.numerical_cols + self.categorical_cols
            ].copy()
            woe_features_df["is_high_risk"] = raw_features["is_high_risk"]
            self.woe_calculator.fit(woe_features_df)

        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")

        df_clean = df.drop(columns=["Value"], errors="ignore")

        c_feat = self.customer_agg.transform(df_clean)
        t_feat = self.time_extractor.transform(df_clean)
        cat_feat = self.cat_extractor.transform(df_clean)

        features = c_feat.merge(t_feat, on="CustomerId", how="left")
        raw_features = features.merge(cat_feat, on="CustomerId", how="left")

        # Add target if needed
        if self.create_target and self.target_creator is not None:
            rfm_data_with_id = raw_features[
                ["CustomerId", "recency", "frequency", "monetary"]
            ].copy()
            target_df = self.target_creator.transform(rfm_data_with_id)
            raw_features = raw_features.merge(target_df, on="CustomerId", how="left")

        # Process features
        processed_df = raw_features.copy()
        processed_df[self.numerical_cols] = self.imputer.transform(
            processed_df[self.numerical_cols]
        )
        processed_df[self.numerical_cols] = self.scaler.transform(
            processed_df[self.numerical_cols]
        )

        # Encode categorical variables
        encoded_array = self.encoder.transform(processed_df[self.categorical_cols])
        encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)

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

            if "provider_provider_" in clean_name:
                clean_name = clean_name.replace("provider_provider_", "provider_")
            if "channel_channel_" in clean_name:
                clean_name = clean_name.replace("channel_channel_", "channel_")

            encoded_cols_clean.append(clean_name)

        encoded_outputs = pd.DataFrame(
            encoded_array, columns=encoded_cols_clean, index=processed_df.index
        )
        final_features = processed_df.drop(columns=self.categorical_cols)
        final_features = pd.concat([final_features, encoded_outputs], axis=1)

        # Apply WoE transformation
        if self.apply_woe and self.woe_calculator is not None:
            woe_input = raw_features[self.numerical_cols + self.categorical_cols].copy()
            woe_features = self.woe_calculator.transform(woe_input)
            final_features = pd.concat([final_features, woe_features], axis=1)

        return final_features

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)

    def get_iv_scores(self):
        if self.woe_calculator is not None:
            return self.woe_calculator.get_feature_importance()
        return None


def process_data(
    df, create_target=False, apply_woe=False, n_clusters=3, random_state=42
):
    pipeline = FeaturePipeline(
        create_target=create_target,
        apply_woe=apply_woe,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    features = pipeline.fit_transform(df)
    return features, pipeline


def main():
    """Main execution function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "data.csv")

    print("=" * 60)
    print("PROXY TARGET WITH WoE/IV ANALYSIS")
    print("=" * 60)

    print("\nLoading dataset records...")
    df = pd.read_csv(data_path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    print("\nExecuting data transforms...")
    features, pipeline = process_data(
        df, create_target=True, apply_woe=True, n_clusters=3, random_state=42
    )

    print(
        f"\nFinal Matrix: {features.shape[0]:,} customers, {features.shape[1]} features."
    )

    if "is_high_risk" in features.columns:
        counts = features["is_high_risk"].value_counts()
        pct = features["is_high_risk"].value_counts(normalize=True)
        print("\nTarget Distribution:")
        print(f"  Low risk (0):  {counts.get(0, 0):,} ({pct.get(0, 0)*100:.2f}%)")
        print(f"  High risk (1): {counts.get(1, 0):,} ({pct.get(1, 0)*100:.2f}%)")

    processed_dir = os.path.join(script_dir, "..", "data", "processed")
    model_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "processed_data.csv")
    features.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    pipeline_path = os.path.join(model_dir, "feature_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved: {pipeline_path}")


if __name__ == "__main__":
    main()
