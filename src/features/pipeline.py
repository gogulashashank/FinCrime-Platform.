import polars as pl
import os

class FeaturePipeline:
    def __init__(self, data_dir='data/synthetic'):
        self.data_dir = data_dir
        
    def load_data(self):
        print("Loading datasets...")
        self.transactions = pl.read_parquet(os.path.join(self.data_dir, 'fact_transactions.parquet'))
        self.accounts = pl.read_parquet(os.path.join(self.data_dir, 'dim_accounts.parquet'))
        self.entities = pl.read_parquet(os.path.join(self.data_dir, 'dim_entities.parquet'))
        
    def engineer_features(self):
        print("Engineering features via Polars...")
        
        # 1. Transaction level features
        txns = self.transactions.with_columns([
            (pl.col("amount") % 1000 == 0).cast(pl.Int32).alias("is_round_amount")
        ])
        
        # We need sender features. We group by sender to calculate 1d windows.
        # For a production system we'd use rolling_sum, but for a portfolio batch, we'll approximate 
        # by self-joining or computing daily stats. Let's do a simple 1-day lookback using rolling groupings if possible,
        # or just daily aggregations.
        
        # Sort by timestamp for rolling
        txns = txns.sort("timestamp")
        
        # Compute Sender Rolling Features (1 Day window)
        sender_rolling = txns.group_by_dynamic(
            index_column="timestamp",
            every="1d",
            by="sender_account_id",
            closed="right"
        ).agg([
            pl.col("amount").count().alias("txn_count_1d"),
            pl.col("amount").sum().alias("total_amount_1d"),
            pl.col("amount").mean().alias("avg_amount_1d")
        ])

        sender_rolling_7d = txns.group_by_dynamic(
            index_column="timestamp",
            every="7d",
            by="sender_account_id",
            closed="right"
        ).agg([
            pl.col("amount").count().alias("txn_count_7d"),
            pl.col("amount").sum().alias("total_amount_7d")
        ])

        sender_rolling_30d = txns.group_by_dynamic(
            index_column="timestamp",
            every="30d",
            by="sender_account_id",
            closed="right"
        ).agg([
            pl.col("amount").count().alias("txn_count_30d"),
            pl.col("amount").sum().alias("total_amount_30d")
        ])
        
        # Join rolling features back to transactions
        # We join on timestamp (cast to date) for approximation or do an asof join.
        # Since group_by_dynamic creates a 'timestamp' column that is truncated to the 'every' interval,
        # we can join back. To be precise, let's just do an asof join.
        txns_with_sender = txns.join_asof(
            sender_rolling.sort("timestamp"),
            by="sender_account_id",
            on="timestamp",
            strategy="backward"
        ).join_asof(
            sender_rolling_7d.sort("timestamp"),
            by="sender_account_id",
            on="timestamp",
            strategy="backward"
        ).join_asof(
            sender_rolling_30d.sort("timestamp"),
            by="sender_account_id",
            on="timestamp",
            strategy="backward"
        )
        
        # Compute Receiver Rolling Features (to find velocity ratio)
        receiver_rolling = txns.group_by_dynamic(
            index_column="timestamp",
            every="1d",
            by="receiver_account_id",
            closed="right"
        ).agg([
            pl.col("amount").sum().alias("total_received_1d")
        ])
        
        # Velocity ratio: Money out (total_amount_1d) / Money in (total_received_1d from previous days)
        # We will join receiver_rolling to txns treating sender_account_id = receiver_account_id
        receiver_rolling = receiver_rolling.rename({"receiver_account_id": "sender_account_id"})
        
        base_table = txns_with_sender.join_asof(
            receiver_rolling.sort("timestamp"),
            by="sender_account_id",
            on="timestamp",
            strategy="backward"
        )
        
        base_table = base_table.with_columns(
            (pl.col("total_amount_1d") / (pl.col("total_received_1d") + 1)).alias("velocity_in_out_ratio")
        )
        
        # Join Entity Risk Information
        # Account -> Entity -> Risk
        acct_entity = self.accounts.join(self.entities, on="entity_id", how="left")
        base_table = base_table.join(
            acct_entity.select(["account_id", "risk_rating", "jurisdiction", "is_pep", "onboarding_date", "last_review_date"]),
            left_on="sender_account_id",
            right_on="account_id",
            how="left"
        )
        
        # Fill nulls
        base_table = base_table.fill_null(0)
        
        self.base_table = base_table
        return base_table
        
    def save_features(self):
        output_path = os.path.join(self.data_dir, 'analytical_base_table.parquet')
        self.base_table.write_parquet(output_path)
        print(f"Saved Analytical Base Table to {output_path}")
