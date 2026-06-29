import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class FinCrimeDataGenerator:
    def __init__(self, num_entities=1000, num_transactions=50000, seed=42):
        self.num_entities = num_entities
        self.num_transactions = num_transactions
        self.seed = seed
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker(['en_GB', 'en_US'])
        
        # Risk Configuration
        self.high_risk_jurisdictions = ['UAE', 'Panama', 'Cayman Islands', 'Malta', 'Cyprus']
        self.low_risk_jurisdictions = ['UK', 'US', 'Germany', 'France', 'Canada']

    def generate_entities(self):
        """Generate dim_entities"""
        entities = []
        for _ in range(self.num_entities):
            is_company = random.random() < 0.2
            
            # Inject Shell Company Typology (usually younger companies, high risk jurisdiction)
            is_shell_candidate = is_company and random.random() < 0.1
            
            jurisdiction = random.choice(self.high_risk_jurisdictions) if is_shell_candidate else random.choice(self.low_risk_jurisdictions + self.high_risk_jurisdictions)
            
            onboarding_date = self.fake.date_between(start_date='-5y', end_date='-1m')
            
            entities.append({
                'entity_id': self.fake.uuid4(),
                'entity_type': 'Company' if is_company else 'Individual',
                'name': self.fake.company() if is_company else self.fake.name(),
                'jurisdiction': jurisdiction,
                'is_pep': random.random() < 0.05 if not is_company else False,
                'onboarding_date': onboarding_date,
                'last_review_date': self.fake.date_between(start_date=onboarding_date, end_date='today'),
                'expected_monthly_volume': round(random.uniform(1000, 100000), 2),
                'risk_rating': 'High' if is_shell_candidate else random.choice(['Low', 'Medium', 'High'])
            })
        
        self.entities_df = pd.DataFrame(entities)
        return self.entities_df

    def generate_accounts(self):
        """Generate dim_accounts based on entities"""
        if not hasattr(self, 'entities_df'):
            self.generate_entities()
            
        accounts = []
        for _, entity in self.entities_df.iterrows():
            # Most entities have 1 account, some have multiple
            num_accounts = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
            for _ in range(num_accounts):
                # Inject dormant account typology
                is_dormant = random.random() < 0.05
                
                accounts.append({
                    'account_id': self.fake.iban(),
                    'entity_id': entity['entity_id'],
                    'account_type': random.choice(['Current', 'Savings', 'Business']),
                    'status': 'Dormant' if is_dormant else 'Active',
                    'balance': round(random.uniform(100, 50000), 2),
                    'open_date': self.fake.date_between(start_date=entity['onboarding_date'], end_date='today')
                })
        
        self.accounts_df = pd.DataFrame(accounts)
        return self.accounts_df

    def _generate_normal_transactions(self, n):
        """Generate baseline normal transactions"""
        account_ids = self.accounts_df['account_id'].tolist()
        txns = []
        for _ in range(n):
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            while sender == receiver:
                receiver = random.choice(account_ids)
                
            txns.append({
                'transaction_id': self.fake.uuid4(),
                'sender_account_id': sender,
                'receiver_account_id': receiver,
                'amount': round(random.uniform(10, 2000), 2),
                'currency': 'GBP',
                'timestamp': self.fake.date_time_between(start_date='-1y', end_date='now'),
                'is_suspicious': 0,
                'typology': 'None'
            })
        return txns

    def _inject_structuring(self, num_cases=50):
        """Inject structuring (smurfing) typology: many transactions just below reporting threshold ($10k)"""
        account_ids = self.accounts_df['account_id'].tolist()
        txns = []
        for _ in range(num_cases):
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            base_time = self.fake.date_time_between(start_date='-1y', end_date='now')
            
            # Generate 4-8 transactions of $9,000 - $9,999 within a few days
            num_txns = random.randint(4, 8)
            for i in range(num_txns):
                txns.append({
                    'transaction_id': self.fake.uuid4(),
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount': round(random.uniform(9000, 9900), 2),
                    'currency': 'GBP',
                    'timestamp': base_time + timedelta(hours=random.randint(1, 48) * i),
                    'is_suspicious': 1,
                    'typology': 'Structuring'
                })
        return txns
        
    def _inject_rapid_movement(self, num_cases=30):
        """Inject rapid movement of funds (layering)"""
        account_ids = self.accounts_df['account_id'].tolist()
        txns = []
        for _ in range(num_cases):
            acct_A = random.choice(account_ids)
            acct_B = random.choice(account_ids)
            acct_C = random.choice(account_ids)
            
            base_time = self.fake.date_time_between(start_date='-1y', end_date='now')
            amount = round(random.uniform(10000, 50000), 2)
            
            # A -> B
            txns.append({
                'transaction_id': self.fake.uuid4(),
                'sender_account_id': acct_A,
                'receiver_account_id': acct_B,
                'amount': amount,
                'currency': 'GBP',
                'timestamp': base_time,
                'is_suspicious': 1,
                'typology': 'Rapid Movement'
            })
            
            # B -> C (rapid transfer out within minutes/hours)
            txns.append({
                'transaction_id': self.fake.uuid4(),
                'sender_account_id': acct_B,
                'receiver_account_id': acct_C,
                'amount': amount * random.uniform(0.95, 1.0), # slightly less due to fees
                'currency': 'GBP',
                'timestamp': base_time + timedelta(minutes=random.randint(5, 120)),
                'is_suspicious': 1,
                'typology': 'Rapid Movement'
            })
        return txns

    def _inject_round_amounts(self, num_cases=100):
        """Inject round amount behavior often seen in human-trafficking or illicit cash flows"""
        account_ids = self.accounts_df['account_id'].tolist()
        txns = []
        for _ in range(num_cases):
            txns.append({
                'transaction_id': self.fake.uuid4(),
                'sender_account_id': random.choice(account_ids),
                'receiver_account_id': random.choice(account_ids),
                'amount': float(random.choice([1000, 5000, 10000, 20000, 50000])),
                'currency': 'GBP',
                'timestamp': self.fake.date_time_between(start_date='-1y', end_date='now'),
                'is_suspicious': 1,
                'typology': 'Round Amount'
            })
        return txns

    def generate_transactions(self):
        """Generate fact_transactions mixing normal and suspicious behaviors"""
        if not hasattr(self, 'accounts_df'):
            self.generate_accounts()
            
        print("Generating baseline transactions...")
        normal_txns = self._generate_normal_transactions(self.num_transactions)
        
        print("Injecting structuring typologies...")
        structuring_txns = self._inject_structuring(num_cases=int(self.num_transactions * 0.005))
        
        print("Injecting rapid movement typologies...")
        rapid_txns = self._inject_rapid_movement(num_cases=int(self.num_transactions * 0.002))
        
        print("Injecting round amount typologies...")
        round_txns = self._inject_round_amounts(num_cases=int(self.num_transactions * 0.01))
        
        all_txns = normal_txns + structuring_txns + rapid_txns + round_txns
        
        self.transactions_df = pd.DataFrame(all_txns)
        # Sort by timestamp
        self.transactions_df.sort_values(by='timestamp', inplace=True)
        self.transactions_df.reset_index(drop=True, inplace=True)
        
        return self.transactions_df

    def save_data(self, output_dir='data/synthetic'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(self.entities_df)} entities...")
        self.entities_df.to_parquet(f'{output_dir}/dim_entities.parquet', index=False)
        self.entities_df.head(100).to_csv(f'data/sample_seed/dim_entities_sample.csv', index=False)
        
        print(f"Saving {len(self.accounts_df)} accounts...")
        self.accounts_df.to_parquet(f'{output_dir}/dim_accounts.parquet', index=False)
        self.accounts_df.head(100).to_csv(f'data/sample_seed/dim_accounts_sample.csv', index=False)
        
        print(f"Saving {len(self.transactions_df)} transactions...")
        self.transactions_df.to_parquet(f'{output_dir}/fact_transactions.parquet', index=False)
        self.transactions_df.head(100).to_csv(f'data/sample_seed/fact_transactions_sample.csv', index=False)
        
        print("Data generation complete.")
