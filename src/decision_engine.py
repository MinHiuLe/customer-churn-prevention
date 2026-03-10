import pandas as pd
import numpy as np
import psycopg2
import os
from datetime import datetime

class DecisionEngine:
    ACTION_MAP = {
        'Persuadable': {
            'VIP':     {'action': 'Voucher 20% + CS Call', 'cost': 50},
            'Regular': {'action': 'Email Voucher 10%',     'cost': 10},
        },
        'Sleeping_Dog': {'action': 'SUPPRESS',    'cost': 0},
        'Lost_Cause':   {'action': 'Escalate CS', 'cost': 30},
        'Sure_Thing':   {'action': 'No Action',   'cost': 0},
    }

    def __init__(self, monthly_budget: float = 5000):
        self.monthly_budget = monthly_budget

    def load_scores(self, path: str = 'data/processed/batch_scores_latest.csv'):
        return pd.read_csv(path)

    def assign_action(self, row):
        seg = row['segment']
        is_vip = row.get('clv_proxy', 0) > row.get('clv_threshold', 500)
        if seg == 'Persuadable':
            tier = 'VIP' if is_vip else 'Regular'
            return self.ACTION_MAP['Persuadable'][tier]
        elif seg in self.ACTION_MAP:
            return self.ACTION_MAP[seg]
        return {'action': 'No Action', 'cost': 0}

    def run(self, df: pd.DataFrame = None):
        if df is None:
            df = self.load_scores()

        clv_threshold = df['clv_proxy'].quantile(0.75)
        df['clv_threshold'] = clv_threshold
        df['is_vip'] = df['clv_proxy'] > clv_threshold

        actions = df.apply(self.assign_action, axis=1)
        df['recommended_action'] = [a['action'] for a in actions]
        df['action_cost'] = [a['cost'] for a in actions]

        df['expected_clv_saved'] = (
            df['churn_probability'] *
            df['clv_proxy'] *
            df['uplift_score'].clip(lower=0)
        ).round(2)

        df['roi'] = np.where(
            df['action_cost'] > 0,
            df['expected_clv_saved'] / df['action_cost'],
            0
        ).round(2)

        actionable = df[
            (df['segment'] == 'Persuadable') &
            (df['churn_probability'] >= df['churn_probability'].quantile(0.70))
        ].copy().sort_values('roi', ascending=False)

        actionable['cumulative_cost'] = actionable['action_cost'].cumsum()
        within_budget = actionable[actionable['cumulative_cost'] <= self.monthly_budget]
        df['targeted'] = df['user_id'].isin(within_budget['user_id'])

        return df, within_budget

    def print_report(self, df, within_budget):
        print("=" * 60)
        print(f"DECISION ENGINE REPORT — Budget: ${self.monthly_budget:,.0f}/month")
        print("=" * 60)
        print(f"\n📊 SEGMENT DISTRIBUTION:")
        print(df['segment'].value_counts().to_string())
        print(f"\n🎯 TARGETED USERS:      {len(within_budget)}")
        print(f"💰 TOTAL COST:          ${within_budget['action_cost'].sum():,.0f}")
        print(f"📈 EXPECTED CLV SAVED:  ${within_budget['expected_clv_saved'].sum():,.0f}")
        print(f"🔥 AVG ROI:             {within_budget['roi'].mean():.2f}x")
        print(f"\n📋 ACTION BREAKDOWN:")
        print(within_budget['recommended_action'].value_counts().to_string())
        print(f"\n⚠️  SLEEPING DOGS SUPPRESSED: {(df['segment'] == 'Sleeping_Dog').sum()}")
        print("=" * 60)

    def log_actions(self, within_budget):
        try:
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', 5432),
                dbname=os.getenv('POSTGRES_DB', 'churn_db'),
                user=os.getenv('POSTGRES_USER', 'churn_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'churn_pass'),
            )
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS action_logs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50),
                    segment VARCHAR(50),
                    recommended_action VARCHAR(100),
                    action_cost FLOAT,
                    expected_clv_saved FLOAT,
                    roi FLOAT,
                    churn_probability FLOAT,
                    executed_at TIMESTAMP DEFAULT NOW()
                )
            """)
            for _, row in within_budget.iterrows():
                cur.execute("""
                    INSERT INTO action_logs
                        (user_id, segment, recommended_action, action_cost,
                         expected_clv_saved, roi, churn_probability)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (row['user_id'], row['segment'], row['recommended_action'],
                      row['action_cost'], row['expected_clv_saved'],
                      row['roi'], row['churn_probability']))
            conn.commit()
            cur.close()
            conn.close()
            print(f"Action logs saved to PostgreSQL ✅ — {len(within_budget)} actions")
        except Exception as e:
            print(f"PostgreSQL logging skipped: {e}")

if __name__ == '__main__':
    engine = DecisionEngine(monthly_budget=5000)
    df = engine.load_scores()
    print(f"Loaded {len(df)} users ✅")
    df_result, within_budget = engine.run(df)
    engine.print_report(df_result, within_budget)
    df_result.to_csv('data/processed/action_plan.csv', index=False)
    within_budget.to_csv('data/processed/targeted_users.csv', index=False)
    print("\nAction plan saved ✅")
    engine.log_actions(within_budget)
