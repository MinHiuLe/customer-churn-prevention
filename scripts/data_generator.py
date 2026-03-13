import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ─── CAUSAL RULES ────────────────────────────────────────────────────────────
# Đây là "ground truth" ẩn mà Uplift Model cần tìm ra
#
# Persuadable:  churn_prob cao, NHƯNG nếu nhận voucher → giảm 40%
# Lost Cause:   churn_prob cao, dù có voucher → vẫn churn
# Sure Thing:   churn_prob thấp, voucher không ảnh hưởng
# Sleeping Dog: churn_prob thấp, NHƯNG nếu bị contact → TĂNG 30%
# ─────────────────────────────────────────────────────────────────────────────

SEGMENTS = ['Persuadable', 'Lost_Cause', 'Sure_Thing', 'Sleeping_Dog']
SEGMENT_WEIGHTS = [0.25, 0.20, 0.35, 0.20]  # Distribution thực tế

def assign_base_churn_prob(segment: str) -> float:
    """Xác suất churn tự nhiên theo segment"""
    probs = {
        'Persuadable':  np.random.beta(7, 3),   # High: 0.6-0.9
        'Lost_Cause':   np.random.beta(9, 2),   # Very high: 0.8-0.99
        'Sure_Thing':   np.random.beta(2, 8),   # Low: 0.1-0.3
        'Sleeping_Dog': np.random.beta(2, 7),   # Low: 0.1-0.35
    }
    return probs[segment]

def apply_treatment_effect(base_prob: float, segment: str, treated: bool) -> float:
    """
    Core causal logic: voucher ảnh hưởng thế nào đến từng segment
    Đây là thứ T-Learner cần học được
    """
    if not treated:
        return base_prob

    effects = {
        'Persuadable':  -0.40,   # Giảm 40% → đây là người cần target
        'Lost_Cause':   -0.02,   # Gần như không đổi → lãng phí voucher
        'Sure_Thing':   -0.05,   # Giảm nhẹ → không cần thiết
        'Sleeping_Dog': +0.30,   # TĂNG 30% → tuyệt đối không contact!
    }
    new_prob = base_prob + effects[segment]
    return np.clip(new_prob, 0.01, 0.99)

def generate_user_features(segment: str) -> dict:
    """Sinh features có correlation với segment — không phải random hoàn toàn"""

    # Persuadable: mới dùng, phí cao, hợp đồng tháng
    if segment == 'Persuadable':
        tenure = np.random.randint(1, 18)
        monthly_charges = np.random.uniform(65, 110)
        contract = np.random.choice([0, 1], p=[0.80, 0.20])
        service_count = np.random.randint(2, 6)
        clv = monthly_charges * tenure * np.random.uniform(0.8, 1.2)

    # Lost Cause: không hài lòng sâu, không có gì giữ lại
    elif segment == 'Lost_Cause':
        tenure = np.random.randint(1, 12)
        monthly_charges = np.random.uniform(75, 115)
        contract = 0  # Luôn month-to-month
        service_count = np.random.randint(1, 4)
        clv = monthly_charges * tenure * np.random.uniform(0.6, 0.9)

    # Sure Thing: dùng lâu, hợp đồng dài, nhiều services
    elif segment == 'Sure_Thing':
        tenure = np.random.randint(24, 72)
        monthly_charges = np.random.uniform(45, 85)
        contract = np.random.choice([1, 2], p=[0.40, 0.60])
        service_count = np.random.randint(4, 9)
        clv = monthly_charges * tenure * np.random.uniform(1.0, 1.3)

    # Sleeping Dog: ổn định, ít services, không muốn bị làm phiền
    else:
        tenure = np.random.randint(12, 48)
        monthly_charges = np.random.uniform(30, 65)
        contract = np.random.choice([0, 1, 2], p=[0.30, 0.40, 0.30])
        service_count = np.random.randint(1, 4)
        clv = monthly_charges * tenure * np.random.uniform(0.9, 1.1)

    return {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'contract_type': contract,
        'service_count': service_count,
        'clv_proxy': round(clv, 2),
        'senior_citizen': np.random.choice([0, 1], p=[0.85, 0.15]),
        'has_partner': np.random.choice([0, 1]),
        'digital_engagement': np.random.randint(0, 3),
    }

def generate_dataset(n_users: int = 5000, treatment_rate: float = 0.5) -> pd.DataFrame:
    """
    Sinh dataset với randomized treatment (A/B test setup)
    treatment_rate: tỉ lệ users được tặng voucher
    """
    records = []

    for i in range(n_users):
        # 1. Assign segment
        segment = np.random.choice(SEGMENTS, p=SEGMENT_WEIGHTS)

        # 2. Random treatment assignment (giả lập A/B test)
        treated = np.random.random() < treatment_rate

        # 3. Sinh features theo segment
        features = generate_user_features(segment)

        # 4. Tính churn probability với causal effect
        base_prob = assign_base_churn_prob(segment)
        final_prob = apply_treatment_effect(base_prob, segment, treated)

        # 5. Simulate actual outcome
        churned = int(np.random.random() < final_prob)

        records.append({
            'user_id': f'U{i:05d}',
            **features,
            'segment_true': segment,       # Ground truth — chỉ dùng để validate
            'treated': int(treated),        # 1 = nhận voucher, 0 = không
            'base_churn_prob': round(base_prob, 4),
            'final_churn_prob': round(final_prob, 4),
            'churned': churned,
        })

    df = pd.DataFrame(records)
    return df

def generate_drift_dataset(n_users: int = 2000, week: int = 1) -> pd.DataFrame:
    """
    Sinh data với concept drift từ tuần 5 trở đi
    Young users (tenure < 12) giảm engagement 60%
    → Evidently sẽ detect drift này và trigger retrain
    """
    df = generate_dataset(n_users, treatment_rate=0.5)

    if week >= 5:
        # Concept drift: young users đột ngột dùng ít services hơn
        young_mask = df['tenure'] < 12
        drift_factor = 0.4  # Giảm 60%

        df.loc[young_mask, 'service_count'] = (
            df.loc[young_mask, 'service_count'] * drift_factor
        ).astype(int).clip(lower=1)

        df.loc[young_mask, 'digital_engagement'] = (
            df.loc[young_mask, 'digital_engagement'] * drift_factor
        ).astype(int)

        # Churn prob của nhóm này tăng theo
        df.loc[young_mask, 'final_churn_prob'] = (
            df.loc[young_mask, 'final_churn_prob'] * 1.35
        ).clip(upper=0.99)

        df.loc[young_mask, 'churned'] = df.loc[young_mask].apply(
            lambda r: int(np.random.random() < r['final_churn_prob']), axis=1
        )

        print(f"Week {week}: Concept drift applied to {young_mask.sum()} young users ⚠️")
    
    df['week'] = week
    return df

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Sinh dataset chính cho training
    print("Generating training dataset...")
    df_train = generate_dataset(n_users=5000)
    df_train.to_csv('data/processed/synthetic_uplift_data.csv', index=False)

    # Sinh data theo tuần để demo drift
    print("Generating weekly datasets for drift demo...")
    all_weeks = []
    for week in range(1, 9):
        df_week = generate_drift_dataset(n_users=500, week=week)
        all_weeks.append(df_week)

    df_weekly = pd.concat(all_weeks, ignore_index=True)
    df_weekly.to_csv('data/processed/weekly_drift_data.csv', index=False)

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n=== DATASET VALIDATION ===")
    print(f"Training set shape: {df_train.shape}")
    print(f"\nSegment distribution:\n{df_train['segment_true'].value_counts(normalize=True).round(3)}")
    print(f"\nChurn rate overall: {df_train['churned'].mean():.3f}")
    print(f"\nChurn rate by segment:")
    print(df_train.groupby('segment_true')['churned'].mean().round(3))
    print(f"\nTreatment effect by segment (key validation):")
    effect = df_train.groupby(['segment_true', 'treated'])['churned'].mean().unstack()
    effect['uplift'] = effect[0] - effect[1]
    print(effect.round(3))
    print("\nData Generator done ✅")

