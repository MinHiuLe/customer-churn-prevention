import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='localhost', port=5432,
    dbname=os.getenv('POSTGRES_DB', 'churn_db'),
    user=os.getenv('POSTGRES_USER', 'churn_user'),
    password=os.getenv('POSTGRES_PASSWORD', 'churn_pass'),
)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        senior_citizen INT, partner INT, dependents INT,
        tenure FLOAT, phone_service INT, multiple_lines INT,
        internet_service INT, online_security INT, online_backup INT,
        device_protection INT, tech_support INT, streaming_tv INT,
        streaming_movies INT, contract_encoded INT, paperless_billing INT,
        payment_method_encoded INT, monthly_charges FLOAT,
        total_charges FLOAT, churn_binary INT,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
conn.commit()

df = pd.read_csv('data/processed/features.csv')
inserted = 0

for idx, row in df.iterrows():
    cid = row.get('customerID', f'CUST_{idx:05d}')
    try:
        cur.execute("""
            INSERT INTO customers (
                customer_id, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies,
                contract_encoded, paperless_billing, payment_method_encoded,
                monthly_charges, total_charges, churn_binary
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (customer_id) DO NOTHING
        """, (
            cid,
            int(row.get('SeniorCitizen', 0)),
            int(row.get('Partner', 0)),
            int(row.get('Dependents', 0)),
            float(row['tenure']),
            int(row.get('PhoneService', 0)),
            int(row.get('MultipleLines', 0)),
            int(row.get('InternetService_encoded', 0)),
            int(row.get('OnlineSecurity', 0)),
            int(row.get('OnlineBackup', 0)),
            int(row.get('DeviceProtection', 0)),
            int(row.get('TechSupport', 0)),
            int(row.get('StreamingTV', 0)),
            int(row.get('StreamingMovies', 0)),
            int(row.get('Contract_encoded', 0)),
            int(row.get('PaperlessBilling', 0)),
            int(row.get('PaymentMethod_encoded', 0)),
            float(row['MonthlyCharges']),
            float(row['TotalCharges']),
            int(row['Churn_binary']),
        ))
        inserted += 1
    except Exception as e:
        pass

conn.commit()
cur.execute("SELECT COUNT(*) FROM customers")
total = cur.fetchone()[0]
cur.close()
conn.close()
print(f"Seeded {inserted} customers ✅ — Total in DB: {total}")
