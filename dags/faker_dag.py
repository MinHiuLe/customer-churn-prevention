from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2
import random
import os
import uuid

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def generate_customers(**context):
    """Sinh 50-100 customers mới mỗi ngày và insert vào PostgreSQL"""

    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', 5432),
        dbname=os.getenv('POSTGRES_DB', 'churn_db'),
        user=os.getenv('POSTGRES_USER', 'churn_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'churn_pass'),
    )
    cur = conn.cursor()

    # Ensure table exists
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

    n_new = random.randint(50, 100)
    inserted = 0

    for _ in range(n_new):
        # Causal data generation
        contract = random.choices([0, 1, 2], weights=[0.55, 0.25, 0.20])[0]
        internet = random.choices([0, 1, 2], weights=[0.20, 0.35, 0.45])[0]
        tenure = max(0, int(random.gauss(32, 24)))
        senior = random.choices([0, 1], weights=[0.84, 0.16])[0]

        # Monthly charges correlated with internet service
        base_charge = {0: 25, 1: 55, 2: 80}[internet]
        monthly = round(base_charge + random.gauss(0, 10), 2)
        monthly = max(20, min(120, monthly))
        total = round(monthly * max(tenure, 1) * random.uniform(0.9, 1.1), 2)

        # Services
        services = [random.randint(0,1) for _ in range(8)]

        # Churn probability — causal model
        churn_prob = 0.3
        if contract == 0: churn_prob += 0.25
        if contract == 2: churn_prob -= 0.20
        if internet == 2: churn_prob += 0.15
        if tenure > 48: churn_prob -= 0.20
        if tenure < 6: churn_prob += 0.15
        if senior: churn_prob += 0.05
        churn_prob = max(0.02, min(0.95, churn_prob))
        churn = 1 if random.random() < churn_prob else 0

        customer_id = f"FAKE_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8].upper()}"

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
            customer_id, senior,
            random.randint(0,1), random.randint(0,1),
            tenure,
            services[0], services[1], internet,
            services[2], services[3], services[4],
            services[5], services[6], services[7],
            contract, random.randint(0,1),
            random.randint(0,3),
            monthly, total, churn
        ))
        inserted += 1

    conn.commit()

    # Log total count
    cur.execute("SELECT COUNT(*) FROM customers")
    total_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"Inserted {inserted} new customers ✅")
    print(f"Total customers in DB: {total_count}")
    return inserted

with DAG(
    'faker_data_generator',
    default_args=default_args,
    description='Daily: generate synthetic customers → PostgreSQL',
    schedule_interval='0 1 * * *',  # 1AM — trước ETL
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['data', 'faker'],
) as dag:

    generate = PythonOperator(
        task_id='generate_customers',
        python_callable=generate_customers,
    )
