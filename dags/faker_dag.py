from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import random
import uuid

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def generate_raw_customers(**context):
    hook = PostgresHook(postgres_conn_id='churn_postgres_conn')
    conn = hook.get_conn()
    cur  = conn.cursor()

    n_new = random.randint(30, 50)
    for _ in range(n_new):
        customer_id = f"FAKE_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
        
        # Giả lập dữ liệu thô (Raw)
        gender = random.choice(['Male', 'Female'])
        senior = random.choices([0, 1], weights=[0.8, 0.2])[0]
        partner = random.choice(['Yes', 'No'])
        dependents = random.choice(['Yes', 'No'])
        tenure = random.randint(0, 72)
        monthly = round(random.uniform(18, 118), 2)
        
        # Lỗi khoảng trắng (10% xác suất)
        total = " " if random.random() < 0.1 else str(round(monthly * tenure, 2))

        cur.execute("""
            INSERT INTO customers (
                customer_id, gender, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service, online_security,
                online_backup, device_protection, tech_support, streaming_tv,
                streaming_movies, contract, paperless_billing, payment_method,
                monthly_charges, total_charges, churn
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            customer_id, gender, senior, partner, dependents, tenure,
            'Yes', random.choice(['Yes', 'No', 'No phone service']),
            random.choice(['DSL', 'Fiber optic', 'No']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Yes', 'No', 'No internet service']),
            random.choice(['Month-to-month', 'One year', 'Two year']),
            random.choice(['Yes', 'No']),
            random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']),
            monthly, total, random.choice(['Yes', 'No'])
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Generated {n_new} raw customers.")

with DAG(
    'faker_data_generator',
    default_args=default_args,
    schedule_interval='0 1 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['data_gen', 'raw']
) as dag:

    generate = PythonOperator(
        task_id='generate_raw_customers',
        python_callable=generate_raw_customers,
    )

    trigger_etl = TriggerDagRunOperator(
        task_id='trigger_etl_pipeline',
        trigger_dag_id='etl_pipeline',
        conf={"reason": "new_raw_data"},
    )

    generate >> trigger_etl