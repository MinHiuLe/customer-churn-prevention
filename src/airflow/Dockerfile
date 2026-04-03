FROM apache/airflow:2.8.2

USER root
# Đã có sẵn libgomp1 cho LightGBM, rất chuẩn!
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    lightgbm==4.3.0 \
    scikit-learn==1.3.2 \
    lifelines==0.27.4 \
    scipy==1.10.1 \
    redis \
    psycopg2-binary \
    mlflow==2.11.1 \
    dvc