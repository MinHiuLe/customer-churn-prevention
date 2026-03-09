FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.27.1 \
    lightgbm==4.3.0 \
    shap==0.46.0 \
    pandas==2.2.0 \
    numpy==1.26.4 \
    scikit-learn==1.4.0 \
    lifelines==0.28.0 \
    scipy==1.11.4 \
    joblib \
    pyarrow==15.0.2

COPY src/serving/main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
