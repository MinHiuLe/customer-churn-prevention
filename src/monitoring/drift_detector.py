import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

class DriftDetector:
    """
    Evidently-based drift detector.
    So sánh reference data vs current data → trigger retrain nếu drift.
    """

    def __init__(self,
                 reference_path: str = 'data/processed/features.csv',
                 current_path: str = 'data/processed/features_latest.csv',
                 drift_threshold: float = 0.3,
                 report_dir: str = 'data/processed/drift_reports'):

        self.reference_path = reference_path
        self.current_path = current_path
        self.drift_threshold = drift_threshold
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.feature_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'recency_risk', 'service_count', 'charge_per_month',
            'clv_proxy', 'contract_stability', 'digital_engagement',
            'SeniorCitizen', 'Contract_encoded', 'InternetService_encoded'
        ]

    def load_data(self):
        reference = pd.read_csv(self.reference_path)
        current = pd.read_csv(self.current_path)

        # Chỉ lấy feature cols có trong cả 2
        cols = [c for c in self.feature_cols
                if c in reference.columns and c in current.columns]

        return reference[cols], current[cols]

    def detect(self):
        print("Loading reference and current data...")
        reference, current = self.load_data()
        print(f"Reference: {reference.shape}, Current: {current.shape}")

        # Evidently drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='tenure'),
            ColumnDriftMetric(column_name='MonthlyCharges'),
            ColumnDriftMetric(column_name='charge_per_month'),
            ColumnDriftMetric(column_name='Contract_encoded'),
        ])

        report.run(reference_data=reference, current_data=current)

        # Extract results
        result = report.as_dict()
        dataset_drift = result['metrics'][0]['result']

        drift_score = dataset_drift['share_of_drifted_columns']
        n_drifted = dataset_drift['number_of_drifted_columns']
        n_total = dataset_drift['number_of_columns']
        drift_detected = dataset_drift['dataset_drift']
        # Override: dùng threshold của mình thay vì Evidently default
        if drift_score >= self.drift_threshold:
            drift_detected = True

        print(f"\n=== DRIFT DETECTION RESULTS ===")
        print(f"Drifted columns:  {n_drifted}/{n_total}")
        print(f"Drift score:      {drift_score:.3f}")
        print(f"Threshold:        {self.drift_threshold}")
        print(f"Drift detected:   {'⚠️  YES' if drift_detected else '✅ NO'}")

        # Column-level details
        print(f"\n=== COLUMN-LEVEL DRIFT ===")
        for metric in result['metrics'][1:]:
            col = metric['result']['column_name']
            drifted = metric['result']['drift_detected']
            score = metric['result'].get('stattest_threshold', 'N/A')
            print(f"  {col:<30} {'DRIFT ⚠️' if drifted else 'OK ✅'}")

        # Save HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.report_dir / f'drift_report_{timestamp}.html'
        report.save_html(str(report_path))
        print(f"\nDrift report saved: {report_path}")

        # Save JSON summary
        summary = {
            'timestamp': timestamp,
            'drift_score': drift_score,
            'n_drifted_columns': n_drifted,
            'n_total_columns': n_total,
            'drift_detected': drift_detected,
            'threshold': self.drift_threshold,
        }
        summary_path = self.report_dir / 'drift_summary_latest.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Trigger retrain flag nếu drift detected
        if drift_detected or drift_score >= self.drift_threshold:
            flag_path = Path('data/processed/drift_detected.flag')
            flag_path.touch()
            print(f"\n🚨 DRIFT FLAG SET — Airflow retrain DAG sẽ trigger!")
        else:
            print(f"\n✅ No action needed")

        return summary

    def simulate_drift(self, week: int = 5):
        """
        Inject synthetic drift vào current data để test.
        Dùng weekly_drift_data.csv đã generate sẵn.
        """
        drift_data_path = 'data/processed/weekly_drift_data.csv'
        if not os.path.exists(drift_data_path):
            print("Weekly drift data not found — using features.csv")
            return

        df = pd.read_csv(drift_data_path)
        if 'week' in df.columns:
            current = df[df['week'] == week]
        else:
            # Inject drift manually
            current = pd.read_csv(self.reference_path).copy()
            current['MonthlyCharges'] *= 1.3   # Giá tăng 30%
            current['tenure'] *= 0.7           # Tenure giảm
            current['charge_per_month'] *= 1.4

        current.to_csv(self.current_path, index=False)
        print(f"Drift injected for week {week} ✅")
        return current


if __name__ == '__main__':
    detector = DriftDetector()
    summary = detector.detect()
    print(f"\nDrift flag exists: {Path('data/processed/drift_detected.flag').exists()}")
