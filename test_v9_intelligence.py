import unittest
import pandas as pd
from monitoring import generate_sensor_batch
from v9_intelligence import health_index, detect_iqr_anomalies, train_synthetic_risk_model, predict_synthetic_risk, project_vibration_crossing

class V9Tests(unittest.TestCase):
    def setUp(self): self.df=generate_sensor_batch(0,8,2026)
    def test_health_bounds(self):
        s=health_index(self.df); self.assertTrue(s.health_index.between(0,100).all()); self.assertIn('risk_band',s)
    def test_anomaly_column(self): self.assertTrue(set(detect_iqr_anomalies(self.df).iqr_anomaly.unique()).issubset({0,1}))
    def test_model_output(self):
        m=train_synthetic_risk_model(7,300); out=predict_synthetic_risk(m,self.df); self.assertTrue(out.synthetic_risk_probability_pct.between(0,100).all())
    def test_projection_minimum_points(self):
        h=pd.concat([self.df.assign(batch_id=i) for i in range(2)],ignore_index=True); p=project_vibration_crossing(h); self.assertTrue((p.projection_note=='Need 3+ observations').all())
    def test_projection_with_history(self):
        batches=[]
        for i in range(4): batches.append(self.df.assign(batch_id=i,vibration_rms=self.df.vibration_rms+i*.15))
        p=project_vibration_crossing(pd.concat(batches,ignore_index=True)); self.assertEqual(len(p),len(self.df)); self.assertTrue((p.slope_per_batch>0).all())
if __name__=='__main__': unittest.main()
