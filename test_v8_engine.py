import unittest
import pandas as pd
from v8_engine import validate_sensor_frame, trend_summary, add_batch_id
from monitoring import generate_sensor_batch


class V8EngineTests(unittest.TestCase):
    def test_quality_pass(self):
        df=generate_sensor_batch(0,4)
        self.assertEqual(validate_sensor_frame(df)["status"],"PASS")
    def test_missing_columns_review(self):
        self.assertEqual(validate_sensor_frame(pd.DataFrame({"rpm":[1]}))["status"],"REVIEW")
    def test_duplicate_detected(self):
        df=pd.DataFrame({"machine_id":["A","A"],"rpm":[1,1],"load_pct":[2,2],"temperature_c":[3,3],"vibration_rms":[4,4]})
        self.assertEqual(validate_sensor_frame(df)["duplicate_rows"],1)
    def test_batch_id_added_without_mutating(self):
        df=generate_sensor_batch(0,2); out=add_batch_id(df,7)
        self.assertNotIn("batch_id",df.columns); self.assertTrue((out.batch_id==7).all())
    def test_trend_has_machine_rows(self):
        a=add_batch_id(generate_sensor_batch(0,3),0); b=add_batch_id(generate_sensor_batch(1,3),1)
        result=trend_summary(pd.concat([a,b],ignore_index=True))
        self.assertEqual(len(result),3); self.assertIn("trend",result.columns)
if __name__=="__main__": unittest.main()
