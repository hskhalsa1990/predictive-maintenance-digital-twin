import unittest
import pandas as pd
from monitoring import generate_sensor_batch
from v10_enterprise import score_enterprise, make_work_order_queue, compare_batches
from v8_engine import add_batch_id, validate_sensor_frame

class V10EnterpriseTests(unittest.TestCase):
    def setUp(self):
        self.data=generate_sensor_batch(0, machines=8, seed=100)

    def test_scoring_columns_and_bounded_scores(self):
        scored=score_enterprise(self.data, seed=100)
        self.assertTrue({'enterprise_priority_score','priority','rank','recommended_action'}.issubset(scored.columns))
        self.assertTrue(scored.enterprise_priority_score.between(0,100).all())
        self.assertEqual(len(scored),8)

    def test_ranking_descending(self):
        scored=score_enterprise(self.data, seed=100)
        self.assertTrue(scored.enterprise_priority_score.is_monotonic_decreasing)

    def test_queue_only_non_routine(self):
        scored=score_enterprise(self.data, seed=100)
        queue=make_work_order_queue(scored)
        self.assertTrue(queue.priority.isin(['Moderate','High','Critical']).all())
        self.assertIn('work_order_status',queue.columns)

    def test_empty_queue_schema(self):
        empty=self.data.iloc[:0]
        q=make_work_order_queue(score_enterprise(empty, seed=100))
        self.assertIn('work_order_status',q.columns)

    def test_audit_passes_clean_generated_batch(self):
        self.assertEqual(validate_sensor_frame(self.data)['status'],'PASS')

    def test_trend_empty_schema(self):
        self.assertIn('trend', compare_batches(pd.DataFrame()).columns)

    def test_repeatable_seed(self):
        a=score_enterprise(self.data, seed=101).enterprise_priority_score.tolist()
        b=score_enterprise(self.data, seed=101).enterprise_priority_score.tolist()
        self.assertEqual(a,b)

if __name__=='__main__': unittest.main()
