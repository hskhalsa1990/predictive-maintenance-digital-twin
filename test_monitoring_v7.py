import unittest
from monitoring import generate_sensor_batch, score_maintenance, build_work_orders

class MonitoringTests(unittest.TestCase):
    def test_batch_is_reproducible(self):
        a = generate_sensor_batch(2, 5)
        b = generate_sensor_batch(2, 5)
        self.assertTrue(a.equals(b))

    def test_batch_machine_count(self):
        self.assertEqual(len(generate_sensor_batch(0, 7)), 7)

    def test_score_is_bounded(self):
        row = {"vibration_rms": 1.0, "temperature_c": 50, "anomaly_flag": 0}
        score, band = score_maintenance(row)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn(band, ["Routine", "Plan inspection", "Priority inspection"])

    def test_work_order_schema(self):
        df = generate_sensor_batch(8, 10)
        orders = build_work_orders(df)
        self.assertIn("machine_id", orders.columns)
        self.assertIn("status", orders.columns)

if __name__ == "__main__":
    unittest.main()
