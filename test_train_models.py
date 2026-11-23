import ast, json, tempfile, unittest
from pathlib import Path
import numpy as np
from train_models import build_frame, extract_features, feature_columns, train, generate_dataset

ROOT = Path(__file__).resolve().parent


def api_fields():
    tree = ast.parse((ROOT / "api.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SensorFeatures")
    return [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]


class TestTrainModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_frame(3, 12, 7)

    def test_feature_columns_match_api_schema(self):
        self.assertEqual(sorted(feature_columns(self.df)), sorted(api_fields()))

    def test_features_are_finite(self):
        cols = feature_columns(self.df)
        self.assertTrue(np.isfinite(self.df[cols].to_numpy(dtype=float)).all())

    def test_reproducible_with_same_seed(self):
        again = build_frame(3, 12, 7)
        np.testing.assert_allclose(self.df.vibration_rms.to_numpy(), again.vibration_rms.to_numpy())

    def test_rejects_too_small_runs(self):
        with self.assertRaises(ValueError):
            generate_dataset(2, 50, 1)
        with self.assertRaises(ValueError):
            generate_dataset(5, 3, 1)

    def test_train_writes_artifacts_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            m = train(3, 24, 3, d)
            for name in ("anomaly_model.joblib", "fault_classifier.joblib", "rul_regressor.joblib",
                         "feature_columns.json", "metrics.json", "metadata.json", "synthetic_sensor_dataset.csv"):
                self.assertTrue((Path(d) / name).exists(), name)
            self.assertEqual(json.loads((Path(d) / "metadata.json").read_text())["holdout_machine"], "UNIT_3")
            self.assertTrue(0 <= m["fault_accuracy"] <= 1)
            self.assertIn("Synthetic", m["notice"])


if __name__ == "__main__":
    unittest.main()
