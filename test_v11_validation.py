import unittest
from v11_validation import make_synthetic_validation_data, evaluate_synthetic_model

class V11ValidationTests(unittest.TestCase):
    def test_data_reproducibility(self):
        a, ya = make_synthetic_validation_data(17, 100)
        b, yb = make_synthetic_validation_data(17, 100)
        self.assertTrue(a.equals(b)); self.assertTrue(ya.equals(yb))
    def test_data_schema_and_binary_labels(self):
        X, y = make_synthetic_validation_data(samples=200)
        self.assertEqual(list(X.columns), ['rpm','load_pct','temperature_c','vibration_rms'])
        self.assertTrue(set(y.unique()).issubset({0,1}))
    def test_evaluation_outputs(self):
        model, metrics, cm, importance, X_test, y_test, prob = evaluate_synthetic_model(2026)
        self.assertEqual(set(metrics), {'Accuracy','Precision','Recall','F1','ROC-AUC'})
        self.assertTrue(all(0 <= v <= 1 for v in metrics.values()))
        self.assertEqual(cm.shape, (2,2)); self.assertEqual(len(importance),4)
        self.assertEqual(len(prob), len(y_test))
if __name__ == '__main__': unittest.main()
