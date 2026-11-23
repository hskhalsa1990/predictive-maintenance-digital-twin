import unittest
from monitoring import generate_sensor_batch
from v10_enterprise import score_enterprise
from v11_validation import evaluate_synthetic_model
from v21_3d_atlas import build_3d_atlas

class TestV21Complete3DAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = generate_sensor_batch(0, machines=8, seed=2026)
        scored = score_enterprise(raw, 2026)
        _, _, _, importance, X_test, y_test, prob = evaluate_synthetic_model(2026)
        cls.atlas = build_3d_atlas(raw, scored, importance, X_test, prob, 2026, y_test)

    def test_at_least_25_diagrams(self):
        self.assertGreaterEqual(sum(len(v) for v in self.atlas.values()), 25)

    def test_all_figures_have_data(self):
        for group, items in self.atlas.items():
            for title, fig, note in items:
                self.assertTrue(len(fig.data), f"Empty figure: {group} / {title}")

    def test_every_trace_is_3d(self):
        allowed = {"scatter3d", "surface", "mesh3d", "cone", "streamtube", "volume", "isosurface"}
        for group, items in self.atlas.items():
            for title, fig, note in items:
                for trace in fig.data:
                    self.assertIn(trace.type, allowed, f"Not 3D: {group} / {title} / {trace.type}")

    def test_major_domains_present(self):
        self.assertIn("Fleet • AI • Operations", self.atlas)
        self.assertIn("Vibration • Bearings", self.atlas)
        self.assertIn("Rotor • FEM • Dynamics", self.atlas)
        self.assertIn("Data Quality • Reporting", self.atlas)

    def test_no_duplicate_diagram_titles(self):
        titles = [title for items in self.atlas.values() for title, _, _ in items]
        self.assertEqual(len(titles), len(set(titles)))

if __name__ == "__main__":
    unittest.main()
