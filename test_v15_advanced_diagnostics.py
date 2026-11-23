import unittest
import numpy as np
from advanced_diagnostics import signal_features, analytic_envelope, envelope_spectrum, rotational_order_spectrum, spectral_band_energy

class TestV15AdvancedDiagnostics(unittest.TestCase):
    def setUp(self):
        self.fs = 2048
        self.t = np.arange(self.fs*2)/self.fs
        self.x = np.sin(2*np.pi*30*self.t) + .2*np.sin(2*np.pi*120*self.t)

    def test_features_are_finite(self):
        f = signal_features(self.x, self.fs)
        self.assertGreater(f['RMS'], 0)
        self.assertGreater(f['Crest Factor'], 1)
        self.assertTrue(all(np.isfinite(list(f.values()))))

    def test_envelope_shape_and_spectrum(self):
        env = analytic_envelope(self.x)
        freq, amp = envelope_spectrum(self.x, self.fs)
        self.assertEqual(env.shape, self.x.shape)
        self.assertEqual(freq.shape, amp.shape)
        self.assertTrue(np.all(amp >= 0))

    def test_order_mapping(self):
        order, amp = rotational_order_spectrum(self.x, self.fs, 1800)
        self.assertEqual(order.shape, amp.shape)
        self.assertAlmostEqual(order[np.argmax(amp)], 1.0, delta=.03)

    def test_band_energy_sums_to_one(self):
        result = spectral_band_energy(self.x, self.fs, {'all': (0, self.fs/2)})
        self.assertAlmostEqual(result['all'], 1.0, places=6)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): signal_features([1, np.nan]*5, self.fs)
        with self.assertRaises(ValueError): rotational_order_spectrum(self.x, self.fs, 0)

if __name__ == '__main__': unittest.main()
