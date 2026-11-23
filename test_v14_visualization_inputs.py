import unittest
import numpy as np
from signal_diagnostics import (generate_vibration_signal, calculate_spectrum,
    bearing_characteristic_frequencies, orbit_coordinates)

class TestV14VisualizationInputs(unittest.TestCase):
    def test_fft_spectrum_is_finite_and_one_sided(self):
        _, signal = generate_vibration_signal(duration_s=2, sample_rate_hz=1024, seed=7)
        freq, amp = calculate_spectrum(signal, 1024)
        self.assertEqual(freq.shape, amp.shape)
        self.assertTrue(np.all(np.isfinite(amp)))
        self.assertGreaterEqual(freq[0], 0)
        self.assertLessEqual(freq[-1], 512)

    def test_bearing_frequency_order(self):
        result = bearing_characteristic_frequencies(1800)
        self.assertEqual(set(result), {"FTF", "BSF", "BPFO", "BPFI"})
        self.assertTrue(all(v > 0 for v in result.values()))
        self.assertLess(result["FTF"], result["BPFO"])

    def test_orbit_requires_equal_probe_lengths(self):
        x, y = orbit_coordinates(np.arange(10), np.arange(10))
        self.assertEqual(len(x), len(y))
        with self.assertRaises(ValueError):
            orbit_coordinates(np.arange(10), np.arange(9))

if __name__ == "__main__":
    unittest.main()
