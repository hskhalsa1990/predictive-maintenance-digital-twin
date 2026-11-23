import unittest
import numpy as np
from signal_diagnostics import (bearing_characteristic_frequencies,
    generate_vibration_signal, calculate_spectrum, orbit_coordinates)

class SignalDiagnosticsTests(unittest.TestCase):
    def test_bearing_frequencies_positive_and_ordered(self):
        f = bearing_characteristic_frequencies(1800)
        self.assertGreater(f['BPFI'], f['BPFO'])
        self.assertGreater(f['BPFO'], f['FTF'])
        self.assertTrue(all(v > 0 for v in f.values()))
    def test_bearing_invalid_geometry(self):
        with self.assertRaises(ValueError): bearing_characteristic_frequencies(0)
        with self.assertRaises(ValueError): bearing_characteristic_frequencies(1800, ball_diameter_mm=50)
    def test_signal_reproducible(self):
        t1,x1=generate_vibration_signal(seed=7); t2,x2=generate_vibration_signal(seed=7)
        np.testing.assert_array_equal(t1,t2); np.testing.assert_array_equal(x1,x2)
    def test_signal_has_expected_length(self):
        t,x=generate_vibration_signal(duration_s=2,sample_rate_hz=1000)
        self.assertEqual(len(t),2000); self.assertEqual(len(x),2000)
    def test_nyquist_validation(self):
        with self.assertRaises(ValueError): generate_vibration_signal(sample_rate_hz=100, fault_hz=60)
    def test_fft_peak_near_input_tone(self):
        fs=2048; t=np.arange(fs)/fs; x=np.sin(2*np.pi*64*t)
        f,a=calculate_spectrum(x,fs)
        self.assertAlmostEqual(float(f[np.argmax(a)]),64.0,delta=1.0)
    def test_fft_rejects_bad_input(self):
        with self.assertRaises(ValueError): calculate_spectrum(np.ones(3),100)
    def test_orbit_shape_validation(self):
        with self.assertRaises(ValueError): orbit_coordinates(np.ones(4),np.ones(3))
    def test_orbit_returns_arrays(self):
        x,y=orbit_coordinates(np.arange(5),np.arange(5)*2)
        self.assertEqual(len(x),5); self.assertEqual(len(y),5)

if __name__ == '__main__': unittest.main()
