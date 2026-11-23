import unittest
import numpy as np
import pandas as pd
from v17_rotor_lab import validate_sensor_csv, campbell_data, bode_unbalance_response, waterfall_spectra

class TestV17RotorLab(unittest.TestCase):
    def test_csv_valid(self):
        df=pd.DataFrame({"timestamp":["2026-01-01T00:00:00Z"],"machine_id":["M1"],"rpm":[1500],"temperature_c":[55],"vibration_rms":[0.5]})
        self.assertTrue(validate_sensor_csv(df)["valid"])
    def test_csv_missing(self):
        self.assertFalse(validate_sensor_csv(pd.DataFrame({"rpm":[1000]}))["valid"])
    def test_csv_duplicate(self):
        df=pd.DataFrame({"timestamp":["2026-01-01","2026-01-01"],"machine_id":["M1","M1"],"rpm":[1000,1000],"temperature_c":[50,50],"vibration_rms":[.2,.2]})
        self.assertEqual(validate_sensor_csv(df)["duplicate_rows"],1)
    def test_campbell(self):
        rpm,orders,nat=campbell_data([600,1200,1800],(20,50),3)
        np.testing.assert_allclose(orders["1X"],[10,20,30]); self.assertEqual(len(nat),2)
    def test_campbell_invalid(self):
        with self.assertRaises(ValueError): campbell_data([1000],max_order=2)
    def test_bode(self):
        rpm,amp,phase=bode_unbalance_response(np.linspace(300,3000,100),30,.05)
        self.assertEqual(len(amp),100); self.assertTrue(np.all(np.isfinite(phase)))
    def test_bode_invalid(self):
        with self.assertRaises(ValueError): bode_unbalance_response([1000,2000],30,0)
    def test_waterfall(self):
        fs=2048; t=np.arange(4096)/fs
        signals=[np.sin(2*np.pi*30*t),np.sin(2*np.pi*40*t)]
        f,rpm,psd=waterfall_spectra(signals,fs,[900,1200],512)
        self.assertEqual(psd.shape[0],2); self.assertEqual(len(rpm),2)
    def test_waterfall_invalid(self):
        with self.assertRaises(ValueError): waterfall_spectra([np.ones(20),np.ones(20)],1000,[1000,1200])

if __name__ == '__main__': unittest.main()
