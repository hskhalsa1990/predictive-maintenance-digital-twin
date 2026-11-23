import unittest
import numpy as np
from v16_rotordynamics import bandpass_envelope_spectrum, tachometer_order_track, bearing_peak_match

class TestV16RotorDynamics(unittest.TestCase):
    def test_bandpass_envelope_shapes(self):
        fs=4096; t=np.arange(fs*2)/fs
        x=(1+.4*np.sin(2*np.pi*30*t))*np.sin(2*np.pi*700*t)
        filt,env,f,a=bandpass_envelope_spectrum(x,fs,500,900)
        self.assertEqual(len(env),len(x)); self.assertEqual(len(f),len(a)); self.assertGreater(f[np.argmax(a)],20)
    def test_bandpass_rejects_bad_band(self):
        with self.assertRaises(ValueError): bandpass_envelope_spectrum(np.ones(64),100,10,60)
    def test_bandpass_rejects_short(self):
        with self.assertRaises(ValueError): bandpass_envelope_spectrum(np.ones(10),100,5,20)
    def test_tach_order_tracking_constant_speed(self):
        fs=4096; duration=4; t=np.arange(fs*duration)/fs
        rpm=1200; fr=rpm/60
        x=np.sin(2*np.pi*fr*t)+.3*np.sin(2*np.pi*3*fr*t)
        pulses=np.arange(0,duration,1/fr)
        rev,ang,order,amp=tachometer_order_track(x,fs,pulses,256)
        self.assertEqual(len(ang),len(order)*2-2 if len(order)>1 else len(ang))
        self.assertLess(abs(order[np.argmax(amp[1:])+1]-1),.08)
    def test_tach_variable_speed(self):
        fs=4096; t=np.arange(fs*4)/fs
        # linearly increasing rotational frequency from 15 to 25 Hz
        phase=2*np.pi*(15*t+0.5*(10/4)*t*t)
        x=np.sin(phase)
        pulse_times=np.linspace(0,4,81)
        _,_,orders,amps=tachometer_order_track(x,fs,pulse_times,256)
        self.assertLess(abs(orders[np.argmax(amps[1:])+1]-1),.08)
    def test_tach_rejects_nonmonotonic(self):
        with self.assertRaises(ValueError): tachometer_order_track(np.ones(100),100,[0,.5,.4])
    def test_bearing_peak_found(self):
        f=np.linspace(0,100,1001); a=np.zeros_like(f); a[200]=1
        out=bearing_peak_match(f,a,{"BPFO":20},.02)
        self.assertTrue(out[0]["peak_found"])
    def test_bearing_peak_missing(self):
        f=np.linspace(0,100,1001); a=np.zeros_like(f)
        out=bearing_peak_match(f,a,{"BPFI":60},.02)
        self.assertFalse(out[0]["peak_found"])

if __name__ == '__main__': unittest.main()
