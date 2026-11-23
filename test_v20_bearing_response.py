import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v19_advanced_rotor import build_timoshenko_rotor
from v20_bearing_response import (apply_linearized_bearing_coefficients,
    synchronous_unbalance_response, stability_sweep)

class TestV20BearingResponse(unittest.TestCase):
    def setUp(self):
        self.rotor=build_timoshenko_rotor(elements=3, disk_node=1)
        self.coeff=dict(kxx=2.2e7,kyy=1.8e7,kxy=1e5,kyx=-8e4,
                        cxx=1400,cyy=1250,cxy=10,cyx=-5)
    def test_anisotropic_cross_coupled_matrices(self):
        r=apply_linearized_bearing_coefficients(self.rotor,self.coeff)
        self.assertAlmostEqual(r['K'][0,2],1e5)
        self.assertAlmostEqual(r['K'][2,0],-8e4)
        self.assertAlmostEqual(r['C'][0,2],10)
    def test_does_not_mutate_input(self):
        before=self.rotor['K'].copy()
        apply_linearized_bearing_coefficients(self.rotor,self.coeff)
        np.testing.assert_allclose(self.rotor['K'],before)
    def test_invalid_coefficients(self):
        bad=dict(self.coeff); bad['kxx']=-1
        with self.assertRaises(ValueError): apply_linearized_bearing_coefficients(self.rotor,bad)
    def test_unbalance_response_shape_and_positive(self):
        r=synchronous_unbalance_response(self.rotor,[600,1200,1800],1e-4)
        self.assertEqual(len(r),3)
        self.assertTrue((r.resultant_amplitude_m>0).all())
    def test_zero_unbalance_zero_response(self):
        r=synchronous_unbalance_response(self.rotor,[600,1200],0)
        np.testing.assert_allclose(r.resultant_amplitude_m,0,atol=1e-15)
    def test_stability_sweep_schema(self):
        r=stability_sweep(self.rotor,[0,600,1200])
        self.assertEqual(len(r),3)
        self.assertIn(r.stability.iloc[0],['Stable/neutral','Unstable'])
    def test_invalid_rpm(self):
        with self.assertRaises(ValueError): synchronous_unbalance_response(self.rotor,[0,100])

if __name__=='__main__': unittest.main()
