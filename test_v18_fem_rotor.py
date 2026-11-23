import unittest
import numpy as np
import pandas as pd
from v18_fem_rotor import build_rotor_matrices, solve_rotor_modes, compare_experimental_modes

class TestV18FEMRotor(unittest.TestCase):
    def test_matrix_dimensions_and_symmetry(self):
        K,M,x,A,I=build_rotor_matrices(elements=4)
        self.assertEqual(K.shape,(10,10)); self.assertTrue(np.allclose(K,K.T)); self.assertTrue(np.allclose(M,M.T))
    def test_positive_definite_mass(self):
        K,M,*_=build_rotor_matrices(elements=4)
        self.assertGreater(np.linalg.eigvalsh(M).min(),0)
    def test_modes_positive_sorted(self):
        r=solve_rotor_modes(elements=6,modes=4)
        self.assertTrue(np.all(r['frequencies_hz']>0)); self.assertTrue(np.all(np.diff(r['frequencies_hz'])>0))
    def test_shapes_normalized(self):
        r=solve_rotor_modes(elements=5,modes=3)
        np.testing.assert_allclose(np.max(np.abs(r['mode_shapes']),axis=0),1)
    def test_critical_speed_conversion(self):
        r=solve_rotor_modes(elements=5,modes=3)
        np.testing.assert_allclose(r['critical_speeds_rpm'],r['frequencies_hz']*60)
    def test_stiffer_shaft_higher_frequency(self):
        a=solve_rotor_modes(diameter_m=.03,elements=6,modes=1)['frequencies_hz'][0]
        b=solve_rotor_modes(diameter_m=.04,elements=6,modes=1)['frequencies_hz'][0]
        self.assertGreater(b,a)
    def test_invalid_geometry(self):
        with self.assertRaises(ValueError): build_rotor_matrices(diameter_m=0)
    def test_invalid_mesh(self):
        with self.assertRaises(ValueError): build_rotor_matrices(elements=1)
    def test_compare_frequency_error(self):
        df=pd.DataFrame({'mode':[1,2],'freq_hz':[10.,30.]})
        out=compare_experimental_modes([11.,33.],df)
        np.testing.assert_allclose(out.error_percent,[10.,10.])
    def test_compare_mac(self):
        shapes=np.array([[0.,0.],[.5,.5],[1.,-1.]])
        df=pd.DataFrame({'mode':[1],'freq_hz':[10.],'shape_node_0':[0.],'shape_node_1':[.5],'shape_node_2':[1.]})
        out=compare_experimental_modes([10.],df,shapes)
        self.assertAlmostEqual(out.MAC.iloc[0],1.0)
    def test_bad_experimental_schema(self):
        with self.assertRaises(ValueError): compare_experimental_modes([10.],pd.DataFrame({'freq':[10]}))

if __name__=='__main__': unittest.main()
