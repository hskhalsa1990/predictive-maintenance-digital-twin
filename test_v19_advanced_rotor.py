import unittest
import numpy as np
from v19_advanced_rotor import build_timoshenko_rotor, solve_rotor_campbell, critical_speed_crossings

class TestV19AdvancedRotor(unittest.TestCase):
    def test_matrix_shapes_and_symmetry(self):
        r=build_timoshenko_rotor(elements=3)
        for key in ('K','M','C'):
            self.assertTrue(np.allclose(r[key],r[key].T))
        self.assertEqual(r['K'].shape,(16,16))
        self.assertTrue(np.allclose(r['G'],-r['G'].T))
    def test_mass_positive(self):
        r=build_timoshenko_rotor(elements=3)
        self.assertGreater(np.linalg.eigvalsh(r['M']).min(),0)
    def test_bearing_damping_nonnegative(self):
        r=build_timoshenko_rotor(elements=3,bearing_damping_n_s_m=1500)
        self.assertGreaterEqual(np.linalg.eigvalsh(r['C']).min(),-1e-9)
    def test_disk_mass_changes_mass_matrix(self):
        a=build_timoshenko_rotor(elements=3,disk_mass_kg=2)
        b=build_timoshenko_rotor(elements=3,disk_mass_kg=10)
        self.assertGreater(np.trace(b['M']),np.trace(a['M']))
    def test_disk_polar_inertia_generates_gyro(self):
        r=build_timoshenko_rotor(elements=3,disk_polar_inertia_kg_m2=.08)
        self.assertGreater(np.linalg.norm(r['G']),0)
        self.assertTrue(np.allclose(r['G'],-r['G'].T))
    def test_campbell_contains_whirl_branches(self):
        r=build_timoshenko_rotor(elements=3)
        c=solve_rotor_campbell(r,[0,500,1000])
        self.assertIn('Forward',set(c.whirl))
        self.assertIn('Backward',set(c.whirl))
        self.assertTrue((c.frequency_hz>0).all())
        self.assertTrue(set(c.whirl).issubset({'Forward','Backward'}))
    def test_gyro_speed_changes_eigenfrequencies(self):
        r=build_timoshenko_rotor(elements=3,disk_polar_inertia_kg_m2=.12)
        c=solve_rotor_campbell(r,[0,5000])
        f0=np.sort(c[c.rpm==0].frequency_hz.to_numpy())
        f1=np.sort(c[c.rpm==5000].frequency_hz.to_numpy())
        self.assertEqual(len(f0),len(f1))
        self.assertGreater(np.max(np.abs(f1-f0)),1e-5)
    def test_invalid_speed_rejected(self):
        r=build_timoshenko_rotor(elements=3)
        with self.assertRaises(ValueError): solve_rotor_campbell(r,[-100,200])
    def test_critical_crossing_schema(self):
        import pandas as pd
        d=pd.DataFrame({'rpm':[0,1000,2000],'frequency_hz':[5,15,25],'whirl':['Forward']*3})
        out=critical_speed_crossings(d,order=1)
        self.assertIn('critical_speed_rpm',out.columns)
    def test_invalid_geometry(self):
        with self.assertRaises(ValueError): build_timoshenko_rotor(diameter_m=0)
    def test_invalid_damping(self):
        with self.assertRaises(ValueError): build_timoshenko_rotor(bearing_damping_n_s_m=-1)

if __name__=='__main__': unittest.main()
