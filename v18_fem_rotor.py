"""V18 educational Euler-Bernoulli FE rotor beam and modal comparison tools.

This is a linear, isotropic, non-gyroscopic beam model with translational bearing
springs. It is not a full Timoshenko/gyroscopic rotor-bearing solver.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import eigh


def _positive(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def build_rotor_matrices(length_m=1.0, diameter_m=0.04, youngs_modulus_pa=2.0e11,
                         density_kg_m3=7800.0, elements=8, bearing_stiffness_n_m=2.0e7):
    """Assemble Euler-Bernoulli shaft K,M matrices for one lateral bending plane."""
    L = _positive("length_m", length_m); d = _positive("diameter_m", diameter_m)
    E = _positive("youngs_modulus_pa", youngs_modulus_pa); rho = _positive("density_kg_m3", density_kg_m3)
    kb = _positive("bearing_stiffness_n_m", bearing_stiffness_n_m)
    if int(elements) != elements or not 2 <= int(elements) <= 100:
        raise ValueError("elements must be an integer from 2 to 100")
    n = int(elements); le = L/n; A=np.pi*d*d/4; I=np.pi*d**4/64
    ke = E*I/le**3*np.array([[12,6*le,-12,6*le],[6*le,4*le**2,-6*le,2*le**2],[-12,-6*le,12,-6*le],[6*le,2*le**2,-6*le,4*le**2]],float)
    me = rho*A*le/420*np.array([[156,22*le,54,-13*le],[22*le,4*le**2,13*le,-3*le**2],[54,13*le,156,-22*le],[-13*le,-3*le**2,-22*le,4*le**2]],float)
    ndof=2*(n+1); K=np.zeros((ndof,ndof)); M=np.zeros_like(K)
    for e in range(n):
        dofs=np.array([2*e,2*e+1,2*e+2,2*e+3])
        K[np.ix_(dofs,dofs)] += ke; M[np.ix_(dofs,dofs)] += me
    # Bearing supports at both shaft ends: translational spring only.
    K[0,0] += kb; K[-2,-2] += kb
    return K,M,np.linspace(0,L,n+1),A,I


def solve_rotor_modes(length_m=1.0, diameter_m=0.04, youngs_modulus_pa=2.0e11,
                      density_kg_m3=7800.0, elements=8, bearing_stiffness_n_m=2.0e7,
                      modes=6):
    """Return positive natural frequencies and normalized lateral mode shapes."""
    K,M,x,A,I=build_rotor_matrices(length_m,diameter_m,youngs_modulus_pa,density_kg_m3,elements,bearing_stiffness_n_m)
    if int(modes) != modes or not 1 <= int(modes) <= min(12, K.shape[0]-1):
        raise ValueError("modes must be an integer between 1 and min(12, available DOFs)")
    vals,vecs=eigh(K,M,check_finite=True)
    keep=np.where(vals>1e-7)[0][:int(modes)]
    if len(keep)<int(modes): raise ValueError("Could not obtain requested positive modes; revise supports/mesh")
    freq=np.sqrt(vals[keep])/(2*np.pi)
    shapes=vecs[0::2,keep]
    for j in range(shapes.shape[1]):
        scale=np.max(np.abs(shapes[:,j])); shapes[:,j] /= scale if scale else 1
        if shapes[np.argmax(np.abs(shapes[:,j])),j] < 0: shapes[:,j]*=-1
    return {"frequencies_hz":freq,"critical_speeds_rpm":freq*60.0,"mode_shapes":shapes,
            "nodes_m":x,"stiffness_matrix":K,"mass_matrix":M,"area_m2":A,"second_moment_m4":I}


def compare_experimental_modes(numerical_frequencies_hz, experimental: pd.DataFrame, numerical_shapes=None):
    """Compare CSV modes (mode,freq_hz; optional shape_node_0..N) with FE modes.

    MAC is calculated only when experimental shape columns are supplied and the
    number of ordinates matches the FE node count.
    """
    required={"mode","freq_hz"}
    if not required.issubset(experimental.columns):
        raise ValueError("Experimental CSV must contain columns: mode, freq_hz")
    nf=np.asarray(numerical_frequencies_hz,dtype=float)
    if nf.ndim!=1 or not len(nf) or np.any(~np.isfinite(nf)) or np.any(nf<=0): raise ValueError("numerical frequencies must be finite and positive")
    exp=experimental.copy(); exp["mode"]=pd.to_numeric(exp["mode"],errors="coerce"); exp["freq_hz"]=pd.to_numeric(exp["freq_hz"],errors="coerce")
    exp=exp.dropna(subset=["mode","freq_hz"])
    if exp.empty or (exp.freq_hz<=0).any(): raise ValueError("Experimental frequencies must be positive numeric values")
    rows=[]; shape_cols=sorted([c for c in exp.columns if c.startswith("shape_node_")],key=lambda s:int(s.rsplit("_",1)[1]))
    for _,r in exp.iterrows():
        idx=int(r["mode"])-1
        if idx<0 or idx>=len(nf): continue
        num=float(nf[idx]); obs=float(r["freq_hz"]); err=100*(num-obs)/obs
        row={"mode":idx+1,"experimental_hz":obs,"numerical_hz":num,"error_hz":num-obs,"error_percent":err,"MAC":np.nan}
        if shape_cols and numerical_shapes is not None:
            exp_shape=pd.to_numeric(r[shape_cols],errors="coerce").to_numpy(dtype=float)
            num_shape=np.asarray(numerical_shapes)[:,idx]
            if len(exp_shape)==len(num_shape) and np.all(np.isfinite(exp_shape)):
                den=float(np.dot(exp_shape,exp_shape)*np.dot(num_shape,num_shape))
                row["MAC"]=float((np.dot(exp_shape,num_shape)**2)/den) if den>0 else np.nan
        rows.append(row)
    if not rows: raise ValueError("No experimental mode numbers matched available numerical modes")
    return pd.DataFrame(rows)
