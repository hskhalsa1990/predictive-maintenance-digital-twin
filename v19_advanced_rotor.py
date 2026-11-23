"""V19 educational Timoshenko rotor FE model with disk gyroscopics and bearing damping.

DOFs per node: [x, slope_y, y, slope_x].  The shaft uses a 2-plane
Timoshenko bending element (shear-flexible stiffness). Translational consistent
mass and a consistent rotary-inertia contribution are included. A rigid disk
may be attached at one node; its polar inertia generates gyroscopic coupling.
Bearing supports are isotropic linear radial springs/dashpots at end nodes.
The Campbell solver uses a damped state-space eigenproblem at each spin speed.
This is a teaching/research model, not a validated industrial rotor solver.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import eig


def _pos(name, value):
    value=float(value)
    if not np.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and > 0")
    return value


def _timoshenko_plane(E, G, A, I, rho, kappa, le):
    """4x4 bending stiffness and mass for [w, rotation, w2, rotation2]."""
    phi=12*E*I/(kappa*G*A*le**2)
    fac=E*I/(le**3*(1+phi))
    K=fac*np.array([[12,6*le,-12,6*le],
        [6*le,(4+phi)*le**2,-6*le,(2-phi)*le**2],
        [-12,-6*le,12,-6*le],
        [6*le,(2-phi)*le**2,-6*le,(4+phi)*le**2]],float)
    # Hermite translational consistent mass + rotary inertia contribution.
    m=rho*A*le/420
    M=m*np.array([[156,22*le,54,-13*le],
        [22*le,4*le**2,13*le,-3*le**2],
        [54,13*le,156,-22*le],
        [-13*le,-3*le**2,-22*le,4*le**2]],float)
    mr=rho*I*le/6
    M[np.ix_([1,3],[1,3])] += mr*np.array([[2,1],[1,2]])
    return K,M


def build_timoshenko_rotor(length_m=1.0, diameter_m=0.04, youngs_modulus_pa=2e11,
        poisson_ratio=0.3, density_kg_m3=7800, elements=8, shear_factor=0.89,
        bearing_stiffness_n_m=2e7, bearing_damping_n_s_m=1200,
        disk_node=None, disk_mass_kg=8.0, disk_polar_inertia_kg_m2=0.035,
        disk_diametral_inertia_kg_m2=0.02):
    """Assemble K,M,G,C; returns matrices and model metadata."""
    L=_pos('length_m',length_m); d=_pos('diameter_m',diameter_m); E=_pos('youngs_modulus_pa',youngs_modulus_pa)
    rho=_pos('density_kg_m3',density_kg_m3); kb=_pos('bearing_stiffness_n_m',bearing_stiffness_n_m)
    cb=float(bearing_damping_n_s_m); nu=float(poisson_ratio); kappa=float(shear_factor)
    if not np.isfinite(cb) or cb<0: raise ValueError('bearing_damping_n_s_m must be finite and >= 0')
    if not 0<nu<0.5: raise ValueError('poisson_ratio must be between 0 and 0.5')
    if not 0<kappa<=1: raise ValueError('shear_factor must be in (0, 1]')
    if int(elements)!=elements or not 2<=int(elements)<=40: raise ValueError('elements must be integer 2..40')
    n=int(elements); nn=n+1; nd=4*nn; le=L/n; A=np.pi*d*d/4; I=np.pi*d**4/64; G=E/(2*(1+nu))
    K=np.zeros((nd,nd)); M=np.zeros_like(K); Gy=np.zeros_like(K); C=np.zeros_like(K)
    # Plane x: [x, slope_y]; plane y: [y, slope_x].
    for e in range(n):
        ke,me=_timoshenko_plane(E,G,A,I,rho,kappa,le)
        for offset in (0,2):
            dofs=np.array([4*e+offset,4*e+offset+1,4*(e+1)+offset,4*(e+1)+offset+1])
            K[np.ix_(dofs,dofs)] += ke; M[np.ix_(dofs,dofs)] += me
    # Isotropic radial bearing spring/dashpot at both end nodes.
    for node in (0,n):
        for dof in (4*node,4*node+2): K[dof,dof]+=kb; C[dof,dof]+=cb
    if disk_node is None: disk_node=n//2
    if int(disk_node)!=disk_node or not 0<=int(disk_node)<=n: raise ValueError('disk_node must be a valid node index')
    md=_pos('disk_mass_kg',disk_mass_kg); jp=_pos('disk_polar_inertia_kg_m2',disk_polar_inertia_kg_m2); jd=_pos('disk_diametral_inertia_kg_m2',disk_diametral_inertia_kg_m2)
    b=4*int(disk_node)
    M[b,b]+=md; M[b+2,b+2]+=md
    M[b+1,b+1]+=jd; M[b+3,b+3]+=jd
    # q=[x, slope_y, y, slope_x]; gyroscopic coupling between disk tilts.
    Gy[b+1,b+3]+=jp; Gy[b+3,b+1]-=jp
    return {'K':K,'M':M,'G':Gy,'C':C,'nodes_m':np.linspace(0,L,nn),'area_m2':A,
            'second_moment_m4':I,'shear_modulus_pa':G,'disk_node':int(disk_node),
            'disk_mass_kg':md,'disk_polar_inertia_kg_m2':jp,'elements':n,
            'bearing_stiffness_n_m':kb,'bearing_damping_n_s_m':cb,
            'theory':'Timoshenko beam; isotropic supports; rigid disk gyroscopic coupling'}


def solve_rotor_campbell(rotor, rpm_values=None):
    """Damped complex eigenanalysis for each RPM; labels forward/backward whirl.

    Returns positive-imaginary eigenvalue branches. Whirl direction is estimated
    from the phase between the disk-node x/y lateral components.
    """
    K,M,G,C=(rotor[k] for k in ('K','M','G','C'))
    if rpm_values is None: rpm_values=np.linspace(0,12000,61)
    rpm=np.asarray(rpm_values,dtype=float)
    if rpm.ndim!=1 or len(rpm)<2 or np.any(~np.isfinite(rpm)) or np.any(rpm<0): raise ValueError('rpm_values must be a finite non-negative 1D array with >=2 values')
    n=K.shape[0]; rows=[]; node=rotor['disk_node']; ix=4*node; iy=ix+2
    MinvK=np.linalg.solve(M,K); MinvC0=np.linalg.solve(M,C); MinvG=np.linalg.solve(M,G)
    Z=np.zeros((n,n)); Id=np.eye(n)
    for speed in rpm:
        omega=speed*2*np.pi/60
        Astate=np.block([[Z,Id],[-MinvK,-(MinvC0+omega*MinvG)]])
        vals,vecs=eig(Astate,check_finite=False)
        for j,lam in enumerate(vals):
            if np.imag(lam)<=1e-7: continue
            q=vecs[:n,j]; phase=np.imag(np.conj(q[ix])*q[iy])
            whirl='Forward' if phase>=0 else 'Backward'
            rows.append({'rpm':float(speed),'frequency_hz':float(abs(np.imag(lam))/(2*np.pi)),
                         'damping_ratio':float(-np.real(lam)/max(abs(lam),1e-15)),
                         'whirl':whirl,'eigenvalue_real':float(np.real(lam)),
                         'eigenvalue_imag':float(np.imag(lam))})
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError('No oscillatory eigenvalue branches found; check supports and parameters')
    # Isotropic models can have exactly repeated lateral-plane eigenvalues.
    out=out.drop_duplicates(subset=['rpm','frequency_hz','whirl']).copy()
    return out.sort_values(['rpm','frequency_hz']).reset_index(drop=True)


def critical_speed_crossings(campbell: pd.DataFrame, order=1):
    """Estimate intersections of whirl branches with N× synchronous line by interpolation."""
    if not {'rpm','frequency_hz','whirl'}.issubset(campbell.columns): raise ValueError('Campbell data missing required columns')
    order=_pos('order',order); target=order/60.0; results=[]
    for whirl,grp in campbell.groupby('whirl'):
        # Track each branch by sorted frequency rank at each speed (branch tracking is approximate).
        for rank in range(grp.groupby('rpm').size().min()):
            series=grp.sort_values(['rpm','frequency_hz']).groupby('rpm').nth(rank).reset_index()
            x=series['frequency_hz'].to_numpy()-target*series['rpm'].to_numpy()
            r=series['rpm'].to_numpy()
            for i in range(len(x)-1):
                if x[i]==0 or x[i]*x[i+1]<0:
                    frac=0 if x[i]==0 else -x[i]/(x[i+1]-x[i])
                    results.append({'whirl':whirl,'order':int(order),'critical_speed_rpm':float(r[i]+frac*(r[i+1]-r[i]))})
    return pd.DataFrame(results).drop_duplicates() if results else pd.DataFrame(columns=['whirl','order','critical_speed_rpm'])
