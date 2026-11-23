"""V21 3D Engineering Atlas.

Interactive 3D visualizations for the major capabilities accumulated in the
v7-v20 predictive-maintenance project. Plots are based on current synthetic
telemetry and reproducible educational numerical models; they are not certified
CAD/FEA or field measurements.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from signal_diagnostics import generate_vibration_signal, calculate_spectrum, bearing_characteristic_frequencies
from advanced_diagnostics import signal_features, envelope_spectrum, rotational_order_spectrum, spectral_band_energy
from v18_fem_rotor import solve_rotor_modes
from v19_advanced_rotor import build_timoshenko_rotor, solve_rotor_campbell
from v20_bearing_response import synchronous_unbalance_response, stability_sweep


def _finish(fig, title, x, y, z):
    fig.update_layout(title=title, scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z,
        aspectmode="auto", camera=dict(eye=dict(x=1.55, y=1.55, z=1.15))),
        margin=dict(l=0, r=0, t=48, b=0), height=430, showlegend=True)
    return fig


def _line3d(x, y, z, name, color=None, width=5, mode="lines"):
    line = dict(width=width)
    if color is not None: line["color"] = color
    return go.Scatter3d(x=x, y=y, z=z, mode=mode, name=name, line=line)


def build_3d_atlas(raw: pd.DataFrame, scored: pd.DataFrame, importance: pd.DataFrame,
                   X_test, probabilities, seed: int = 2026, y_test=None):
    """Return category -> list[(figure title, Plotly figure, provenance note)]."""
    rng = np.random.default_rng(int(seed))
    categories: dict[str, list] = {}
    def add(group, title, fig, note="Synthetic/demo visualization"):
        categories.setdefault(group, []).append((title, fig, note))

    # 1. Fleet health, anomaly, ML, prioritization and maintenance intelligence.
    if not scored.empty:
        priority_colors = {"Critical":"#d62728", "High":"#ff7f0e", "Moderate":"#e6b800", "Low":"#2ca02c"}
        f = go.Figure()
        for band, g in scored.groupby("priority"):
            f.add_trace(go.Scatter3d(x=g.temperature_c, y=g.vibration_rms, z=g.health_index,
                mode="markers+text", text=g.machine_id, name=str(band),
                marker=dict(size=7, color=priority_colors.get(str(band), "gray"), opacity=.9)))
        add("Fleet • AI • Operations", "01 • Fleet Health Digital Twin", _finish(f,"01 • Fleet Health Digital Twin","Temperature (°C)","Vibration RMS","Health index (0–100)"),
            "Uses current scored synthetic fleet telemetry.")

        f = go.Figure(go.Scatter3d(x=scored.temperature_c, y=scored.vibration_rms,
            z=scored.synthetic_risk_probability_pct, mode="markers+text", text=scored.machine_id,
            marker=dict(size=7, color=scored.synthetic_risk_probability_pct, colorscale="Turbo", colorbar=dict(title="Risk %"))))
        add("Fleet • AI • Operations", "02 • AI Risk Probability Space", _finish(f,"02 • AI Risk Probability Space","Temperature (°C)","Vibration RMS","Synthetic risk probability (%)"),
            "Risk probabilities come from the existing synthetic-label ML model.")

        f = go.Figure(go.Scatter3d(x=scored.health_index, y=scored.enterprise_priority_score,
            z=scored.synthetic_risk_probability_pct, mode="markers+text", text=scored.machine_id,
            marker=dict(size=7, color=scored["rank"], colorscale="Plasma", colorbar=dict(title="Rank"))))
        add("Fleet • AI • Operations", "03 • Maintenance Priority Cube", _finish(f,"03 • Maintenance Priority Cube","Health index","Priority score","Synthetic risk (%)"))

        if "iqr_anomaly" in scored:
            f = go.Figure(go.Scatter3d(x=scored.rpm, y=scored.load_pct, z=scored.iqr_anomaly,
                mode="markers+text", text=scored.machine_id,
                marker=dict(size=8, color=scored.iqr_anomaly, colorscale="Reds", cmin=0, cmax=1,
                            colorbar=dict(title="IQR anomaly"))))
            add("Fleet • AI • Operations", "04 • Sensor Anomaly Map", _finish(f,"04 • Sensor Anomaly Map","RPM","Load (%)","IQR anomaly flag"))

    # ML validation decision-space visualization.
    try:
        X = X_test.copy() if hasattr(X_test, "copy") else pd.DataFrame(X_test)
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        cols = list(X.columns[:3])
        while len(cols) < 3:
            X[f"feature_{len(cols)+1}"] = 0.0; cols.append(f"feature_{len(cols)}")
        p = np.asarray(probabilities, dtype=float)
        f = go.Figure(go.Scatter3d(x=X[cols[0]], y=X[cols[1]], z=X[cols[2]], mode="markers",
            marker=dict(size=3, color=p, colorscale="Turbo", colorbar=dict(title="Predicted risk"), opacity=.8),
            text=[f"Predicted risk: {v:.3f}" for v in p], name="Holdout samples"))
        add("Fleet • AI • Operations", "05 • ML Holdout Decision Space", _finish(f,"05 • ML Holdout Decision Space",cols[0],cols[1],cols[2]),
            "Three first feature columns from the held-out synthetic dataset; color is model risk probability.")
        if y_test is not None:
            actual=np.asarray(y_test,dtype=int); predicted=(p>=0.5).astype(int)
            counts=np.zeros((2,2),dtype=int)
            for aa,pp in zip(actual,predicted): counts[int(aa),int(pp)]+=1
            xx=[]; yy=[]; zz=[]; labels=[]
            for aa in range(2):
                for pp in range(2):
                    xx.append(aa); yy.append(pp); zz.append(int(counts[aa,pp])); labels.append(f"Actual {aa} / Predicted {pp}: {counts[aa,pp]}")
            f=go.Figure(go.Scatter3d(x=xx,y=yy,z=zz,mode="markers+text",text=labels,
                marker=dict(size=[max(6,8+v/12) for v in zz],color=zz,colorscale="Blues",colorbar=dict(title="Count"))))
            add("Fleet • AI • Operations", "07 • 3D Confusion Matrix", _finish(f,"07 • 3D Confusion Matrix","Actual class (0/1)","Predicted class (0/1)","Sample count"),
                "Confusion counts are calculated from the held-out synthetic test split.")
    except Exception:
        pass

    if not importance.empty:
        f = go.Figure(go.Scatter3d(x=np.arange(len(importance)), y=importance.importance_mean,
            z=importance.importance_std, mode="markers+text", text=importance.feature,
            marker=dict(size=10, color=importance.importance_mean, colorscale="Viridis", colorbar=dict(title="Mean importance"))))
        add("Fleet • AI • Operations", "06 • 3D Model Explainability", _finish(f,"06 • 3D Model Explainability","Feature index","Mean permutation importance","Importance standard deviation"),
            "Permutation importance is model sensitivity, not causal attribution.")

    # Shared reproducible signal for FFT, orbit, envelope, order, features and bearing.
    rpm = float(pd.to_numeric(raw.rpm, errors="coerce").median()) if not raw.empty else 1800.0
    rpm = max(rpm, 60.0); shaft_hz = rpm/60.0; fs = 2048
    t, sig = generate_vibration_signal(4.0, fs, shaft_hz, min(90.0, fs/4), seed=int(seed)+311)
    freq, amp = calculate_spectrum(sig, fs)
    # 3D rotating shaft / orbit trajectory.
    tx = np.linspace(0, 4*np.pi, 180); axial = np.linspace(0, 1, 60)
    A, T = np.meshgrid(axial, tx)
    center_x = .012*np.sin(2*np.pi*T/2 + .3*A); center_y = .009*np.cos(2*np.pi*T/2 + .3*A)
    theta = np.linspace(0,2*np.pi,48); TH, AA = np.meshgrid(theta, axial)
    rad=.025; sx=center_x[:60,:] if False else rad*np.cos(TH)
    sy=rad*np.sin(TH); sz=AA
    f=go.Figure(go.Surface(x=sx,y=sy,z=sz,surfacecolor=np.sin(TH*6+AA*4),colorscale="Turbo",showscale=False,opacity=.82,name="Shaft body"))
    f.add_trace(_line3d(center_x[:,30],center_y[:,30],tx/(4*np.pi),"Orbit centreline","#f7f7f7",4))
    add("Vibration • Bearings", "07 • 3D Shaft Geometry & Orbit", _finish(f,"07 • 3D Shaft Geometry & Orbit","X displacement / radius (m)","Y displacement / radius (m)","Normalized axial/time coordinate"),
        "Shaft body is a geometric illustration; orbit path is generated from a deterministic synthetic trajectory.")

    # FFT: frequency x harmonic order x amplitude.
    orders=np.arange(1,9); f0=shaft_hz
    sel=freq<=min(500,fs/2); fx=freq[sel]; ax=amp[sel]
    f=go.Figure()
    for k in orders:
        target=k*f0; idx=int(np.argmin(np.abs(fx-target)))
        f.add_trace(go.Scatter3d(x=[target],y=[k],z=[ax[idx]],mode="markers+text",text=[f"{k}X"],name=f"{k}X",marker=dict(size=6)))
    add("Vibration • Bearings", "08 • FFT Harmonic Peak Map", _finish(f,"08 • FFT Harmonic Peak Map","Frequency (Hz)","Shaft harmonic order","FFT amplitude (a.u.)"),
        "Markers sample the existing synthetic FFT near each integer shaft order.")

    # 3D time-frequency-amplitude STFT.
    frame=256; hop=64; win=np.hanning(frame); rows=[]; times=[]
    for start in range(0,len(sig)-frame+1,hop):
        chunk=sig[start:start+frame]; rows.append(np.abs(np.fft.rfft((chunk-np.mean(chunk))*win))); times.append((start+frame/2)/fs)
    spec=np.asarray(rows).T; sf=np.fft.rfftfreq(frame,1/fs); mask=sf<=300
    f=go.Figure(go.Surface(x=np.asarray(times),y=sf[mask],z=spec[mask,:],colorscale="Viridis",colorbar=dict(title="Magnitude")))
    add("Vibration • Bearings", "09 • 3D STFT Spectrogram", _finish(f,"09 • 3D STFT Spectrogram","Time (s)","Frequency (Hz)","STFT magnitude"))

    # Thermal field: 3D spatial temperature surface for thermal-stress capability.
    gx=np.linspace(-2.5,2.5,48); gy=np.linspace(-1.8,1.8,40); GX,GY=np.meshgrid(gx,gy)
    temp_field=34+28*np.exp(-((GX-.65)**2/.65+(GY+.25)**2/.32))+15*np.exp(-((GX+1.25)**2/.45+(GY-.65)**2/.38))
    f=go.Figure(go.Surface(x=GX,y=GY,z=temp_field,colorscale="Hot",colorbar=dict(title="°C")))
    add("Vibration • Bearings", "10 • 3D Thermal Hotspot Field", _finish(f,"10 • 3D Thermal Hotspot Field","Machine surface X (m)","Machine surface Y (m)","Temperature (°C)"),
        "Illustrative spatial heat field; telemetry temperature is a scalar per asset, not a measured thermal image.")

    # 3D bearing geometry: race rings + rolling elements, with characteristic-frequency points.
    nballs=8; ball_r=.065; pitch_r=.27; z0=0.0
    f=go.Figure()
    for radius,zv,label in [(0.36,-.08,"Outer race"),(0.18,.08,"Inner race")]:
        th=np.linspace(0,2*np.pi,120)
        f.add_trace(go.Scatter3d(x=radius*np.cos(th),y=radius*np.sin(th),z=np.full_like(th,zv),mode="lines",name=label,line=dict(width=9)))
    th=np.linspace(0,2*np.pi,80)
    for j in range(nballs):
        a=2*np.pi*j/nballs; cx=pitch_r*np.cos(a); cy=pitch_r*np.sin(a)
        u=np.linspace(0,2*np.pi,16); v=np.linspace(0,np.pi,12); U,V=np.meshgrid(u,v)
        f.add_trace(go.Surface(x=cx+ball_r*np.cos(U)*np.sin(V),y=cy+ball_r*np.sin(U)*np.sin(V),z=z0+ball_r*np.cos(V),showscale=False,opacity=.85,name=f"Rolling element {j+1}",showlegend=(j==0)))
    bearing=bearing_characteristic_frequencies(rpm,8,10,50,0)
    for j,(name,val) in enumerate(bearing.items()):
        f.add_trace(go.Scatter3d(x=[0.48+0.05*j],y=[0],z=[0.12],mode="markers+text",text=[f"{name}: {val:.1f} Hz"],name=name,marker=dict(size=5)))
    add("Vibration • Bearings", "10 • Rolling Bearing 3D Assembly & Fault Map", _finish(f,"10 • Rolling Bearing 3D Assembly & Fault Map","Bearing radial X (normalized)","Bearing radial Y (normalized)","Axial Z (normalized)"),
        "Idealized bearing geometry with theoretical FTF/BSF/BPFO/BPFI annotations; not a CAD bearing model.")

    # Envelope spectrum across signal windows.
    env_rows=[]; env_times=[]; env_freq=None
    for start in range(0,len(sig)-1024+1,512):
        ef,ea=envelope_spectrum(sig[start:start+1024],fs); env_freq=ef; env_rows.append(ea); env_times.append((start+512)/fs)
    env_mat=np.asarray(env_rows)
    f=go.Figure(go.Surface(x=np.asarray(env_times),y=np.asarray(env_freq),z=env_mat.T,colorscale="Plasma"))
    add("Vibration • Bearings", "11 • 3D Envelope Spectrum Evolution", _finish(f,"11 • 3D Envelope Spectrum Evolution","Window centre time (s)","Envelope frequency (Hz)","Envelope amplitude"))

    # Order spectrum as RPM x order x amplitude (multiple synthetic RPM slices).
    rpm_slices=np.linspace(max(300,rpm*.65),rpm*1.35,12); order_rows=[]; ord_grid=np.linspace(0,8,160)
    for rr in rpm_slices:
        _,aa=rotational_order_spectrum(sig,fs,rr); ff,_=calculate_spectrum(sig,fs)
        oo=ff/(rr/60); order_rows.append(np.interp(ord_grid,oo,aa,left=0,right=0))
    f=go.Figure(go.Surface(x=ord_grid,y=rpm_slices,z=np.asarray(order_rows),colorscale="Viridis",colorbar=dict(title="Amplitude")))
    add("Vibration • Bearings", "12 • 3D Rotational Order Map", _finish(f,"12 • 3D Rotational Order Map","Rotational order","RPM sweep","Order amplitude"),
        "Order mapping uses frequency/RPM conversion; this is not tachometer angular resampling.")

    # Signal condition indicators over successive windows.
    feat_rows=[]; seg_ids=[]
    for j,start in enumerate(range(0,len(sig)-512+1,256)):
        feat_rows.append(signal_features(sig[start:start+512],fs)); seg_ids.append(j)
    fd=pd.DataFrame(feat_rows)
    f=go.Figure(go.Scatter3d(x=fd["RMS"],y=fd["Crest Factor"],z=fd["Kurtosis"],mode="markers+lines",
        text=[f"Window {v}" for v in seg_ids],marker=dict(size=5,color=seg_ids,colorscale="Turbo",colorbar=dict(title="Window"))))
    add("Vibration • Bearings", "13 • 3D Signal Condition Indicator Trajectory", _finish(f,"13 • 3D Signal Condition Indicator Trajectory","RMS","Crest factor","Kurtosis"))

    # Spectral band energy 3D point chart.
    bands=spectral_band_energy(sig,fs); names=list(bands); vals=list(bands.values())
    f=go.Figure(go.Scatter3d(x=np.arange(len(names)),y=vals,z=np.asarray(vals)**0.5,mode="markers+text",text=names,marker=dict(size=12,color=vals,colorscale="Viridis",colorbar=dict(title="Energy fraction"))))
    add("Vibration • Bearings", "14 • 3D Spectral Band Energy", _finish(f,"14 • 3D Spectral Band Energy","Band index","Normalized energy fraction","Square-root energy"))

    # Campbell / gyroscopic branches, Bode and waterfall from V17/V19/V20 engines.
    rotor=build_timoshenko_rotor(length_m=1.0,diameter_m=.04,elements=4,bearing_stiffness_n_m=2e7,
        bearing_damping_n_s_m=1200,disk_mass_kg=8,disk_polar_inertia_kg_m2=.035,disk_diametral_inertia_kg_m2=.02)
    rpm_grid=np.linspace(0,9000,19)
    camp=solve_rotor_campbell(rotor,rpm_grid)
    f=go.Figure()
    for whirl,color in [("Forward","royalblue"),("Backward","firebrick")]:
        q=camp[camp.whirl==whirl]
        if not q.empty: f.add_trace(go.Scatter3d(x=q.rpm,y=q.frequency_hz,z=q.damping_ratio,mode="markers",name=whirl,marker=dict(size=3,color=color)))
    add("Rotor • FEM • Dynamics", "15 • 3D Gyroscopic Campbell Branches", _finish(f,"15 • 3D Gyroscopic Campbell Branches","Rotor speed (RPM)","Damped frequency (Hz)","Modal damping ratio"),
        "Branches are computed by the project's linear Timoshenko/gyroscopic eigenmodel; branch identity tracking remains approximate.")

    # Bode response from existing simplified SDOF engine represented in 3D.
    rr=np.linspace(300,9000,80); ratio=(rr/60)/30.0; zeta=.04
    den=np.sqrt((1-ratio**2)**2+(2*zeta*ratio)**2); bode_amp=ratio**2/den; bode_phase=np.degrees(np.arctan2(2*zeta*ratio,1-ratio**2))
    f=go.Figure(go.Scatter3d(x=rr,y=bode_amp,z=bode_phase,mode="lines",line=dict(width=7,color=bode_amp,colorscale="Turbo",colorbar=dict(title="Amplitude"))))
    add("Rotor • FEM • Dynamics", "16 • 3D Bode Amplitude–Phase Response", _finish(f,"16 • 3D Bode Amplitude–Phase Response","RPM","Normalized amplitude","Phase (degrees)"),
        "Uses the existing simplified SDOF response formulation.")

    # 3D waterfall spectral slices.
    wf_rpm=np.linspace(600,9000,12); wf_freq=np.linspace(0,300,120); wf=[]
    for rr0 in wf_rpm:
        sig0=.8*np.sin(2*np.pi*(rr0/60)*np.arange(2048)/fs)+.16*np.sin(2*np.pi*(2*rr0/60)*np.arange(2048)/fs)+.04*rng.normal(size=2048)
        ff,aa=calculate_spectrum(sig0,fs); wf.append(np.interp(wf_freq,ff,aa,left=0,right=0))
    f=go.Figure(go.Surface(x=wf_freq,y=wf_rpm,z=np.asarray(wf),colorscale="Viridis",colorbar=dict(title="FFT amplitude")))
    add("Rotor • FEM • Dynamics", "17 • 3D RPM Waterfall Spectrum", _finish(f,"17 • 3D RPM Waterfall Spectrum","Frequency (Hz)","RPM","FFT amplitude"))

    # FE modal shapes from Euler-Bernoulli engine.
    fe=solve_rotor_modes(length_m=1.0,diameter_m=.04,elements=8,modes=4)
    xnodes=fe["nodes_m"]; shapes=fe["mode_shapes"]
    f=go.Figure()
    for m in range(min(4,shapes.shape[1])):
        f.add_trace(_line3d(xnodes,np.full_like(xnodes,m+1),shapes[:,m],f"Mode {m+1}: {fe['frequencies_hz'][m]:.1f} Hz",width=6))
    add("Rotor • FEM • Dynamics", "18 • 3D FE Modal Shape Family", _finish(f,"18 • 3D FE Modal Shape Family","Axial position (m)","Mode number","Normalized lateral mode shape"),
        "Mode shapes from the included Euler–Bernoulli FE solver.")

    # Timoshenko shaft mesh with illustrative bending envelope from first FE mode.
    xx=rotor["nodes_m"]; mode=np.sin(np.pi*xx/max(xx[-1],1e-9)); th=np.linspace(0,2*np.pi,36); XX,TH=np.meshgrid(xx,th)
    bend=.012*mode[None,:]
    f=go.Figure(go.Surface(x=XX,y=.02*np.cos(TH),z=.02*np.sin(TH)+bend,colorscale="Viridis",showscale=False,opacity=.9))
    f.add_trace(_line3d(xx,np.zeros_like(xx),bend[0],"Bending centreline","red",5))
    add("Rotor • FEM • Dynamics", "19 • 3D Timoshenko Shaft / Bending Envelope", _finish(f,"19 • 3D Timoshenko Shaft / Bending Envelope","Axial position (m)","Y radius (m)","Z + bending displacement (m)"),
        "Geometric rendering uses the current Timoshenko mesh and an illustrative first-mode envelope; not a full 3D solid FE mesh.")

    # Disk mass / inertia geometry.
    th=np.linspace(0,2*np.pi,80); rr0=np.linspace(.02,.18,18); TH,RR=np.meshgrid(th,rr0)
    f=go.Figure(go.Surface(x=RR*np.cos(TH),y=RR*np.sin(TH),z=np.zeros_like(RR),colorscale="Blues",showscale=False))
    f.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers+text",text=[f"Disk {rotor['disk_mass_kg']:.1f} kg"],marker=dict(size=8,color="orange")))
    add("Rotor • FEM • Dynamics", "20 • 3D Concentrated Disk Mass & Inertia", _finish(f,"20 • 3D Concentrated Disk Mass & Inertia","Radial X (m)","Radial Y (m)","Axial Z (m)"),
        "Disk is represented as a rigid concentrated mass with polar and diametral inertia in the numerical model.")

    # Bearing coefficient response surface: displacement direction and force magnitude.
    disp=np.linspace(-50e-6,50e-6,25); dx,dy=np.meshgrid(disp,disp)
    kxx=2.0e7; kyy=2.2e7; kxy=1.5e6; kyx=-1.0e6
    fx=kxx*dx+kxy*dy; fy=kyx*dx+kyy*dy; mag=np.sqrt(fx**2+fy**2)
    f=go.Figure(go.Surface(x=dx*1e6,y=dy*1e6,z=mag,colorscale="Turbo",colorbar=dict(title="Force (N)")))
    add("Rotor • FEM • Dynamics", "21 • 3D Linearized Bearing Stiffness Map", _finish(f,"21 • 3D Linearized Bearing Stiffness Map","X displacement (µm)","Y displacement (µm)","Radial force magnitude (N)"),
        "Uses an illustrative anisotropic/cross-coupled coefficient matrix; replace with measured or validated coefficients.")

    # Cross-coupled force vector field.
    f=go.Figure()
    for a in np.linspace(0,2*np.pi,12,endpoint=False):
        x0=30*np.cos(a); y0=30*np.sin(a); fx0=(kxx*x0+kxy*y0); fy0=(kyx*x0+kyy*y0)
        scale=1e-6
        f.add_trace(go.Scatter3d(x=[x0,x0+fx0*scale],y=[y0,y0+fy0*scale],z=[0,0.5],mode="lines+markers",showlegend=False,line=dict(width=4)))
    add("Rotor • FEM • Dynamics", "22 • 3D Cross-Coupled Bearing Force Field", _finish(f,"22 • 3D Cross-Coupled Bearing Force Field","X displacement (µm)","Y displacement (µm)","Illustrative axial offset"))

    # Unbalance response and stability sweeps.
    urpm=np.linspace(300,9000,24); unb=synchronous_unbalance_response(rotor,urpm,unbalance_kg_m=1e-4)
    f=go.Figure(go.Scatter3d(x=unb.rpm,y=unb.resultant_amplitude_m*1e6,z=unb.x_phase_deg,mode="lines+markers",marker=dict(size=4,color=unb.resultant_amplitude_m,colorscale="Turbo",colorbar=dict(title="Response (m)"))))
    add("Rotor • FEM • Dynamics", "23 • 3D Synchronous Unbalance Response", _finish(f,"23 • 3D Synchronous Unbalance Response","RPM","Resultant displacement (µm)","X response phase (degrees)"),
        "Computed from the V20 steady-state harmonic response solver.")

    stab=stability_sweep(rotor,np.linspace(0,9000,18))
    f=go.Figure(go.Scatter3d(x=stab.rpm,y=stab.least_damped_frequency_hz,z=stab.max_real_eigenvalue_per_s,mode="lines+markers",marker=dict(size=5,color=stab.max_real_eigenvalue_per_s,colorscale="RdYlGn_r",colorbar=dict(title="Max Re(λ)"))))
    add("Rotor • FEM • Dynamics", "24 • 3D Rotor Stability Map", _finish(f,"24 • 3D Rotor Stability Map","RPM","Least-damped frequency (Hz)","Maximum eigenvalue real part (1/s)"),
        "Linearized state-space screening from the V20 model; positive real part indicates instability in this model.")

    # FE vs experimental modal comparison: do not fabricate experimental data.
    f=go.Figure()
    f.add_trace(go.Scatter3d(x=np.arange(1,len(fe["frequencies_hz"])+1),y=fe["frequencies_hz"],z=np.zeros(len(fe["frequencies_hz"])),mode="markers+lines+text",text=["Numerical"]*len(fe["frequencies_hz"]),name="Numerical FE"))
    f.add_trace(go.Scatter3d(x=[],y=[],z=[],mode="markers",name="Experimental CSV (not uploaded)"))
    add("Rotor • FEM • Dynamics", "25 • 3D Numerical vs Experimental Modal Comparison", _finish(f,"25 • 3D Numerical vs Experimental Modal Comparison","Mode number","Numerical frequency (Hz)","Experimental frequency (Hz)"),
        "Experimental series is intentionally empty until the user uploads measured modal-test results; no measurement is invented.")

    # CSV audit quality profile: actual telemetry channel values, normalized for a common 3D view.
    sensor_cols=[c for c in ["rpm","temperature_c","vibration_rms","load_pct"] if c in raw]
    if sensor_cols:
        norm=raw[sensor_cols].apply(pd.to_numeric,errors="coerce")
        norm=(norm-norm.min())/(norm.max()-norm.min()).replace(0,1)
        f=go.Figure()
        for j,col in enumerate(sensor_cols):
            f.add_trace(go.Scatter3d(x=np.arange(len(raw)),y=np.full(len(raw),j),z=norm[col],mode="markers+lines",name=col))
        add("Data Quality • Reporting", "26 • 3D Sensor Data Quality Profile", _finish(f,"26 • 3D Sensor Data Quality Profile","Telemetry row","Sensor channel","Min-max normalized value"),
            "Shows current synthetic telemetry channels; uploaded CSV quality validation remains in the v17 tab.")

    return categories
