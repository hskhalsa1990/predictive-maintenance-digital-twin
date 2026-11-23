from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from monitoring import generate_sensor_batch
from v8_engine import add_batch_id, validate_sensor_frame
from v10_enterprise import score_enterprise, make_work_order_queue, compare_batches
from v11_validation import evaluate_synthetic_model
from signal_diagnostics import (generate_vibration_signal, calculate_spectrum,
    bearing_characteristic_frequencies, orbit_coordinates)
from advanced_diagnostics import (signal_features, envelope_spectrum,
    rotational_order_spectrum, spectral_band_energy)
from v17_rotor_lab import campbell_data, bode_unbalance_response, waterfall_spectra, validate_sensor_csv
from v18_fem_rotor import solve_rotor_modes, compare_experimental_modes
from v19_advanced_rotor import build_timoshenko_rotor, solve_rotor_campbell, critical_speed_crossings
from v20_bearing_response import apply_linearized_bearing_coefficients, synchronous_unbalance_response, stability_sweep
from v21_3d_atlas import build_3d_atlas

st.set_page_config(page_title="Industrial AI Reliability Suite v21", layout="wide", page_icon="🏭")
st.title("Industrial AI Reliability Suite | v21")
st.caption("Synthetic fleet twin • ML validation • Explainability • Interactive 3D engineering visualization")
@st.cache_data(show_spinner="Training reproducible demo model and evaluating holdout...")
def validation(seed): return evaluate_synthetic_model(int(seed))
with st.sidebar:
    st.header("Simulation Settings")
    fleet=st.slider("Fleet size",3,30,12)
    seed=st.number_input("Random seed",0,999999,2026)
    scenario=st.selectbox("Stress scenario",["Normal","Vibration stress","Thermal stress","Combined stress"])
    batch=st.number_input("Synthetic batch",0,9999,0)
    st.caption("All telemetry is simulated. No live equipment connection.")
raw=generate_sensor_batch(int(batch),machines=fleet,seed=int(seed))
if scenario in ["Vibration stress","Combined stress"]: raw["vibration_rms"]*=1.5
if scenario in ["Thermal stress","Combined stress"]: raw["temperature_c"]+=14
raw=add_batch_id(raw,int(batch))
audit=validate_sensor_frame(raw)
scored=score_enterprise(raw,int(seed)); queue=make_work_order_queue(scored)
model,metrics,cm,importance,X_test,y_test,prob=validation(int(seed))
health_avg=float(scored.health_index.mean()) if len(scored) else 0
critical=int((scored.priority=="Critical").sum()); high=int((scored.priority=="High").sum())
cols=st.columns(5)
cols[0].metric("Mean Health",f"{health_avg:.1f}/100")
cols[1].metric("Critical Assets",critical)
cols[2].metric("High Priority",high)
cols[3].metric("Draft Work Orders",len(queue))
cols[4].metric("Sensor Audit",audit.get("status","N/A"))
tabs=st.tabs(["Executive", "Asset Ranking", "Model Validation", "Explainability", "Work Orders & Data", "3D Engineering Lab", "Engineering Diagnostic Charts", "Advanced Diagnostics v15", "Rotor Dynamics v17", "FE Rotor & Modal Lab v18", "Timoshenko Rotor Lab v19", "Bearing & Unbalance Lab v20", "Complete 3D Engineering Atlas v21"])
with tabs[0]:
    c1,c2=st.columns(2)
    c1.plotly_chart(px.bar(scored,x="machine_id",y="health_index",color="priority",range_y=[0,100],title="Machine Health Index"),use_container_width=True)
    c2.plotly_chart(px.pie(scored,names="priority",title="Fleet Priority Distribution"),use_container_width=True)
    st.subheader("Top Risk Assets")
    st.dataframe(scored.head(5)[["machine_id","priority","enterprise_priority_score","health_index","synthetic_risk_probability_pct","recommended_action"]],use_container_width=True,hide_index=True)
with tabs[1]:
    st.plotly_chart(px.bar(scored,x="machine_id",y="enterprise_priority_score",color="priority",title="Transparent Composite Priority Score"),use_container_width=True)
    st.dataframe(scored,use_container_width=True,hide_index=True)
with tabs[2]:
    st.warning("Evaluation is on a held-out synthetic dataset with rule-generated labels. High metrics indicate learning the artificial rule—not verified industrial failure prediction.")
    st.write("Holdout metrics (25% test split)")
    st.columns(5)
    metric_cols=st.columns(5)
    for col,(name,val) in zip(metric_cols,metrics.items()): col.metric(name,f"{val:.3f}")
    fig=go.Figure(data=go.Heatmap(z=cm,x=["Predicted normal","Predicted risk"],y=["Actual normal","Actual risk"],text=cm,texttemplate="%{text}",colorscale="Blues",showscale=False))
    fig.update_layout(title="Confusion Matrix",xaxis_title="Prediction",yaxis_title="Ground-truth synthetic label")
    st.plotly_chart(fig,use_container_width=True)
with tabs[3]:
    st.caption("Permutation importance measures performance drop when a feature is shuffled. It is model-level explanation, not causality.")
    st.plotly_chart(px.bar(importance,x="importance_mean",y="feature",orientation="h",error_x="importance_std",title="Permutation Feature Importance (F1)"),use_container_width=True)
    st.dataframe(importance,use_container_width=True,hide_index=True)
with tabs[4]:
    st.info("Draft work orders only—review with qualified personnel. No automatic dispatch or maintenance instruction.")
    st.dataframe(queue,use_container_width=True,hide_index=True)
    st.download_button("Export Work Orders CSV",queue.to_csv(index=False).encode(),"v11_draft_work_orders.csv","text/csv")
    st.subheader("Sensor Quality Audit")
    st.json(audit)
    st.download_button("Export Scored Fleet CSV",scored.to_csv(index=False).encode(),"v11_scored_fleet.csv","text/csv")
    st.download_button("Export Telemetry CSV",raw.to_csv(index=False).encode(),"v11_synthetic_telemetry.csv","text/csv")

with tabs[5]:
    st.header("Interactive 3D Engineering Visualization Lab")
    st.caption("Conceptual digital-twin and sensor visualizations using synthetic data—not CAD/FEA results.")
    t = np.linspace(0, 2*np.pi, 80)
    z = np.linspace(-2, 2, 35)
    T, Z = np.meshgrid(t, z)
    # Shaft surface: cylindrical geometry with a subtle vibration displacement.
    vib_amp = float(raw.vibration_rms.mean()) if "vibration_rms" in raw else 1.0
    R = 0.55 + 0.035*np.sin(6*T + 2*Z)*min(vib_amp, 5)
    X, Y = R*np.cos(T), R*np.sin(T)
    shaft = go.Figure(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.sin(6*T+Z), colorscale="Turbo", showscale=True, colorbar_title="Vibration pattern"))
    shaft.update_layout(title="1 • Conceptual Rotating Shaft / Vibration Pattern", scene=dict(aspectmode="data", xaxis_title="X",yaxis_title="Y",zaxis_title="Axial position"), margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(shaft, use_container_width=True)

    c1,c2=st.columns(2)
    # 3D vibration surface over time and frequency (synthetic signal).
    time=np.linspace(0,10,100); freq=np.linspace(0,50,60); TT,FF=np.meshgrid(time,freq)
    VV=(0.25+0.015*TT)*np.sin(2*np.pi*FF/18 + TT)*np.exp(-FF/65) + 0.12*np.sin(2*np.pi*FF/7)
    vibration=go.Figure(go.Surface(x=TT,y=FF,z=VV,colorscale="Viridis",colorbar_title="Amplitude"))
    vibration.update_layout(title="2 • Vibration Surface (Time × Frequency)",scene=dict(xaxis_title="Time (s)",yaxis_title="Frequency (Hz)",zaxis_title="Relative amplitude"),margin=dict(l=0,r=0,t=45,b=0))
    c1.plotly_chart(vibration,use_container_width=True)
    # Thermal spatial field with multiple synthetic hot spots.
    gx=np.linspace(-3,3,70); gy=np.linspace(-2,2,55); GX,GY=np.meshgrid(gx,gy)
    base=35 + 32*np.exp(-((GX-0.8)**2/0.7+(GY+0.2)**2/0.35)) + 18*np.exp(-((GX+1.5)**2/0.5+(GY-0.8)**2/0.45))
    heat=go.Figure(go.Surface(x=GX,y=GY,z=base,colorscale="Hot",colorbar_title="°C"))
    heat.update_layout(title="3 • Synthetic Thermal Field / Hotspots",scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Temperature (°C)"),margin=dict(l=0,r=0,t=45,b=0))
    c2.plotly_chart(heat,use_container_width=True)

    # Fleet risk landscape: three sensor/health dimensions.
    risk=go.Figure()
    palette={"Critical":"red","High":"orange","Moderate":"gold","Low":"green"}
    for label,grp in scored.groupby("priority"):
        risk.add_trace(go.Scatter3d(x=grp.temperature_c,y=grp.vibration_rms,z=grp.synthetic_risk_probability_pct,mode="markers+text",text=grp.machine_id,textposition="top center",name=str(label),marker=dict(size=7,color=palette.get(str(label),"gray"))))
    risk.update_layout(title="4 • Fleet Risk Space",scene=dict(xaxis_title="Temperature (°C)",yaxis_title="Vibration RMS",zaxis_title="Synthetic risk (%)"),margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(risk,use_container_width=True)

    # Conceptual health degradation path.
    steps=np.arange(0,60); health_path=np.clip(100-0.42*steps-0.003*steps**2,0,100); vibration_path=0.7+0.018*steps+0.002*steps**2; thermal_path=40+0.12*steps+0.008*steps**2
    trajectory=go.Figure(go.Scatter3d(x=steps,y=vibration_path,z=thermal_path,mode="lines+markers",marker=dict(color=health_path,colorscale="RdYlGn",size=4,colorbar=dict(title="Health score")),line=dict(width=6)))
    trajectory.update_layout(title="5 • Conceptual Degradation Trajectory",scene=dict(xaxis_title="Simulation step",yaxis_title="Vibration RMS",zaxis_title="Temperature (°C)"),margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(trajectory,use_container_width=True)
    st.info("All five 3D plots are interactive (rotate, zoom, hover). The surfaces and trajectory are illustrative simulations, not physical validation or finite-element analysis.")


with tabs[6]:
    st.header("Advanced Signal Diagnostics")
    st.caption("Four requested engineering diagrams: FFT spectrum, 3D shaft orbit, 3D spectrogram and 2D bearing fault-frequency diagnosis.")
    selected = st.selectbox("Diagnostic asset", raw.machine_id.tolist(), key="diag_asset_v14")
    asset = raw.loc[raw.machine_id == selected].iloc[0]
    rpm = float(asset.rpm)
    shaft_hz = rpm / 60.0
    duration = 4.0
    fs = 2048
    t_sig, sig = generate_vibration_signal(duration, fs, shaft_hz, min(90.0, fs/4), seed=int(seed)+int(batch))
    freq, amplitude = calculate_spectrum(sig, fs)

    st.subheader("1 • FFT Frequency Analysis — 2D Spectrum")
    # Detect prominent local maxima without requiring an extra dependency.
    local = np.where((amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] >= amplitude[2:]))[0] + 1
    local = local[np.argsort(amplitude[local])[-8:]] if len(local) else np.array([], dtype=int)
    fft_fig = go.Figure()
    fft_fig.add_trace(go.Scatter(x=freq, y=amplitude, mode="lines", name="FFT amplitude", line=dict(width=1.6)))
    fft_fig.add_trace(go.Scatter(x=freq[local], y=amplitude[local], mode="markers+text", text=[f"{freq[i]:.1f} Hz" for i in local], textposition="top center", name="Detected peaks", marker=dict(size=8, symbol="diamond")))
    for harmonic in (1, 2, 3, 4):
        f_h = harmonic * shaft_hz
        if f_h <= 300:
            fft_fig.add_vline(x=f_h, line_dash="dot", opacity=0.55, annotation_text=f"{harmonic}× shaft", annotation_position="top")
    fft_fig.update_layout(title=f"FFT amplitude spectrum • {selected} • {rpm:.0f} RPM", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (a.u.)", legend=dict(orientation="h"))
    fft_fig.update_xaxes(range=[0, 300])
    st.plotly_chart(fft_fig, use_container_width=True)
    if len(local):
        peak_df = pd.DataFrame({"Peak frequency (Hz)": freq[local], "Amplitude (a.u.)": amplitude[local]}).sort_values("Amplitude (a.u.)", ascending=False)
        st.dataframe(peak_df.round(4), use_container_width=True, hide_index=True)

    st.subheader("2 • Shaft Orbit Analysis — 3D X-Y-Time Trajectory")
    # Orthogonal proximity probes: a slightly elliptical orbit with a 2x component.
    orbit_x = np.sin(2*np.pi*shaft_hz*t_sig) + 0.15*np.sin(4*np.pi*shaft_hz*t_sig)
    orbit_y = 0.78*np.cos(2*np.pi*shaft_hz*t_sig + 0.25) + 0.06*np.cos(4*np.pi*shaft_hz*t_sig)
    ox, oy = orbit_coordinates(orbit_x, orbit_y)
    orbit3d = go.Figure(go.Scatter3d(x=ox, y=oy, z=t_sig, mode="lines", line=dict(color=t_sig, colorscale="Turbo", width=5, colorbar=dict(title="Time (s)")), name="Shaft centreline"))
    orbit3d.add_trace(go.Scatter3d(x=[ox[0]], y=[oy[0]], z=[t_sig[0]], mode="markers", marker=dict(size=5, color="lime"), name="Start"))
    orbit3d.update_layout(title="3D shaft orbit • X displacement × Y displacement × Time", scene=dict(xaxis_title="X displacement (a.u.)", yaxis_title="Y displacement (a.u.)", zaxis_title="Time (s)", aspectmode="cube"), margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(orbit3d, use_container_width=True)
    st.caption("3D orbit is a time-parameterized trajectory from two synthetic orthogonal probes; it is not a measured shaft centreline.")

    st.subheader("3 • 3D Spectrogram — Time × Frequency × Amplitude")
    frame = 256
    hop = 64
    win = np.hanning(frame)
    rows = []
    for start_idx in range(0, len(sig)-frame+1, hop):
        chunk = sig[start_idx:start_idx+frame] - np.mean(sig[start_idx:start_idx+frame])
        rows.append(2*np.abs(np.fft.rfft(chunk*win))/max(np.sum(win), 1e-12))
    spec = np.asarray(rows).T
    stft_freq = np.fft.rfftfreq(frame, 1/fs)
    stft_time = np.arange(len(rows))*hop/fs
    # Limit displayed frequency range to keep the diagnostic plot legible.
    mask = stft_freq <= 300
    spec3d = go.Figure(go.Surface(x=stft_time, y=stft_freq[mask], z=spec[mask, :], colorscale="Viridis", colorbar_title="Amplitude (a.u.)"))
    spec3d.update_layout(title="Short-time FFT spectrogram • 3D surface", scene=dict(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", zaxis_title="Amplitude (a.u.)"), margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(spec3d, use_container_width=True)

    st.subheader("4 • Bearing Diagnosis — 2D Fault Frequency Map")
    b1, b2, b3, b4 = st.columns(4)
    rolling_elements = b1.number_input("Rolling elements", min_value=1, max_value=30, value=8, step=1)
    ball_diameter = b2.number_input("Ball diameter (mm)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    pitch_diameter = b3.number_input("Pitch diameter (mm)", min_value=2.0, max_value=300.0, value=50.0, step=0.5)
    contact_angle = b4.number_input("Contact angle (°)", min_value=0.0, max_value=45.0, value=0.0, step=1.0)
    if ball_diameter >= pitch_diameter:
        st.error("Ball diameter must be smaller than pitch diameter.")
    else:
        bearing = bearing_characteristic_frequencies(rpm, int(rolling_elements), float(ball_diameter), float(pitch_diameter), float(contact_angle))
        bearing_fig = go.Figure(go.Scatter(x=freq, y=amplitude, mode="lines", name="Measured/demo FFT", line=dict(width=1.5)))
        bearing_rows = []
        colors = {"FTF":"#636EFA", "BSF":"#EF553B", "BPFO":"#00CC96", "BPFI":"#AB63FA"}
        for fault, fault_hz in bearing.items():
            # Plot fundamental and first two harmonics when inside the display band.
            for harmonic in (1, 2, 3):
                marker_hz = fault_hz * harmonic
                if marker_hz <= 300:
                    idx = int(np.argmin(np.abs(freq-marker_hz)))
                    bearing_fig.add_vline(x=marker_hz, line_dash="dash", line_color=colors[fault], opacity=0.65,
                                          annotation_text=f"{fault}" if harmonic == 1 else f"{fault} ×{harmonic}", annotation_position="top")
                    bearing_rows.append({"Fault frequency":fault, "Harmonic":harmonic, "Theoretical Hz":marker_hz,
                                         "Nearest FFT bin (Hz)":float(freq[idx]), "FFT amplitude (a.u.)":float(amplitude[idx])})
        bearing_fig.update_layout(title=f"Bearing characteristic frequencies overlaid on FFT • {selected}", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (a.u.)", legend=dict(orientation="h"))
        bearing_fig.update_xaxes(range=[0, 300])
        st.plotly_chart(bearing_fig, use_container_width=True)
        bearing_df = pd.DataFrame(bearing_rows)
        st.dataframe(bearing_df.round(4), use_container_width=True, hide_index=True)
        st.download_button("Export bearing diagnosis CSV", bearing_df.to_csv(index=False).encode(), "bearing_fault_frequency_diagnosis.csv", "text/csv")
    st.download_button("Export FFT spectrum CSV", pd.DataFrame({"frequency_hz":freq,"amplitude_au":amplitude}).to_csv(index=False).encode(), "synthetic_fft_spectrum.csv", "text/csv")
    st.warning("All signals are synthetic. Bearing frequencies are ideal kinematic estimates. Confirm bearing geometry, load, slip, resonance, sensor calibration and measured vibration before any real maintenance decision.")


st.markdown("---")
st.markdown("**Engineering scope:** synthetic sensor streams; rule-derived labels; illustrative scoring thresholds. No calibrated RUL, live IoT/SCADA, or safety validation. Confirm any real asset decision with site procedures and measured data.")


with tabs[7]:
    st.header("V15 • Advanced Vibration Diagnostics")
    st.caption("Envelope spectrum, rotational-order view, signal condition indicators and spectral-band energy. All default signals are synthetic.")
    selected_v15 = st.selectbox("Select diagnostic machine", raw.machine_id.tolist(), key="v15_diag_asset")
    row_v15 = raw.loc[raw.machine_id == selected_v15].iloc[0]
    rpm_v15 = float(row_v15.rpm)
    shaft_hz_v15 = rpm_v15 / 60.0
    fs_v15 = 2048
    t_v15, sig_v15 = generate_vibration_signal(4.0, fs_v15, shaft_hz_v15, min(90.0, fs_v15/4), seed=int(seed)+int(batch)+101)

    st.subheader("1 • Signal Condition Indicators")
    feat_v15 = signal_features(sig_v15, fs_v15)
    metric_cols_v15 = st.columns(6)
    for col, (label, value) in zip(metric_cols_v15, feat_v15.items()):
        col.metric(label, f"{value:.3f}")
    st.caption("Kurtosis is the non-excess (Gaussian ≈ 3) fourth standardized moment. Indicators are descriptive, not universal alarm thresholds.")

    st.subheader("2 • Envelope Spectrum")
    env_freq, env_amp = envelope_spectrum(sig_v15, fs_v15)
    env_fig = go.Figure(go.Scatter(x=env_freq, y=env_amp, mode="lines", name="Envelope spectrum"))
    env_fig.update_layout(title="Demodulated envelope spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (a.u.)")
    env_fig.update_xaxes(range=[0, 300])
    st.plotly_chart(env_fig, use_container_width=True)
    env_df = pd.DataFrame({"frequency_hz": env_freq, "envelope_amplitude_au": env_amp})
    st.download_button("Export envelope spectrum CSV", env_df.to_csv(index=False).encode(), "v15_envelope_spectrum.csv", "text/csv")

    st.subheader("3 • Rotational Order Spectrum")
    order, order_amp = rotational_order_spectrum(sig_v15, fs_v15, rpm_v15)
    order_fig = go.Figure(go.Scatter(x=order, y=order_amp, mode="lines", name="Order spectrum"))
    for n_order in (1, 2, 3, 4):
        order_fig.add_vline(x=n_order, line_dash="dot", annotation_text=f"{n_order}×")
    order_fig.update_layout(title=f"Frequency normalized by shaft speed • {rpm_v15:.0f} RPM", xaxis_title="Rotational order (× shaft speed)", yaxis_title="Amplitude (a.u.)")
    order_fig.update_xaxes(range=[0, min(12, float(order.max()))])
    st.plotly_chart(order_fig, use_container_width=True)
    st.warning("This is frequency-axis normalization only. True order tracking requires a tachometer/keyphasor and angular resampling during speed variation.")

    st.subheader("4 • Spectral Band Energy")
    bands_v15 = spectral_band_energy(sig_v15, fs_v15)
    band_df_v15 = pd.DataFrame({"Frequency band": list(bands_v15), "Normalized energy fraction": list(bands_v15.values())})
    band_c1, band_c2 = st.columns([1, 1])
    band_c1.plotly_chart(px.bar(band_df_v15, x="Frequency band", y="Normalized energy fraction", title="Relative energy by frequency band", range_y=[0, 1]), use_container_width=True)
    band_c2.dataframe(band_df_v15.round(4), use_container_width=True, hide_index=True)
    st.download_button("Export v15 signal indicators CSV", pd.DataFrame([feat_v15]).to_csv(index=False).encode(), "v15_signal_condition_indicators.csv", "text/csv")
    st.info("Envelope analysis is most useful when impacts excite a resonance band; this demonstration applies the analytic-signal envelope directly to a synthetic waveform without a calibrated band-pass stage. Use measured, calibrated data and a validated resonance band for field diagnosis.")


with tabs[8]:
    st.header("V17 • Rotor Dynamics & Sensor Data Lab")
    st.caption("Campbell, Bode and waterfall visualizations use simplified models/synthetic signals. CSV validation checks structure and basic ranges only.")
    rpm_min, rpm_max = st.slider("RPM sweep", 300, 6000, (600, 3600), step=100)
    rpm_grid = np.linspace(rpm_min, rpm_max, 120)
    natural_hz = st.number_input("Illustrative rotor natural frequency (Hz)", 5.0, 300.0, 30.0, 1.0)
    damping = st.slider("Damping ratio", 0.01, 0.20, 0.04, 0.01)

    st.subheader("1 • Campbell Diagram")
    camp_rpm, order_lines, natural = campbell_data(rpm_grid, (natural_hz, natural_hz*2.1), 4)
    camp_fig = go.Figure()
    for label, values in order_lines.items():
        camp_fig.add_trace(go.Scatter(x=camp_rpm, y=values, mode="lines", name=f"{label} synchronous order"))
    for idx, value in enumerate(natural):
        camp_fig.add_hline(y=float(value), line_dash="dash", annotation_text=f"Natural frequency {value:.1f} Hz")
    camp_fig.update_layout(title="Campbell Diagram • synchronous excitation vs RPM", xaxis_title="Speed (RPM)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(camp_fig, use_container_width=True)

    st.subheader("2 • Bode Response (Simplified SDOF Model)")
    brpm, amp, phase = bode_unbalance_response(rpm_grid, natural_hz, damping)
    bode1, bode2 = st.columns(2)
    bode1.plotly_chart(go.Figure(go.Scatter(x=brpm, y=amp, mode="lines")).update_layout(title="Normalized synchronous amplitude", xaxis_title="RPM", yaxis_title="Normalized amplitude"), use_container_width=True)
    bode2.plotly_chart(go.Figure(go.Scatter(x=brpm, y=phase, mode="lines")).update_layout(title="Phase response", xaxis_title="RPM", yaxis_title="Phase (degrees)"), use_container_width=True)

    st.subheader("3 • 3D Waterfall Spectrum")
    fs17 = 2048
    t17 = np.arange(4096) / fs17
    rpm_steps = np.linspace(rpm_min, rpm_max, 12)
    signals17 = [0.8*np.sin(2*np.pi*(r/60)*t17) + 0.18*np.sin(2*np.pi*(2*r/60)*t17) + 0.05*np.random.default_rng(i).normal(size=t17.size) for i, r in enumerate(rpm_steps)]
    wf_freq, wf_rpm, wf_psd = waterfall_spectra(signals17, fs17, rpm_steps, nperseg=1024)
    wf_fig = go.Figure(go.Surface(x=wf_freq, y=wf_rpm, z=10*np.log10(wf_psd + 1e-12), colorscale="Viridis", colorbar_title="PSD (dB/Hz)"))
    wf_fig.update_layout(title="3D Waterfall • Frequency × RPM × Power Spectral Density", scene=dict(xaxis_title="Frequency (Hz)", yaxis_title="RPM", zaxis_title="PSD (dB/Hz)"), margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(wf_fig, use_container_width=True)

    st.subheader("4 • Real Sensor CSV Schema Audit")
    st.write("Required columns: timestamp, machine_id, rpm, temperature_c, vibration_rms")
    uploaded = st.file_uploader("Upload sensor CSV for structural validation", type=["csv"], key="v17_sensor_csv")
    if uploaded is not None:
        try:
            user_frame = pd.read_csv(uploaded)
            audit17 = validate_sensor_csv(user_frame)
            st.json(audit17)
            if audit17["valid"]:
                st.success("Basic CSV schema and range checks passed. This does not validate sensor calibration or diagnostic accuracy.")
                st.dataframe(user_frame.head(100), use_container_width=True, hide_index=True)
            else:
                st.error("CSV validation found issues. Review the report before using this data.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    st.warning("Campbell and Bode outputs are simplified illustrative rotor models, not a transfer-matrix/FEM rotor-bearing solver. Waterfall signals are simulated. Use validated modal parameters and measured run-up/coast-down data for engineering conclusions.")


with tabs[9]:
    st.header("V18 • Finite Element Rotor & Modal Analysis")
    st.caption("Linear Euler–Bernoulli shaft beam FE model with end translational bearing springs. Single-plane, isotropic, non-gyroscopic educational model.")
    g1,g2,g3=st.columns(3)
    length=g1.number_input("Shaft length (m)",0.2,3.0,1.0,0.1,key="v18_len")
    diameter=g2.number_input("Shaft diameter (m)",0.01,0.15,0.04,0.005,key="v18_dia")
    elements=g3.slider("Beam elements",2,30,8,key="v18_elements")
    g4,g5,g6=st.columns(3)
    young=g4.number_input("Young's modulus (GPa)",10.0,250.0,200.0,5.0,key="v18_young")*1e9
    density=g5.number_input("Density (kg/m³)",1000.0,20000.0,7800.0,100.0,key="v18_density")
    bearing_k=g6.number_input("Each bearing radial stiffness (MN/m)",0.1,500.0,20.0,1.0,key="v18_kb")*1e6
    n_modes=st.slider("Modes to extract",1,8,6,key="v18_modes")
    try:
        fe=solve_rotor_modes(length,diameter,young,density,elements,bearing_k,n_modes)
        modal_df=pd.DataFrame({"Mode":np.arange(1,n_modes+1),"Natural frequency (Hz)":fe["frequencies_hz"],"1X critical speed estimate (RPM)":fe["critical_speeds_rpm"]})
        m1,m2,m3=st.columns(3)
        m1.metric("First natural frequency",f"{fe['frequencies_hz'][0]:.2f} Hz")
        m2.metric("1X critical speed estimate",f"{fe['critical_speeds_rpm'][0]:.0f} RPM")
        m3.metric("FE nodes",str(len(fe['nodes_m'])))
        st.subheader("1 • FE Modal Results & Critical-Speed Validation")
        st.dataframe(modal_df.round(3),use_container_width=True,hide_index=True)
        op_low,op_high=st.slider("Operating speed range (RPM)",300,12000,(600,3600),step=100,key="v18_oprange")
        crit=fe["critical_speeds_rpm"]
        near=crit[(crit>=op_low)&(crit<=op_high)]
        if len(near): st.warning("Potential 1X synchronous crossing(s) inside selected operating range: "+", ".join(f"{v:.0f} RPM" for v in near)+". This is a screening result, not a safe operating limit.")
        else: st.success("No extracted 1X natural-frequency crossing falls inside this operating range in the current simplified model.")
        camp=go.Figure()
        rpm_line=np.linspace(300,max(12000,op_high),200)
        camp.add_trace(go.Scatter(x=rpm_line,y=rpm_line/60,mode="lines",name="1X synchronous line"))
        for i,f_hz in enumerate(fe["frequencies_hz"]):
            camp.add_hline(y=float(f_hz),line_dash="dash",annotation_text=f"FE mode {i+1}: {f_hz:.1f} Hz")
        camp.update_layout(title="FE natural frequencies vs 1X synchronous excitation",xaxis_title="Rotor speed (RPM)",yaxis_title="Frequency (Hz)")
        st.plotly_chart(camp,use_container_width=True)

        st.subheader("2 • Mode Shape Visualization")
        mode_pick=st.selectbox("Select FE mode",list(range(1,n_modes+1)),index=0,key="v18_mode_pick")
        shape=fe["mode_shapes"][:,mode_pick-1]
        mode_fig=go.Figure(go.Scatter(x=fe["nodes_m"],y=shape,mode="lines+markers",name=f"Mode {mode_pick}"))
        mode_fig.add_hline(y=0,line_color="gray",line_width=1)
        mode_fig.update_layout(title=f"Lateral bending mode {mode_pick} • {fe['frequencies_hz'][mode_pick-1]:.2f} Hz",xaxis_title="Axial position (m)",yaxis_title="Normalized lateral displacement")
        st.plotly_chart(mode_fig,use_container_width=True)

        st.subheader("3 • Experimental vs Numerical Modal Comparison")
        st.write("Upload a CSV with required columns `mode,freq_hz`. Optional shape columns: `shape_node_0`, `shape_node_1`, … matching FE node count.")
        template=pd.DataFrame({"mode":np.arange(1,n_modes+1),"freq_hz":np.full(n_modes,np.nan)})
        st.download_button("Download experimental modal CSV template",template.to_csv(index=False).encode(),"v18_experimental_modal_template.csv","text/csv")
        exp_file=st.file_uploader("Upload experimental modal test results",type=["csv"],key="v18_modal_csv")
        if exp_file is not None:
            try:
                exp_df=pd.read_csv(exp_file)
                comparison=compare_experimental_modes(fe["frequencies_hz"],exp_df,fe["mode_shapes"])
                st.dataframe(comparison.round(4),use_container_width=True,hide_index=True)
                valid_mac=comparison["MAC"].dropna()
                if len(valid_mac): st.caption(f"Mean Modal Assurance Criterion (MAC): {valid_mac.mean():.3f} (1.0 indicates identical sampled shape vectors).")
                st.download_button("Export numerical vs experimental comparison",comparison.to_csv(index=False).encode(),"v18_modal_comparison.csv","text/csv")
                cmp_fig=go.Figure()
                cmp_fig.add_trace(go.Scatter(x=comparison["mode"],y=comparison["experimental_hz"],mode="markers+lines",name="Experimental"))
                cmp_fig.add_trace(go.Scatter(x=comparison["mode"],y=comparison["numerical_hz"],mode="markers+lines",name="FE numerical"))
                cmp_fig.update_layout(title="Experimental vs FE natural frequencies",xaxis_title="Mode number",yaxis_title="Natural frequency (Hz)")
                st.plotly_chart(cmp_fig,use_container_width=True)
            except Exception as exc: st.error(f"Modal comparison could not be calculated: {exc}")
        st.download_button("Export FE modal results",modal_df.to_csv(index=False).encode(),"v18_fe_modal_results.csv","text/csv")
        st.info(f"Model geometry: A={fe['area_m2']:.6g} m², I={fe['second_moment_m4']:.6g} m⁴. FE stiffness and consistent mass matrices are assembled from {elements} Euler–Bernoulli beam elements.")
    except Exception as exc:
        st.error(f"FE model error: {exc}")
    st.warning("Engineering limitations: this is a single-plane Euler–Bernoulli beam with simple end translational springs. It omits gyroscopic effects, shear deformation, rotary inertia, disk masses, anisotropic bearing coefficients, damping, seals, fluid-film effects and full rotor-bearing coupling. Critical-speed values are undamped 1X frequency crossings only. Experimental comparison is meaningful only with properly measured, correctly paired modes and matching mode-shape coordinates; no experimental data is bundled.")


with tabs[10]:
    st.header("V19 • Timoshenko Rotor Dynamics & Gyroscopic Campbell Lab")
    st.caption("Two-plane Timoshenko beam FE model with disk mass/polar inertia, gyroscopic coupling, isotropic bearing stiffness and damping. Educational linear model; validate against a trusted rotor solver and measured data before design use.")
    a,b,c=st.columns(3)
    L=a.number_input("Shaft length (m)",0.2,3.0,1.0,0.1,key="v19_L")
    dia=b.number_input("Shaft diameter (m)",0.01,0.15,0.04,0.005,key="v19_d")
    nel=c.slider("Timoshenko elements",2,12,6,key="v19_nel")
    d,e,f=st.columns(3)
    E=d.number_input("Young's modulus (GPa)",10.0,250.0,200.0,5.0,key="v19_E")*1e9
    rho=e.number_input("Density (kg/m³)",1000.0,20000.0,7800.0,100.0,key="v19_rho")
    nu=f.number_input("Poisson ratio",0.20,0.49,0.30,0.01,key="v19_nu")
    g,h,i=st.columns(3)
    kb=g.number_input("Bearing stiffness (MN/m per support)",0.1,500.0,20.0,1.0,key="v19_kb")*1e6
    cb=h.number_input("Bearing damping (N·s/m per support)",0.0,50000.0,1200.0,100.0,key="v19_cb")
    kappa=i.number_input("Shear correction factor",0.50,1.00,0.89,0.01,key="v19_kappa")
    j,k,l=st.columns(3)
    disk_mass=j.number_input("Disk mass (kg)",0.1,100.0,8.0,0.5,key="v19_dm")
    jp=k.number_input("Disk polar inertia (kg·m²)",0.001,5.0,0.035,0.005,key="v19_jp")
    jd=l.number_input("Disk diametral inertia (kg·m²)",0.001,5.0,0.020,0.005,key="v19_jd")
    disk_node=st.slider("Disk node index",0,int(nel),int(nel)//2,key="v19_disknode")
    max_rpm=st.slider("Maximum rotor speed (RPM)",1000,24000,12000,500,key="v19_maxrpm")
    try:
        rotor=build_timoshenko_rotor(length_m=L,diameter_m=dia,youngs_modulus_pa=E,density_kg_m3=rho,
            poisson_ratio=nu,elements=nel,shear_factor=kappa,bearing_stiffness_n_m=kb,
            bearing_damping_n_s_m=cb,disk_node=disk_node,disk_mass_kg=disk_mass,
            disk_polar_inertia_kg_m2=jp,disk_diametral_inertia_kg_m2=jd)
        camp19=solve_rotor_campbell(rotor,np.linspace(0,max_rpm,41))
        m1,m2,m3,m4=st.columns(4)
        m1.metric("FE DOFs",str(rotor['K'].shape[0]))
        m2.metric("Disk node",str(disk_node))
        m3.metric("Bearing damping",f"{cb:.0f} N·s/m")
        m4.metric("Campbell samples",str(camp19.rpm.nunique()))
        st.subheader("1 • Forward / Backward Whirl Campbell Diagram")
        fig19=go.Figure()
        for whirl,color in [("Forward","royalblue"),("Backward","firebrick")]:
            sub=camp19[camp19.whirl==whirl]
            if not sub.empty:
                fig19.add_trace(go.Scatter(x=sub.rpm,y=sub.frequency_hz,mode="markers",name=f"{whirl} whirl",marker=dict(size=4,color=color),customdata=sub[["damping_ratio"]],hovertemplate="RPM=%{x:.0f}<br>Frequency=%{y:.2f} Hz<br>Damping ratio=%{customdata[0]:.4f}<extra></extra>"))
        rpm_line=np.linspace(0,max_rpm,200)
        fig19.add_trace(go.Scatter(x=rpm_line,y=rpm_line/60,mode="lines",name="1X synchronous",line=dict(color="black",dash="dash")))
        fig19.add_trace(go.Scatter(x=rpm_line,y=2*rpm_line/60,mode="lines",name="2X synchronous",line=dict(dash="dot")))
        fig19.update_layout(title="Gyroscopic Campbell Diagram • whirl branches",xaxis_title="Rotor speed (RPM)",yaxis_title="Damped eigenfrequency (Hz)",hovermode="closest")
        st.plotly_chart(fig19,use_container_width=True)
        crossings=critical_speed_crossings(camp19,1)
        if crossings.empty: st.info("No 1X crossing detected within this sampled RPM range. This is a numerical screening result, not a safe-speed certification.")
        else:
            st.subheader("2 • Estimated 1X Critical-Speed Crossings")
            st.dataframe(crossings.round(2),use_container_width=True,hide_index=True)
            st.download_button("Export Campbell branches CSV",camp19.to_csv(index=False).encode(),"v19_campbell_branches.csv","text/csv")
        st.subheader("3 • Assembled Rotor Matrices")
        matrix_cols=st.columns(4)
        for col,key,label in zip(matrix_cols,["K","M","G","C"],["Stiffness K","Mass M","Gyroscopic G","Damping C"]):
            col.metric(label,f"{np.linalg.norm(rotor[key]):.3e}","Frobenius norm")
        st.caption(f"Timoshenko shear modulus: {rotor['shear_modulus_pa']/1e9:.2f} GPa • Disk mass {disk_mass:.2f} kg • Polar inertia {jp:.4f} kg·m²")
        st.download_button("Export Campbell data CSV",camp19.to_csv(index=False).encode(),"v19_campbell_data.csv","text/csv")
        st.download_button("Export matrix diagnostics CSV",pd.DataFrame({"matrix":["K","M","G","C"],"frobenius_norm":[np.linalg.norm(rotor[x]) for x in ["K","M","G","C"]]}).to_csv(index=False).encode(),"v19_matrix_diagnostics.csv","text/csv")
    except Exception as exc:
        st.error(f"V19 rotor model could not be solved: {exc}")
    st.warning("Model scope: linear isotropic bearings; no cross-coupled fluid-film coefficients, seals, shaft axial/torsional DOFs, nonlinear contact, unbalance response or full experimental correlation. Timoshenko shear stiffness and rotary inertia are included; disk gyroscopic coupling is concentrated at one rigid disk node. Forward/backward labels use disk-node orbit phase and should be checked against a validated rotor-dynamics package.")


with tabs[11]:
    st.header("V20 • Bearing Dynamics, Stability & Unbalance Response")
    st.caption("V19 Timoshenko rotor FE matrices augmented with user-supplied linearized anisotropic/cross-coupled bearing coefficients. This is not a hydrodynamic Reynolds-equation bearing solver.")
    st.subheader("1 • Linearized Bearing Coefficients")
    st.write("Enter direct and cross-coupled coefficients for each end support. Values must come from measured data, a bearing calculation, or a trusted reference model for engineering use.")
    c1,c2,c3,c4=st.columns(4)
    kxx=c1.number_input("Kxx (MN/m)",0.1,500.0,22.0,0.5,key="v20_kxx")*1e6
    kyy=c2.number_input("Kyy (MN/m)",0.1,500.0,18.0,0.5,key="v20_kyy")*1e6
    kxy=c3.number_input("Kxy (MN/m)",-100.0,100.0,0.1,0.1,key="v20_kxy")*1e6
    kyx=c4.number_input("Kyx (MN/m)",-100.0,100.0,-0.1,0.1,key="v20_kyx")*1e6
    d1,d2,d3,d4=st.columns(4)
    cxx=d1.number_input("Cxx (N·s/m)",0.0,50000.0,1400.0,100.0,key="v20_cxx")
    cyy=d2.number_input("Cyy (N·s/m)",0.0,50000.0,1250.0,100.0,key="v20_cyy")
    cxy=d3.number_input("Cxy (N·s/m)",-50000.0,50000.0,10.0,10.0,key="v20_cxy")
    cyx=d4.number_input("Cyx (N·s/m)",-50000.0,50000.0,-5.0,10.0,key="v20_cyx")
    coeff20={"kxx":kxx,"kyy":kyy,"kxy":kxy,"kyx":kyx,"cxx":cxx,"cyy":cyy,"cxy":cxy,"cyx":cyx}
    rpm_max20=st.slider("V20 maximum speed (RPM)",1000,24000,12000,500,key="v20_rpmmax")
    unb=st.number_input("Residual unbalance U (kg·m)",0.0,0.05,0.0001,0.0001,format="%.5f",key="v20_unb")
    phase20=st.number_input("Unbalance phase (degrees)",-180.0,180.0,0.0,5.0,key="v20_phase")
    try:
        base20=build_timoshenko_rotor(length_m=1.0,diameter_m=0.04,youngs_modulus_pa=2e11,
            density_kg_m3=7800,poisson_ratio=0.3,elements=6,shear_factor=0.89,
            bearing_stiffness_n_m=20e6,bearing_damping_n_s_m=1200,
            disk_node=3,disk_mass_kg=8,disk_polar_inertia_kg_m2=0.035,disk_diametral_inertia_kg_m2=0.02)
        rotor20=apply_linearized_bearing_coefficients(base20,coeff20)
        speeds20=np.linspace(300,rpm_max20,60)
        response20=synchronous_unbalance_response(rotor20,speeds20,unb,phase20)
        stability20=stability_sweep(rotor20,np.linspace(0,rpm_max20,41))
        peak=response20.loc[response20.resultant_amplitude_m.idxmax()]
        s1,s2,s3=st.columns(3)
        s1.metric("Peak synchronous displacement",f"{peak.resultant_amplitude_m*1e6:.2f} µm")
        s2.metric("Peak response speed",f"{peak.rpm:.0f} RPM")
        unstable=int((stability20.stability=="Unstable").sum())
        s3.metric("Unstable sampled speeds",str(unstable))
        st.subheader("2 • 1X Unbalance Response")
        respfig=go.Figure()
        respfig.add_trace(go.Scatter(x=response20.rpm,y=response20.x_amplitude_m*1e6,mode="lines",name="X amplitude"))
        respfig.add_trace(go.Scatter(x=response20.rpm,y=response20.y_amplitude_m*1e6,mode="lines",name="Y amplitude"))
        respfig.add_trace(go.Scatter(x=response20.rpm,y=response20.resultant_amplitude_m*1e6,mode="lines",name="Resultant",line=dict(width=3,dash="dash")))
        respfig.update_layout(xaxis_title="Rotor speed (RPM)",yaxis_title="Steady-state displacement (µm)",title="Synchronous unbalance response")
        st.plotly_chart(respfig,use_container_width=True)
        st.download_button("Export unbalance response CSV",response20.to_csv(index=False).encode(),"v20_unbalance_response.csv","text/csv")
        st.subheader("3 • Linear Stability Screening")
        stabfig=go.Figure(go.Scatter(x=stability20.rpm,y=stability20.max_real_eigenvalue_per_s,mode="lines+markers",name="Rightmost eigenvalue"))
        stabfig.add_hline(y=0,line_dash="dash",line_color="red")
        stabfig.update_layout(xaxis_title="Rotor speed (RPM)",yaxis_title="Maximum eigenvalue real part (1/s)",title="Linear stability indicator • positive means unstable")
        st.plotly_chart(stabfig,use_container_width=True)
        st.dataframe(stability20.round(5),use_container_width=True,hide_index=True)
        st.download_button("Export stability sweep CSV",stability20.to_csv(index=False).encode(),"v20_stability_sweep.csv","text/csv")
        st.subheader("4 • Bearing Coefficient Matrix")
        st.dataframe(pd.DataFrame([[kxx,kxy],[kyx,kyy]],index=["X equation","Y equation"],columns=["X displacement","Y displacement"]).round(2),use_container_width=True)
    except Exception as exc:
        st.error(f"V20 calculation error: {exc}")
    st.warning("Limitations: coefficients are user-entered linearized values; no Reynolds equation, cavitation, thermal-viscous solution or nonlinear journal orbit is solved. Unbalance response assumes linear steady-state 1X forcing and a fixed FE model. Stability results depend strongly on coefficient quality, support assumptions and model fidelity. Validate against measured run-up/coast-down data and a trusted rotor-bearing solver before design decisions.")


with tabs[12]:
    st.header("V21 • Complete 3D Engineering Atlas")
    st.caption("3D diagrams for the major project capabilities accumulated from v7 through v20. Rotate, zoom and hover each chart. Model-derived plots and illustrative geometry are explicitly identified.")
    try:
        atlas = build_3d_atlas(raw, scored, importance, X_test, prob, int(seed), y_test)
        total = sum(len(items) for items in atlas.values())
        k1,k2,k3 = st.columns(3)
        k1.metric("3D diagrams", total)
        k2.metric("Engineering domains", len(atlas))
        k3.metric("Project coverage", "v7–v20")
        group = st.selectbox("Choose engineering domain", list(atlas.keys()), key="v21_atlas_group")
        st.subheader(group)
        for idx in range(0, len(atlas[group]), 2):
            left,right = st.columns(2)
            for col,item in zip((left,right), atlas[group][idx:idx+2]):
                title, figure, note = item
                with col:
                    st.plotly_chart(figure, use_container_width=True, key="v21_"+str(abs(hash(title))))
                    st.caption(note)
        st.download_button("Export 3D Atlas inventory CSV",
            pd.DataFrame([{"domain":domain,"diagram":title,"provenance":note} for domain,items in atlas.items() for title,_,note in items]).to_csv(index=False).encode(),
            "v21_3d_engineering_atlas_inventory.csv", "text/csv")
    except Exception as exc:
        st.error(f"3D Atlas could not be generated: {exc}")
    st.warning("Important: not every visual is a separate 3D physical simulation. Some are 3D renderings of existing numerical outputs; geometry-only diagrams are labeled illustrative. Experimental comparison remains numerical-only until measured modal data is uploaded. No live sensors, certified CAD, CFD or industrial safety validation are implied.")
