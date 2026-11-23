# Changelog

Release history consolidated from the per-version release notes (v6 to v21). Original wording is kept; each version's heading is demoted one level.

## V21 — Complete 3D Engineering Atlas

### Purpose
V21 adds a dedicated 3D Engineering Atlas covering the major capabilities accumulated in the v7–v20 project. It complements the existing 2D diagnostic charts; it does not replace their numerical data or engineering interpretation.

### Included 3D diagrams (28)

#### Fleet, AI and operations
1. Fleet health digital twin (temperature, vibration, health)
2. AI risk probability space
3. Maintenance priority cube
4. Sensor anomaly map
5. ML holdout decision space
6. 3D confusion matrix
7. Model explainability (permutation importance)

#### Vibration and bearing diagnostics
8. Shaft geometry and orbit
9. FFT harmonic peak map
10. STFT spectrogram surface
11. Thermal hotspot field
12. Rolling-bearing 3D assembly with FTF/BSF/BPFO/BPFI labels
13. Envelope spectrum evolution
14. Rotational-order map across RPM slices
15. Signal condition-indicator trajectory (RMS, crest factor, kurtosis)
16. Spectral band energy map

#### Rotor, FEM and dynamics
17. Gyroscopic Campbell branches (forward/backward)
18. Bode amplitude-phase response
19. RPM waterfall spectrum
20. FE modal shape family
21. Timoshenko shaft/bending envelope
22. Concentrated disk mass and inertia geometry
23. Linearized bearing stiffness surface
24. Cross-coupled bearing force field
25. Synchronous unbalance response
26. Rotor stability map
27. Numerical vs experimental modal comparison placeholder (experimental series remains empty until measurements are uploaded)

#### Data quality and reporting
28. Sensor data quality profile across current telemetry channels

### Implementation
- New module: `src/v21_3d_atlas.py`
- New dashboard: `app/fleet_monitor_v21.py`
- New tab: `Complete 3D Engineering Atlas v21`
- New requirements file: `requirements-v21.txt`
- New tests: `tests/test_v21_3d_atlas.py`
- All charts are Plotly 3D traces (Scatter3d/Surface) with rotate, zoom and hover.

### Important engineering scope
- Most diagrams use existing synthetic telemetry and reproducible numerical models.
- The thermal field and bearing geometry are illustrative visualizations, not measured thermal maps or CAD.
- The ML model is trained on rule-generated synthetic labels; its plots do not demonstrate validated failure prediction.
- Rotordynamic plots use the included educational linear Timoshenko/gyroscopic model and supplied linearized bearing coefficients.
- Experimental modal comparison intentionally does not invent experimental values; upload measured modal results to populate the experimental series.
- No live sensor integration, certified CAD/FEA, CFD/Reynolds bearing solution, or industrial safety validation is claimed.

## V20 — Bearing Dynamics, Stability & Unbalance Response

### Added
- User-supplied linearized anisotropic bearing stiffness coefficients: Kxx, Kyy, Kxy, Kyx.
- User-supplied linearized bearing damping coefficients: Cxx, Cyy, Cxy, Cyx.
- Bearing coefficient replacement at the V19 end support nodes without mutating the source rotor model.
- Steady-state synchronous 1X response to rotating residual unbalance, using the complex dynamic stiffness equation.
- X, Y and resultant displacement amplitudes, force amplitude and response phase versus RPM.
- Speed-swept state-space eigenvalue stability screen, rightmost eigenvalue, least-damped oscillation, damping ratio and logarithmic decrement.
- Streamlit tab: `Bearing & Unbalance Lab v20`; CSV exports for response and stability sweep.

### Run
```bash
pip install -r requirements-v20.txt
streamlit run app/fleet_monitor_v20.py
python -m unittest discover -s tests -v
```

### Model assumptions and limitations
- Bearing coefficients are entered by the user. The application does **not** solve the Reynolds equation and does not calculate fluid-film coefficients, cavitation, thermal-viscous effects or nonlinear journal-bearing orbits.
- The unbalance response is a linear steady-state 1X harmonic solution on the V19 FE model. It omits nonlinear response, rub/contact, transient run-up, seal forces and distributed unbalance.
- The stability sweep is a linear eigenvalue screening based on the supplied constant coefficients. Positive rightmost eigenvalue real part indicates instability in this mathematical model; it is not a safety certification.
- The demonstration rotor geometry is configured in the v20 dashboard source. Update it and coefficient inputs to match a documented rotor/support system.
- Validate coefficient conventions, units, support locations and results against measured run-up/coast-down data and a trusted rotor-bearing solver before design decisions.

## V19 — Timoshenko Rotor Dynamics & Gyroscopic Campbell Lab

### Added
- Two-plane Timoshenko beam finite-element shaft model with shear-flexible bending stiffness.
- Consistent translational beam mass and rotary-inertia contribution.
- Rigid disk mass, diametral inertia and concentrated polar-inertia gyroscopic coupling.
- Isotropic end-bearing radial stiffness and viscous damping matrices.
- Speed-dependent damped state-space eigenanalysis.
- Forward/backward whirl classification from disk-node orbit phase.
- Campbell plot with 1X/2X synchronous lines and estimated 1X crossing table.
- CSV exports for Campbell branches and matrix diagnostics.

### Run
```bash
pip install -r requirements-v19.txt
streamlit run app/fleet_monitor_v19.py
python -m unittest discover -s tests -v
```

### Model assumptions / limitations
Linear isotropic bearings, Eulerian stationary-frame formulation, one concentrated rigid disk, and a simplified two-plane Timoshenko beam. It does not include cross-coupled fluid-film bearing coefficients, seals, nonlinear contact, axial/torsional shaft DOFs, distributed disks, shear-center offsets, unbalance response, or experimental correlation. Whirl labels are estimated from disk-node orbit phase. Critical-speed crossings are interpolated from a finite RPM grid. Validate against benchmark cases and a trusted rotor-dynamics solver before engineering decisions.

## v18 — Finite Element Rotor & Modal Research Lab

### New capabilities
- Euler–Bernoulli beam finite-element shaft model for one lateral bending plane.
- Consistent beam element stiffness and mass matrices, assembled over a configurable mesh.
- Translational bearing spring supports at both shaft ends.
- Generalized eigenvalue solution for natural frequencies and normalized lateral mode shapes.
- First-order synchronous critical-speed screening: `critical RPM = natural frequency (Hz) × 60`.
- Operating-speed-range crossing check and Campbell-style overlay of FE natural frequencies against the 1X line.
- Experimental-vs-numerical frequency comparison from CSV (`mode,freq_hz`).
- Optional Modal Assurance Criterion (MAC) when experimental shape columns `shape_node_0...shape_node_N` match the FE node count.
- CSV templates and exportable FE modal/comparison tables.

### Run
```bash
pip install -r requirements-v18.txt
streamlit run app/fleet_monitor_v18.py
python -m unittest discover -s tests -v
```

### Experimental CSV format
Required columns:
```csv
mode,freq_hz
1,78.2
2,289.1
```
Optional mode-shape ordinates can be appended as `shape_node_0`, `shape_node_1`, ... in node order. The number of shape ordinates must equal the number of FE nodes for MAC to be calculated.

### Model assumptions and limitations
This is an educational, linear, isotropic, non-gyroscopic Euler–Bernoulli beam model with translational spring supports at both ends and one lateral bending plane. It omits Timoshenko shear deformation, rotary inertia, disk masses, gyroscopic coupling, anisotropic/dynamic bearing coefficients, damping, seals, fluid-film effects, thermal effects, and full rotor-bearing coupling. The reported critical speeds are undamped 1X synchronous frequency crossings, not validated stability limits. Experimental comparison is only meaningful when mode numbers are correctly paired and measurement coordinates/scaling are compatible. No experimental measurements are bundled. Do not use for safety-critical decisions or as a substitute for validated commercial rotor-dynamics/FEA software and qualified engineering review.

## v17 — Rotor Dynamics & Sensor Data Lab

### New capabilities
- Campbell diagram with 1X–4X synchronous excitation lines and user-defined illustrative natural-frequency lines.
- Simplified single-degree-of-freedom (SDOF) unbalance Bode amplitude/phase response.
- 3D Welch-PSD waterfall spectrum across an RPM sweep.
- Sensor CSV schema/range audit for timestamp, machine_id, rpm, temperature_c and vibration_rms.
- Eight focused unit tests for data validation and rotor visualization calculations.

### Run
```bash
pip install -r requirements-v17.txt
streamlit run app/fleet_monitor_v17.py
python -m unittest discover -s tests -v
```

### Engineering limitations
All dashboard defaults and waterfall signals are synthetic. Campbell lines are synchronous-order overlays against user-entered illustrative natural frequencies. The Bode chart is a normalized SDOF model, not a validated rotor-bearing transfer-matrix model. CSV checks validate schema/basic values only; they do not establish calibration, sensor mounting quality, data completeness, or fault diagnosis. Do not use for safety-critical decisions.

## v15 — Advanced Vibration Diagnostics

### Added
- Signal condition indicators: RMS, peak, peak-to-peak, crest factor, shape factor and non-excess kurtosis.
- FFT-based analytic-signal envelope and envelope spectrum.
- Rotational-order spectrum (frequency bins normalized by shaft frequency).
- Normalized spectral-band energy summary and CSV exports.
- Dedicated Advanced Diagnostics v15 dashboard tab.
- Five additional unit tests for numerical validity and input handling.

### Run
```bash
pip install -r requirements-v15.txt
streamlit run app/fleet_monitor_v14.py
python -m unittest discover -s tests -v
```

### Engineering limitations
The included dashboard signal remains synthetic. The envelope spectrum is demonstrated without a machine-specific band-pass/resonance selection; it is not a validated bearing fault classifier. The order spectrum is frequency normalization only, not tachometer-based order tracking under variable speed. Condition indicators have no universal alarm limits. Validate with calibrated measurements, machine-specific baselines and qualified engineering review.

## v14 — Engineering Diagnostic Charts

Adds four dedicated interactive diagnostic visualizations to the Streamlit app:

1. 2D FFT amplitude spectrum with detected local peaks and 1×–4× shaft-speed markers.
2. 3D shaft orbit trajectory using orthogonal synthetic X/Y probe signals and time as Z.
3. 3D short-time FFT spectrogram (time, frequency, amplitude).
4. 2D bearing fault-frequency map overlaying FTF, BSF, BPFO and BPFI fundamentals and harmonics on the FFT spectrum. Bearing geometry is editable in the dashboard.

Run: `pip install -r requirements-v14.txt` then `streamlit run app/fleet_monitor_v14.py`.
Tests: `python -m unittest discover -s tests -v`.

**Engineering limitation:** all telemetry and vibration signals are synthetic. The 3D orbit is a time-parameterized trajectory, not a measured rotor orbit. Bearing frequencies are ideal kinematic estimates and are not a certified fault diagnosis.

## v13 — Advanced Signal Diagnostics Edition

Builds on the v12 3D Engineering Lab and adds a signal-processing workbench:

- Reproducible synthetic vibration waveform with configurable machine RPM-derived shaft frequency.
- One-sided Hann-window FFT amplitude spectrum.
- Orthogonal-probe shaft centreline orbit visualization.
- Interactive 3D short-time FFT spectrogram (frequency × time × amplitude).
- Idealized FTF, BSF, BPFO and BPFI calculations from explicit assumed rolling-bearing geometry.
- CSV export of the FFT spectrum.
- Input validation and automated tests for the signal-processing utilities.

### Run

```bash
pip install -r requirements-v13.txt
streamlit run app/fleet_monitor_v11.py
```

### Tests

```bash
python -m unittest discover -s tests -v
```

### Engineering limitations

All signals are synthetic. Bearing geometry is assumed (8 rolling elements, 10 mm rolling-element diameter, 50 mm pitch diameter, 0° contact angle). The FFT uses a Hann window; frequency resolution depends on record length and sample rate. No calibrated sensor acquisition, envelope analysis, order tracking, ISO severity assessment, measured bearing geometry, or validated fault classifier is included. Do not use outputs for safety-critical maintenance decisions.

## v12 — Advanced 3D Engineering Visualization Edition

Adds an interactive **3D Engineering Lab** tab to the Streamlit application:
- Conceptual rotating shaft geometry with vibration-pattern surface
- Vibration surface across time and frequency
- Synthetic spatial thermal field with hotspots
- Fleet risk 3D scatter (temperature, vibration, synthetic risk)
- Conceptual degradation trajectory colored by health score

All plots are interactive Plotly 3D figures. They use synthetic/conceptual data and are not CAD, CFD, FEA, calibrated sensor data, or safety validation.

Run: `streamlit run app/fleet_monitor_v11.py`

## v11 — Industrial AI Reliability Suite

### Added
- Held-out 25% validation of a reproducible synthetic Random Forest dataset.
- Accuracy, Precision, Recall, F1 and ROC-AUC reporting.
- Confusion matrix visualization.
- Permutation feature importance with repeat variability.
- Executive KPIs and high-risk asset table.
- Export buttons for telemetry, scored fleet, and draft work orders.
- Fixed KPI calculations in the new v11 dashboard; v10 remains preserved.

### Run
```bash
pip install -r requirements-v11.txt
streamlit run app/fleet_monitor_v11.py
python -m unittest discover -s tests -v
```

### Limitations
All data and labels are synthetic and rule-generated. Validation metrics evaluate the model's ability to reproduce the artificial rule, not real equipment failures. Thresholds are illustrative. There is no live sensor/PLC/SCADA/CMMS integration or validated remaining useful life. Never use this demo as the sole basis for operational or safety decisions.

## v10 — Enterprise Fleet Intelligence

### Highlights
- Executive overview with fleet health, critical/high priority counts, draft work-order count, and audit status.
- Composite machine priority score: illustrative 50% health-derived risk, 35% synthetic classifier probability, 15% anomaly flag.
- Ranked fleet table with recommendations and transparent synthetic probability label.
- Scenario controls: normal, elevated vibration, thermal stress, and mixed stress.
- Vibration/temperature history, trend summary, data-quality audit, and CSV exports.
- Draft work-order queue (not dispatched or operational instructions).
- Seven automated tests covering score bounds, sorting, queue, audit, trend and reproducibility.

### Start
```bash
pip install -r requirements-v10.txt
streamlit run app/fleet_monitor_v10.py
```

### Tests
```bash
python -m unittest tests/test_v10_enterprise.py -v
```

### Limitations
All telemetry and model labels are synthetic. Scores, priorities, thresholds and recommendations are illustrative. No live PLC/SCADA, CMMS or sensor connection; no validated failure probability or remaining useful life. Do not use for real maintenance/safety decisions.

## v9 — AI Fleet Intelligence Upgrade

### Added
- Transparent 0–100 Health Index and risk bands
- IQR-based per-batch anomaly flags
- Synthetic Random Forest risk classifier (explicitly synthetic threshold labels)
- Linear vibration threshold projection across simulated batches (requires >=3 observations)
- Streamlit AI fleet dashboard with 3D map, risk bars, projections and CSV exports
- 5 automated v9 tests

### Launch
```bash
pip install -r requirements-v9.txt
streamlit run app/fleet_monitor_v9.py
python -m unittest discover -s tests -v
```

### Important limitations
No live equipment data. Model target labels are generated from illustrative thresholds. Projected crossing is a linear extrapolation across batch numbers, not calibrated time or Remaining Useful Life (RUL). Use only for education and portfolio demonstration; no operational or safety decisions.

## v8 Fleet Intelligence Upgrade

### Added
- Multi-batch fleet history in dashboard session
- Vibration and temperature trend plots per machine
- First-to-last machine trend summary (Worsening / Stable / Improving)
- Sensor-data quality audit (required columns, missing values, duplicates, numeric checks)
- Batch/history/work-order CSV exports
- Reset and reproducibility seed controls
- New unit tests for v8 helpers

### Launch
```bash
pip install -r requirements-v7.txt
streamlit run app/fleet_monitor_v8.py
python -m unittest discover -s tests -v
```

Trend heuristics and priority bands are illustrative. Data are synthetic. No live machine, PLC, SCADA or IoT integration is present. Validate against site-approved/OEM limits before operational use.

## v7 Professional Upgrade

### New
- Reproducible synthetic sensor-stream generator for multiple machines
- 3D fleet digital-twin map (RPM / temperature / vibration)
- Transparent maintenance priority scoring
- Draft maintenance work-order queue and CSV export
- Streamlit fleet monitoring center with batch simulation
- Notebook exports for telemetry, work queue and interactive 3D map
- Unit tests for the monitoring helper module

### Run
Install: `pip install -r requirements-v7.txt`

Notebook: open `notebooks/Predictive_Maintenance_Industrial_Digital_Twin_v7_Professional.ipynb` and run cells from top to bottom.

Dashboard: `streamlit run app/fleet_monitor_v7.py`

Tests: `python -m unittest discover -s tests -v`

All sensor readings are synthetic. The dashboard does not connect to PLC/SCADA/IoT hardware, and generated work orders are drafts only. The priority thresholds are illustrative, not safety limits or certified maintenance decisions.

## v6 Advanced Upgrade

- Parametric 3D shaft visualization with editable dimensions
- 3D operating envelope: RPM / load / vibration
- Transparent weighted maintenance-priority score
- Sensor data quality audit helper
- Optional Streamlit demo dashboard
- Reproducibility and leakage-prevention guidance
- Export of v6 diagrams as interactive HTML

Run `pip install -r requirements-v6.txt`, open the v6 notebook, and execute cells in order.
Optional dashboard: `streamlit run app/streamlit_dashboard.py`.

All examples use synthetic/conceptual values. The 3D shaft is not a manufacturing CAD drawing.
The v5 stress contour is not validated FEA. No live hardware integration is claimed.
