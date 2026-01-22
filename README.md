# SmallBiz Talks — Streamlit App Packaging & Operations Guide


This document is the authoritative reference for:
- how the SmallBiz Talks app works
- how KPIs are generated and updated
- how plots and CSVs are produced
- how to build a self-contained Windows EXE using PyInstaller
- how to safely ship the app to non-technical users

This process is NOT theoretical. It has been:
- built end-to-end
- debugged across many failure modes
- tested on clean machines
- confirmed stable

------------------------------------------------------------
INTENDED USE & DATA ACCESS
------------------------------------------------------------


This application is designed for **internal use within the SBDC network** to support operational reporting, analysis, and data reconciliation workflows.

Because it operates on **confidential client and attendee records**, the underlying datasets are **not included** in this repository and are **not publicly accessible**. As a result, the application may not be directly runnable or meaningful outside of its intended environment.

Within the organization, however, it provides significant value by:
- automating previously manual processes
- improving data consistency across systems
- enabling reliable KPI tracking and trend analysis
- supporting staff with repeatable, auditable workflows

------------------------------------------------------------
SCOPE AND AUDIENCE
------------------------------------------------------------

**Primary audience**
- SBDC staff
- analysts
- technical collaborators supporting SBDC operations

**Scope**
- webinar attendance analysis
- CRM matching and enrichment
- KPI generation and visualization
- internal reporting workflows

This project prioritizes **correctness, repeatability, and internal usability** over general-purpose reuse. It is not intended to be a plug-and-play public application.

------------------------------------------------------------
PROJECT GOAL
------------------------------------------------------------

Deliver a **local-only Streamlit application** that:

- processes Zoom webinar attendance files
- matches attendees to NeoSerra CRM
- maintains master tables over time
- computes longitudinal KPIs
- generates professional charts
- generates geographic visualizations for client and non-client attendance

- runs via **double-click `.exe`**
- requires **no Python installation**

------------------------------------------------------------
FINAL RESULT
------------------------------------------------------------

- `WebinarApp.exe` launches Streamlit locally
- Browser opens automatically
- App shuts down cleanly when browser closes
- All CSV outputs persist to disk
- KPIs update incrementally
- Graphs and maps are saved & overwritten cleanly
- Tested on bare-bones Windows machines

------------------------------------------------------------
CORE DESIGN PRINCIPLES
------------------------------------------------------------

1. Streamlit must be launched via CLI (not `bootstrap.run`)
2. Folder-based PyInstaller builds (`--onedir`, not `--onefile`)
3. External data lives next to the EXE
4. Master tables are append-only with deduplication
5. KPIs are computed from masters, not per-session
6. Plots are generated once and saved
7. New runs never corrupt prior logic

------------------------------------------------------------
FINAL SHIPPED FOLDER STRUCTURE
------------------------------------------------------------

You must ship **the entire folder below (zipped)**:

```
dist/
  WebinarApp/
    WebinarApp.exe
    _internal/
    run_app.py
    app.py
    scripts/
      run_webinar_neoserra_match.py
      attendance_cleaning.py
      master_tables.py
      kpis.py
      attendance_plots.py
      center_loading.py
      zip_codes.py
      never_attended.py
    data/
      centers.csv
    kpis/
      webinar_kpis.csv
      *.png
      never_attended.csv
    .streamlit/
      config.toml
    hooks/
      hook-google.py
      hook-streamlit.py
```

DO NOT:
- ship only the `.exe`
- move files into `_internal/`
- edit anything inside `_internal/`

------------------------------------------------------------
PIPELINE OVERVIEW
------------------------------------------------------------

### Inputs
- Zoom webinar attendee CSV(s)
- NeoSerra client export
- Centers lookup

### Cleaning
- email normalization
- attendance normalization
- deduplication per webinar

### Matching
- Zoom → NeoSerra via email
- fallback ZIP → center mapping

### Master Tables
- `people_master.csv`
- `attendance_master.csv`
- safe upserts via `_attendance_key`

### Geographic Mapping Outputs
The application produces two separate geographic maps to support center-level analysis:

**1. Client Distribution Map**
- Includes attendees identified as existing clients
- Used to understand geographic spread of current client participation


**2. Non-Client Center Assignment Map**
- Includes attendees not matched to existing clients
- Points (Zip codes) are color-coded by assignment to nearest center


------------------------------------------------------------
KPI DEFINITIONS (STANDING AUDIENCE MODEL)
------------------------------------------------------------

People register ONCE and receive all future invites.

| Metric | Definition |
|------|-----------|
| Total Audience | cumulative unique registrations up to date |
| Attendees | people who attended that webinar |
| No-Shows | audience − attendees |
| Engagement Rate | attendees / total audience |
| First-Time | first-ever attendance |
| Repeat | attended previously |
| Client Share | client attendees / attendees |

------------------------------------------------------------
VIRTUAL ENV SETUP
------------------------------------------------------------
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------
PYINSTALLER BUILD COMMAND
------------------------------------------------------------

BUILD:
```
.\.venv\Scripts\pyinstaller `
  --name WebinarApp `
  --noconsole `
  --additional-hooks-dir hooks `
  --hidden-import pgeocode `
  --collect-all folium `
  run_app.py
```
------------------------------------------------------------
ADD NECESSARY FOLDERS   
------------------------------------------------------------
When PyInstaller finishes, it creates:
```
dist/
  WebinarApp/
    WebinarApp.exe
    _internal/
```
The folder is NOT complete yet. You must manually add project files next to the EXE.


Files and Folders You MUST Place Inside `dist/WebinarApp`
After building the EXE, ensure the following structure exists:
```
dist/
  WebinarApp/
    WebinarApp.exe          ← user launches this
    _internal/              ← PyInstaller runtime (DO NOT TOUCH)

    app.py                  ← Streamlit app entrypoint
    run_app.py              ← Streamlit launcher

    scripts/                ← ALL pipeline logic
      __init__.py
      run_webinar_neoserra_match.py
      attendance_cleaning.py
      master_tables.py
      kpis.py
      attendance_plots.py
      zip_codes.py
      center_loading.py
      never_attended.py
      ...

    data/
      centers.csv            ← required reference data

    .streamlit/
      config.toml            ← Streamlit configuration
```

------------------------------------------------------------
PACKAGING FOR DELIVERY
------------------------------------------------------------

1. Zip the folder:
   `dist/WebinarApp/`
2. Send ZIP
3. Recipient unzips
4. Double-click `WebinarApp.exe`

Use .zip format — NOT .7z or .rar

------------------------------------------------------------
STATUS
------------------------------------------------------------

Works on my machine.  
Works on their machine.  