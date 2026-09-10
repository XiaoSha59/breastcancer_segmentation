# Reproducibility backend

Small local FastAPI backend for the interactive research demo.

## What it actually runs

The browser cannot send arbitrary shell commands. The backend launches one fixed script only:

```text
reproducibility/run_live_audit.py
```

That script performs a real, lightweight audit of the committed research artifacts:

1. reconstructs the leakage-controlled external cohort,
2. checks timing/phase-selection summary counts,
3. checks external performance values,
4. checks LCC postprocessing results,
5. recomputes canonical failure-attribution statistics,
6. runs the final manuscript-number verification.

It does **not** rerun nnU-Net training or GPU inference.

## Local development

From the MRI-Project root:

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility_backend/requirements.txt
python -m uvicorn reproducibility_backend.app:app --host 127.0.0.1 --port 8768 --reload
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r reproducibility_backend/requirements.txt
python -m uvicorn reproducibility_backend.app:app --host 127.0.0.1 --port 8768 --reload
```

Health check:

```text
http://127.0.0.1:8768/api/health
```

The Vite frontend defaults to this API address.

## One-process production-style local demo

First build the frontend:

```bash
cd reproducibility_site
npm install
npm run build
cd ..
```

Then run FastAPI without `--reload`:

```bash
python -m uvicorn reproducibility_backend.app:app --host 127.0.0.1 --port 8768
```

Open:

```text
http://127.0.0.1:8768/
```

FastAPI serves the compiled `reproducibility_site/dist/` automatically when it exists.

## Security note

Keep this backend bound to `127.0.0.1` for now. Do not expose the live-run endpoint publicly without adding authentication/rate limiting. The current API intentionally accepts no arbitrary command or file path from the browser.
