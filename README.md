# SHESHIELD — Women Safety Risk Prediction (Final Year Project)

This zip contains **all 3 parts integrated**:
1) **Model training** (creates an extended/large dataset + trains ML model)
2) **Backend** (Flask API + server)
3) **Frontend** (HTML/CSS/JS served by Flask, uses the **real-time API**)

## Folder structure

```
sheshield/
  backend/
    app.py
    model_training.py
    requirements.txt
    data/
      data.csv
      data_extended.csv        (generated after training)
    model/
      model.pkl               (generated after training)
    templates/
      index.html
      map.html                (generated after training if --generate-assets)
    static/
      css/style.css
      js/app.js
      images/                 (charts from training, optional)
  scripts/
    sanity_test.py
```

## Setup (Windows / macOS / Linux)

### 1) Install dependencies

Open a terminal inside `sheshield/backend`:

```bash
python -m venv .venv
```

Activate:
- **Windows (PowerShell)**: `.venv\\Scripts\\Activate.ps1`
- **Windows (CMD)**: `.venv\\Scripts\\activate.bat`
- **macOS/Linux**: `source .venv/bin/activate`

Install:
```bash
pip install -r requirements.txt
```

### 2) Train the model (and generate a large dataset)

```bash
python model_training.py --target-rows 5000 --generate-assets
```

Outputs:
- `backend/model/model.pkl`
- `backend/data/data_extended.csv`
- `backend/templates/map.html` (interactive map)
- `backend/static/images/*.png` (charts)

### 3) Run the backend + frontend

```bash
python app.py
```

Open:
- http://127.0.0.1:5000

## Real-time prediction API

### Endpoint
`POST /api/predict`

Example body:
```json
{
  "crime_rate": 7,
  "lighting": "Poor",
  "police_distance": 2.5,
  "time_of_day": "Night",
  "crowd_density": "Low"
}
```

## Contact Us (demo endpoint)

`POST /api/contact` saves messages to: `backend/data/contact_messages.csv`

## One-command quick test

In a **second terminal**, from the `sheshield` folder:
```bash
python scripts/sanity_test.py
```

It will call the API and print the prediction.
