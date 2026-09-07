## SIGMA Quickstart Guide

### Backend Setup

1. Navigate to the backend directory:
```powershell
cd Sigma\backend
```

2. Create and activate virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Run the FastAPI server:
```powershell
python main.py
```

The backend will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Frontend Setup

1. Navigate to the frontend directory:
```powershell
cd Sigma\frontend
```

2. Install dependencies:
```powershell
npm install
```

3. Run the development server:
```powershell
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Using the Application

1. Open `http://localhost:5173` in your browser
2. Click "Launch Workstation" on the landing page
3. Toggle between "Real Recording" or "Synthetic Signal"
4. Drag and drop a .IQ or .wav file into the drop zone
5. View the analysis results:
   - Spectrum visualization (frequency vs magnitude)
   - Waterfall spectrogram (live scrolling)
   - Constellation diagram (I/Q scatter plot)
   - Signal parameters (carrier frequency, sample rate, bandwidth, SNR, symbol rate)
   - Hypothesis validation chain showing which modulation schemes passed all stages

### API Endpoints

- `POST /api/v1/upload` - Upload signal file
- `GET /api/v1/analysis/{analysis_id}` - Get spectrum, waterfall, constellation data
- `GET /api/v1/results/{analysis_id}` - Get parameters and hypothesis validation results

### Project Structure

```
Sigma/
├── backend/
│   ├── api/
│   │   ├── upload.py      (File upload endpoint)
│   │   ├── analysis.py    (Visualization data endpoint)
│   │   └── results.py     (Parameters and hypothesis endpoint)
│   ├── core/
│   │   └── config.py      (Application settings)
│   └── main.py            (FastAPI application entry point)
└── frontend/
    ├── src/
    │   ├── components/    (Reusable UI components)
    │   ├── pages/         (LandingView and AnalysisDashboard)
    │   └── App.tsx        (Main application with view routing)
    └── package.json
```

### Tech Stack

**Backend:**
- Python FastAPI
- Pydantic v2
- Uvicorn ASGI server

**Frontend:**
- React 19 + TypeScript
- Vite 8
- Tailwind CSS 3
- Custom SVG/Canvas visualizations

### Design Philosophy

SIGMA implements a closed-loop validation approach to signal analysis. Unlike traditional systems that rely solely on ML confidence scores, SIGMA physically validates each hypothesis by attempting:

1. Carrier and timing synchronization
2. Symbol demodulation
3. Forward Error Correction (Viterbi) decoding

Only hypotheses that pass all validation stages are marked as proven. This provides explainable evidence rather than statistical guesses.
