# SIGMA Frontend Dashboard

This directory houses the React + TypeScript frontend GUI application for the SIGMA platform, scaffolded with Vite.

## Structure
* `public/`: Static asset directories.
* `src/components/`: Modular presentation components:
  * `FileUpload/`: Drag and drop file ingestion.
  * `SignalOverview/`: Metadata inspection.
  * `SpectrumViewer/`: PSD estimation lines.
  * `WaterfallViewer/`: 2D spectrogram waterfall.
  * `ConstellationViewer/`: I/Q complex symbol scatter plots.
  * `ParameterPanel/`: Estimated parameter lists (SNR, offset).
  * `HypothesisRanking/`: Hypothesis lists.
* `src/pages/`: AnalysisDashboard assembly page.
* `src/services/`: Client API layer (`api.ts`).
* `src/types/`: TypeScript interface definitions.

## Getting Started

### 1. Install dependencies
```bash
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```

The application will run locally on `http://localhost:5173`. It is configured to communicate with the FastAPI backend at `http://localhost:8000`.
