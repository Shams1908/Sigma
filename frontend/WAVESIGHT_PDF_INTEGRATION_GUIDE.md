# WAVESIGHT PDF Report Integration Guide

## Overview

This guide explains how to integrate the new WAVESIGHT PDF export functionality into your Workstation component.

## What's Been Created

1. **WavesightPDFExporter.tsx** - Core PDF generation logic with forensic aesthetics
2. **WavesightExportButton.tsx** - React component for triggering PDF export
3. **canvasExtractor.ts** - Utility for extracting canvas elements from visualization components

## Features Implemented

### ✅ ZERO-TOLERANCE PROTOCOLS
- Project renamed from SIGMA to WAVESIGHT throughout PDF
- Image extraction happens **outside** React render cycle using `toDataURL()`
- No modifications to visualization component render loops

### ✅ INTELLIGENCE DOSSIER AESTHETICS
- Deep black background (#050505)
- Neon teal (#00E5FF) headers
- Purple (#B200FF) accents for statistical data
- Monospace (Courier) fonts for all data
- Barcode-style Signal ID placeholder

### ✅ FORENSIC ANALYTICS INJECTION
- All existing metrics preserved (Sample Rate, Bandwidth, SNR, Symbol Rate, Carrier Offset)
- **NEW SECTION**: Statistical Fingerprint
  - Crest Factor (Peak-to-Average Power Ratio)
  - Kurtosis (Signal Impulsiveness)
  - Skewness (Asymmetry)
- **Automated Executive Summary** with algorithmic assessment

### ✅ HIGH-FIDELITY VISUAL EXPORT
- Captures 2D Spectrum, Waterfall, Time Domain, Constellation
- Arranged in 2x2 grid with proper scaling (object-fit: contain)
- Direct canvas capture via `toDataURL('image/jpeg', 0.9)`

## Integration Steps

### Step 1: Install Dependencies

The required packages have been added to `package.json`:
```json
"jspdf": "^2.5.2",
"jspdf-autotable": "^3.8.4"
```

Run:
```bash
npm install
```

### Step 2: Add Visualization Refs to Workstation

Add refs for each visualization component container:

```typescript
// In Workstation.tsx, add these refs at the top of the component
const spectrumContainerRef = useRef<HTMLDivElement>(null);
const waterfallContainerRef = useRef<HTMLDivElement>(null);
const waveformContainerRef = useRef<HTMLDivElement>(null);
const constellationContainerRef = useRef<HTMLDivElement>(null);
```

### Step 3: Attach Refs to Visualization Containers

Wrap each visualization component with a ref:

```typescript
{/* Spectrum - find InteractiveSpectrum or SpectrumViewer and wrap */}
<div ref={spectrumContainerRef}>
  <InteractiveSpectrum ... />
</div>

{/* Waterfall */}
<div ref={waterfallContainerRef}>
  <WaterfallViewer data={waterfallData} isLive={true} />
</div>

{/* Constellation */}
<div ref={constellationContainerRef}>
  <ConstellationViewer data={constellationData} />
</div>

{/* Waveform */}
<div ref={waveformContainerRef}>
  <WaveformViewer 
    iData={waveformData.i}
    qData={waveformData.q}
    timeData={waveformData.time}
    sampleRate={waveformData.sampleRate}
  />
</div>
```

### Step 4: Import and Use WavesightExportButton

```typescript
import WavesightExportButton from '../components/WavesightExportButton';

// In your Export Toolbar or wherever you want the button:
<WavesightExportButton
  signalId={analysisId || 'UNKNOWN'}
  fileName={uploadedFile?.name || 'unknown_signal.iq'}
  parameters={signalParams ? {
    sampleRate: signalParams.sampleRate,
    bandwidth: signalParams.bandwidth,
    snr: signalParams.snr,
    symbolRate: signalParams.symbolRate,
    carrierOffset: signalParams.carrierFrequency
  } : null}
  hypotheses={hypotheses}
  diagnostics={diagnostics}
  spectrumRef={spectrumContainerRef}
  waterfallRef={waterfallContainerRef}
  waveformRef={waveformContainerRef}
  constellationRef={constellationContainerRef}
  iData={waveformData?.i}
  qData={waveformData?.q}
  disabled={!analysisId || !signalParams}
/>
```

### Step 5: Replace Existing PDF Export

Find the current PDF export button (likely in ExportToolbar) and replace:

```typescript
// OLD CODE:
onExportReport={async () => {
  if (!analysisId) return;
  try {
    window.open(`http://localhost:8000/api/v1/results/${analysisId}/pdf`, '_blank');
  } catch (err) {
    console.error('Export PDF failed:', err);
  }
}}

// NEW CODE: Use WavesightExportButton component instead
```

## Data Flow

```
User clicks "EXPORT PDF REPORT"
    ↓
WavesightExportButton.handleExport()
    ↓
exportWavesightPDF() in WavesightPDFExporter
    ↓
1. Create jsPDF document with dark theme
2. Add WAVESIGHT header with Signal ID barcode
3. Generate executive summary from diagnostics
4. Add signal parameters table
5. Calculate & add statistical fingerprint (Crest Factor, Kurtosis, Skewness)
6. Add diagnostics table
7. Capture canvas images from each visualization
8. Add 2x2 image grid on second page
9. Add hypothesis validation results table
10. Add footer to all pages
    ↓
PDF downloaded: WAVESIGHT_{signalId}_{timestamp}.pdf
```

## Canvas Extraction

The `canvasExtractor.ts` utility automatically finds canvas elements within component containers:

```typescript
// Searches DOM tree for canvas elements
findCanvasInComponent(containerRef.current)
    ↓
// Captures as JPEG with 90% quality
canvas.toDataURL('image/jpeg', 0.9)
```

This approach:
- ✅ Doesn't modify component internals
- ✅ Works with any visualization using canvas
- ✅ Happens outside render cycle
- ✅ Prevents WebGL/Canvas crashes

## Styling & Theme

The PDF uses WAVESIGHT's dark forensic theme:

| Element | Color | Usage |
|---------|-------|-------|
| Background | #050505 | Page background |
| Headers | #00E5FF (Teal) | Section titles, main parameters |
| Accents | #B200FF (Purple) | Statistical data, special metrics |
| Text | #DCDCDC (Light Gray) | Body text |
| Tables | Monospace (Courier) | All data readouts |

## Testing

1. Upload a signal file
2. Wait for analysis to complete
3. Click "EXPORT PDF REPORT"
4. Verify PDF contains:
   - ✅ WAVESIGHT branding (not SIGMA)
   - ✅ Signal ID with barcode placeholder
   - ✅ Executive summary
   - ✅ All signal parameters
   - ✅ Statistical fingerprint section
   - ✅ Four visualization images in 2x2 grid
   - ✅ Hypothesis validation table
   - ✅ Dark theme throughout

## Troubleshooting

### PDF shows blank images
- Ensure visualization components have rendered before export
- Check browser console for canvas.toDataURL() errors
- Verify refs are attached to correct DOM elements

### TypeScript errors
- Run `npm install` to ensure jspdf types are installed
- Check all interface definitions match your data structures

### Canvas security errors
- Canvas must not have CORS-tainted content
- Ensure all images loaded into canvas are from same origin

## Backend Integration

The backend PDF exporter (`backend/reporting/pdf_export.py`) has also been updated:
- ✅ "SIGMA" → "WAVESIGHT" in title
- ✅ Footer updated to WAVESIGHT branding

Both frontend and backend exporters can coexist. The frontend version provides:
- Better control over styling
- Direct access to live canvas state
- No server round-trip required
- Richer statistical analysis

## Future Enhancements

Consider adding:
- [ ] User-configurable color themes
- [ ] Custom logo/watermark upload
- [ ] Multi-page hypothesis details
- [ ] Embedded metadata (PDF/A compliance)
- [ ] Digital signature support
- [ ] Export to other formats (HTML, DOCX)
