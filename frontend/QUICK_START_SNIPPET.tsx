import { useRef } from 'react';
import WavesightExportButton from '../components/WavesightExportButton';

const spectrumContainerRef = useRef<HTMLDivElement>(null);
const waterfallContainerRef = useRef<HTMLDivElement>(null);
const waveformContainerRef = useRef<HTMLDivElement>(null);
const constellationContainerRef = useRef<HTMLDivElement>(null);

<div ref={spectrumContainerRef}>
  <SpectrumViewer data={spectrumData} />
</div>

<div ref={waterfallContainerRef}>
  <WaterfallViewer data={waterfallData} isLive={true} />
</div>

<div ref={constellationContainerRef}>
  <ConstellationViewer data={constellationData} />
</div>

<div ref={waveformContainerRef}>
  <WaveformViewer 
    iData={waveformData.i}
    qData={waveformData.q}
    timeData={waveformData.time}
    sampleRate={waveformData.sampleRate}
  />
</div>

<WavesightExportButton
  signalId={analysisId || 'UNKNOWN'}
  fileName={uploadedFile?.name || 'signal.iq'}
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
