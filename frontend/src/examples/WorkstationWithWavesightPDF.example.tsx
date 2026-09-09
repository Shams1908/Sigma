import { useRef } from 'react';
import WavesightExportButton from '../components/WavesightExportButton';

export default function WorkstationWithWavesightPDFExample() {
  
  const spectrumContainerRef = useRef<HTMLDivElement>(null);
  const waterfallContainerRef = useRef<HTMLDivElement>(null);
  const waveformContainerRef = useRef<HTMLDivElement>(null);
  const constellationContainerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="workspace-grid">
      
      <div ref={spectrumContainerRef} className="spectrum-container">
        {/* Your SpectrumViewer or InteractiveSpectrum component */}
      </div>

      <div ref={waterfallContainerRef} className="waterfall-container">
        {/* <WaterfallViewer data={waterfallData} isLive={true} /> */}
      </div>

      <div ref={constellationContainerRef} className="constellation-container">
        {/* <ConstellationViewer data={constellationData} /> */}
      </div>

      <div ref={waveformContainerRef} className="waveform-container">
        {/* <WaveformViewer iData={...} qData={...} timeData={...} sampleRate={...} /> */}
      </div>

      <div className="export-toolbar">
        <WavesightExportButton
          signalId="example-signal-id-001"
          fileName="test_signal.iq"
          parameters={{
            sampleRate: 250000,
            bandwidth: 50000,
            snr: 15.5,
            symbolRate: 9600,
            carrierOffset: 1250
          }}
          hypotheses={[
            {
              id: 'hyp-1',
              modulation: 'QPSK',
              symbolRate: 9600,
              mlConfidence: 0.89,
              validation: {
                syncPassed: true,
                demodPassed: true,
                fecPassed: false
              },
              isWinner: true
            },
            {
              id: 'hyp-2',
              modulation: 'BPSK',
              symbolRate: 4800,
              mlConfidence: 0.72,
              validation: {
                syncPassed: true,
                demodPassed: false,
                fecPassed: false
              },
              isWinner: false
            }
          ]}
          diagnostics={{
            evm_rms: 0.085,
            timing_error_rms: 0.012,
            sync_locked: true,
            demod_locked: true,
            fec_valid: false,
            snr: 15.5,
            carrier_offset: 1250
          }}
          spectrumRef={spectrumContainerRef}
          waterfallRef={waterfallContainerRef}
          waveformRef={waveformContainerRef}
          constellationRef={constellationContainerRef}
          iData={Array(1000).fill(0).map(() => Math.random() * 2 - 1)}
          qData={Array(1000).fill(0).map(() => Math.random() * 2 - 1)}
          disabled={false}
        />
      </div>
    </div>
  );
}
