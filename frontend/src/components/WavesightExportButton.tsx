import { RefObject } from 'react';
import { FileDown } from 'lucide-react';
import { exportWavesightPDF } from './WavesightPDFExporter';

interface ExportParameters {
  sampleRate: number;
  bandwidth: number;
  snr: number;
  symbolRate: number;
  carrierOffset: number;
}

interface Hypothesis {
  id: string;
  modulation: string;
  symbolRate: number;
  mlConfidence: number;
  validation: {
    syncPassed: boolean;
    demodPassed: boolean;
    fecPassed: boolean;
  };
  isWinner: boolean;
}

interface Diagnostics {
  evm_rms: number | null;
  timing_error_rms: number | null;
  sync_locked: boolean;
  demod_locked: boolean;
  fec_valid: boolean;
  snr: number;
  carrier_offset: number;
}

interface WavesightExportButtonProps {
  signalId: string;
  fileName: string;
  parameters: ExportParameters | null;
  hypotheses: Hypothesis[];
  diagnostics: Diagnostics | null;
  spectrumRef: RefObject<HTMLDivElement>;
  waterfallRef: RefObject<HTMLDivElement>;
  waveformRef: RefObject<HTMLDivElement>;
  constellationRef: RefObject<HTMLDivElement>;
  iData?: number[];
  qData?: number[];
  disabled?: boolean;
}

export default function WavesightExportButton({
  signalId,
  fileName,
  parameters,
  hypotheses,
  diagnostics,
  spectrumRef,
  waterfallRef,
  waveformRef,
  constellationRef,
  iData,
  qData,
  disabled = false
}: WavesightExportButtonProps) {
  
  const handleExport = async () => {
    if (!parameters) {
      alert('Cannot export: Signal parameters not available');
      return;
    }

    try {
      await exportWavesightPDF(
        signalId,
        fileName,
        parameters,
        hypotheses,
        diagnostics,
        {
          spectrumRef,
          waterfallRef,
          waveformRef,
          constellationRef
        },
        iData,
        qData
      );
    } catch (error) {
      console.error('PDF Export failed:', error);
      alert(`PDF Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={disabled || !parameters}
      className="flex items-center gap-2 px-4 py-2 bg-[#00E5FF] text-black font-mono text-sm font-bold rounded hover:bg-[#00D5EF] disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition-all"
      title="Export WAVESIGHT Forensics Report"
    >
      <FileDown className="w-4 h-4" strokeWidth={2} />
      EXPORT PDF REPORT
    </button>
  );
}
