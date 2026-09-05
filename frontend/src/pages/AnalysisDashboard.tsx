import { useState, useCallback } from 'react';
import SpectrumViewer from '../../../../frontend/src/components/SpectrumViewer';
import WaterfallViewer from '../../../../frontend/src/components/WaterfallViewer';
import ConstellationViewer from '../../../../frontend/src/components/ConstellationViewer';
import DataReadouts from '../../../../frontend/src/components/DataReadouts';
import HypothesisExplorer from '../../../../frontend/src/components/HypothesisExplorer';

interface AnalysisDashboardProps {
  onBack: () => void;
}

interface SignalData {
  spectrum: { frequency: number; magnitudeDb: number }[];
  waterfall: number[][];
  constellation: { i: number; q: number }[];
  parameters: {
    carrierFrequency: number;
    sampleRate: number;
    bandwidth: number;
    snr: number;
    symbolRate: number;
  };
  hypotheses: {
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
  }[];
}

export default function AnalysisDashboard({ onBack }: AnalysisDashboardProps) {
  const [signalData, setSignalData] = useState<SignalData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSynthetic, setIsSynthetic] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const validFile = files.find(f => 
      f.name.endsWith('.iq') || 
      f.name.endsWith('.IQ') || 
      f.name.endsWith('.wav') || 
      f.name.endsWith('.WAV')
    );

    if (!validFile) {
      alert('Please upload a .IQ or .wav file');
      return;
    }

    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', validFile);
    formData.append('is_synthetic', isSynthetic.toString());

    try {
      const uploadResponse = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const uploadResult = await uploadResponse.json();
      const analysisId = uploadResult.analysis_id;

      const [analysisRes, resultsRes] = await Promise.all([
        fetch(`http://localhost:8000/api/v1/analysis/${analysisId}`),
        fetch(`http://localhost:8000/api/v1/results/${analysisId}`)
      ]);

      const analysisData = await analysisRes.json();
      const resultsData = await resultsRes.json();

      setSignalData({
        spectrum: analysisData.spectrum,
        waterfall: analysisData.waterfall,
        constellation: analysisData.constellation,
        parameters: resultsData.parameters,
        hypotheses: resultsData.hypotheses
      });
    } catch (error) {
      console.error('Upload or analysis failed:', error);
      alert('Failed to process file. Make sure the backend server is running.');
    } finally {
      setIsUploading(false);
    }
  }, [isSynthetic]);

  const handleClear = () => {
    setSignalData(null);
  };

  if (!signalData) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-200">
        <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
          <button
            onClick={onBack}
            className="text-2xl font-bold tracking-wider hover:text-teal-400 transition-colors"
          >
            SIGMA
          </button>
        </header>

        <div className="flex items-center justify-center min-h-[calc(100vh-73px)] p-6">
          <div className="max-w-2xl w-full">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold mb-3 text-white">Signal Analysis Workstation</h2>
              <p className="text-slate-400">Upload .IQ or .wav files for automated modulation analysis</p>
            </div>

            <div className="mb-6 flex items-center justify-center gap-4">
              <button
                onClick={() => setIsSynthetic(false)}
                className={`px-6 py-2 rounded-lg font-mono text-sm transition-all ${
                  !isSynthetic
                    ? 'bg-teal-500 text-slate-900 font-bold'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                Real Recording
              </button>
              <button
                onClick={() => setIsSynthetic(true)}
                className={`px-6 py-2 rounded-lg font-mono text-sm transition-all ${
                  isSynthetic
                    ? 'bg-teal-500 text-slate-900 font-bold'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                Synthetic Signal
              </button>
            </div>

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-16 transition-all ${
                isDragging
                  ? 'border-teal-400 bg-teal-500/10'
                  : 'border-slate-700 bg-slate-800/50'
              } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <div className="text-center">
                <div className="mb-4">
                  <svg
                    className="w-16 h-16 mx-auto text-slate-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                </div>
                {isUploading ? (
                  <div className="text-teal-400 font-mono">PROCESSING...</div>
                ) : (
                  <>
                    <div className="text-lg font-semibold mb-2 text-slate-300">
                      Drop your signal file here
                    </div>
                    <div className="text-sm text-slate-500 font-mono">
                      Accepts .IQ and .WAV formats
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="mt-6 text-center">
              <div className="inline-flex items-center gap-2 text-xs text-slate-500 font-mono">
                <div className="w-2 h-2 bg-slate-700 rounded-full"></div>
                <span>Backend: http://localhost:8000</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <button
          onClick={handleClear}
          className="text-2xl font-bold tracking-wider hover:text-teal-400 transition-colors"
        >
          SIGMA
        </button>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-mono text-teal-400 tracking-wider">SYSTEM ONLINE</span>
          </div>
          <button
            onClick={onBack}
            className="px-4 py-2 text-sm font-mono bg-slate-800 hover:bg-slate-700 rounded transition-colors"
          >
            Exit
          </button>
        </div>
      </header>

      <div className="grid grid-cols-12 grid-rows-3 gap-4 p-6" style={{ height: 'calc(100vh - 73px)' }}>
        <div className="col-span-3 row-span-3 space-y-4">
          <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6 h-48">
            <div className="text-teal-400 font-mono text-xs tracking-wider mb-4">SIGNAL FILE</div>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-mono">Type</span>
                <span className="text-slate-200 font-mono">{isSynthetic ? 'Synthetic' : 'Real'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-mono">Status</span>
                <span className="text-teal-400 font-mono">Analyzed</span>
              </div>
            </div>
          </div>
          <DataReadouts
            carrierFrequency={signalData.parameters.carrierFrequency}
            sampleRate={signalData.parameters.sampleRate}
            bandwidth={signalData.parameters.bandwidth}
            snr={signalData.parameters.snr}
            symbolRate={signalData.parameters.symbolRate}
          />
        </div>

        <div className="col-span-6 row-span-1">
          <SpectrumViewer data={signalData.spectrum} />
        </div>

        <div className="col-span-3 row-span-1">
          <ConstellationViewer data={signalData.constellation} />
        </div>

        <div className="col-span-6 row-span-2">
          <WaterfallViewer data={signalData.waterfall} isLive={true} />
        </div>

        <div className="col-span-3 row-span-2">
          <HypothesisExplorer hypotheses={signalData.hypotheses} />
        </div>
      </div>
    </div>
  );
}
