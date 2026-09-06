import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Topography from '../components/Topography';
import SpectrumViewer from '../components/SpectrumViewer';
import WaterfallViewer from '../components/WaterfallViewer';
import ConstellationViewer from '../components/ConstellationViewer';
import DataReadouts from '../components/DataReadouts';
import HypothesisExplorer from '../components/HypothesisExplorer';
import WaveformViewer from '../components/WaveformViewer';
import DiagnosticsPanel from '../components/DiagnosticsPanel';
import ExportToolbar from '../components/ExportToolbar';
import * as api from '../api/sigma';

interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

interface ConstellationPoint {
  i: number;
  q: number;
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

export default function Workstation() {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [spectrumData, setSpectrumData] = useState<SpectrumPoint[]>([]);
  const [waterfallData, setWaterfallData] = useState<number[][]>([]);
  const [constellationData, setConstellationData] = useState<ConstellationPoint[]>([]);
  const [signalParams, setSignalParams] = useState<{
    carrierFrequency: number;
    sampleRate: number;
    bandwidth: number;
    snr: number;
    symbolRate: number;
  } | null>(null);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [waveformData, setWaveformData] = useState<{ i: number[]; q: number[]; time: number[]; sampleRate: number } | null>(null);
  const [diagnostics, setDiagnostics] = useState<{
    evm_rms: number | null;
    timing_error_rms: number | null;
    sync_locked: boolean;
    demod_locked: boolean;
    fec_valid: boolean;
    snr: number;
    carrier_offset: number;
  } | null>(null);

  // File upload handlers
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.endsWith('.iq') || file.name.endsWith('.wav')) {
        setUploadedFile(file);
        handleFileUpload(file);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setUploadedFile(files[0]);
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setAnalysisProgress(0);
    
    try {
      console.log('Uploading file:', file.name);
      const uploadResponse = await api.uploadFile(file);
      console.log('Upload response:', uploadResponse);
      
      const signalId = uploadResponse.signal_id;

      console.log('Fetching real-time visualizations for signal:', signalId);
      setAnalysisProgress(20);
      
      try {
        console.log('Fetching spectrum...');
        const spectrum = await api.getSpectrum(signalId);
        console.log('Spectrum received:', spectrum.data?.length, 'points');
        setSpectrumData(spectrum.data);
      } catch (specErr) {
        console.error('Spectrum fetch failed:', specErr);
      }
      
      try {
        console.log('Fetching waterfall...');
        const waterfall = await api.getWaterfall(signalId);
        console.log('Waterfall received:', waterfall.data?.length, 'rows');
        setWaterfallData(waterfall.data);
      } catch (waterfallErr) {
        console.error('Waterfall fetch failed:', waterfallErr);
      }
      
      try {
        console.log('Fetching constellation...');
        const constellation = await api.getConstellation(signalId);
        console.log('Constellation received:', constellation.data?.length, 'points');
        setConstellationData(constellation.data);
      } catch (constErr) {
        console.error('Constellation fetch failed:', constErr);
      }
      
      try {
        console.log('Fetching real DSP parameters...');
        const params = await api.getSignalParameters(signalId);
        console.log('Parameters received:', params);
        setSignalParams(params);
      } catch (paramErr) {
        console.error('Parameters fetch failed:', paramErr);
      }
      
      try {
        console.log('Fetching waveform...');
        const waveform = await api.getWaveform(signalId);
        console.log('Waveform received:', waveform.i?.length, 'samples');
        setWaveformData(waveform);
      } catch (waveformErr) {
        console.error('Waveform fetch failed:', waveformErr);
      }
      
      setAnalysisProgress(70);

      console.log('Starting full hypothesis analysis for signal ID:', signalId);
      console.log('(This may take time or fail if ML model is not available)');
      
      try {
        const analysisResponse = await api.startAnalysis(signalId);
        const analysisId = analysisResponse.analysis_id;
        setAnalysisId(analysisId);
        console.log('Analysis ID:', analysisId, 'Status:', analysisResponse.status);

        console.log('Polling analysis status (max 30 seconds)...');
        const finalStatus = await api.pollAnalysisStatus(
          analysisId,
          (status) => {
            console.log('Analysis status update:', status);
            const progress = 70 + (status.progress || 0) * 0.3;
            setAnalysisProgress(progress);
          },
          30000,
          2000
        );

        console.log('Analysis complete. Final status:', finalStatus);

        if (finalStatus.status === 'done') {
          console.log('Fetching hypothesis results from analysis...');
          const results = await api.getResults(analysisId);
          console.log('Hypothesis results:', results);

          if (results.hypotheses?.hypotheses) {
            setHypotheses(results.hypotheses.hypotheses);
          }
          
          try {
            console.log('Fetching diagnostics...');
            const diag = await api.getDiagnostics(analysisId);
            console.log('Diagnostics received:', diag);
            setDiagnostics(diag);
          } catch (diagErr) {
            console.error('Diagnostics fetch failed:', diagErr);
          }
        } else {
          console.warn('Analysis did not complete successfully:', finalStatus.status);
        }
      } catch (analysisErr) {
        console.warn('Full hypothesis analysis not available (ML model may be missing):', analysisErr);
        console.log('Showing DSP-only results without hypothesis validation');
      }

      setAnalysisProgress(100);
      setIsAnalyzing(false);
    } catch (err) {
      console.error('Upload error:', err);
      setIsAnalyzing(false);
      
      let errorMessage = 'Upload failed';
      if (err instanceof api.APIError) {
        errorMessage = err.message;
        if (err.status === 404) {
          errorMessage = 'Backend not found. Is the server running on localhost:8000?';
        } else if (err.status === 0 || errorMessage.includes('Network error')) {
          errorMessage = 'Cannot connect to backend. Make sure FastAPI server is running on localhost:8000';
        }
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    }
  };

  return (
    <div className="relative min-h-screen bg-black text-white">
      {/* WebGL Topography Background */}
      <div className="fixed inset-0 z-0 opacity-30">
        <Topography
          lowColor="#0a0014"
          midColor="#6d28d9"
          highColor="#d8b4fe"
          speed={0.2}
          bands={1.5}
          thickness={0.01}
          glow={0.5}
          pixelSize={1.0}
          mouseInteraction={false}
        />
        <div 
          className="absolute inset-0" 
          style={{
            background: 'radial-gradient(circle at center, transparent 0%, rgba(0,0,0,0.8) 70%, rgb(0,0,0) 100%)'
          }}
        />
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        {/* Header */}
        <motion.header
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="border-b border-[#222222] bg-[#0A0A0A]/80 backdrop-blur-sm"
        >
          <div className="max-w-[1800px] mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-2xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
                SIGMA
              </div>
              <div className="h-6 w-[1px] bg-[#222222]" />
              <div className="text-sm text-gray-400 font-mono">WORKSTATION</div>
            </div>
            
            <div className="flex items-center gap-4">
              {uploadedFile && (
                <div className="flex items-center gap-2 bg-[#111111] border border-[#222222] rounded-lg px-4 py-2">
                  <div className="w-2 h-2 bg-sigma-teal rounded-full animate-pulse" />
                  <span className="text-xs font-mono text-gray-400">{uploadedFile.name}</span>
                </div>
              )}
              
              <motion.button
                onClick={() => navigate('/')}
                className="px-4 py-2 bg-[#111111] border border-[#222222] hover:border-sigma-purple text-sm font-semibold rounded-lg transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Back to Home
              </motion.button>
            </div>
          </div>
        </motion.header>

        {/* Main Workspace */}
        <div className="max-w-[1800px] mx-auto p-6 space-y-6">
          {/* File Upload Zone */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300
              ${isDragging 
                ? 'border-sigma-teal bg-sigma-teal/5 scale-[1.02]' 
                : 'border-sigma-teal/30 hover:border-sigma-teal/60 bg-[#0A0A0A]'
              }
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".iq,.wav"
              onChange={handleFileSelect}
              className="hidden"
            />
            
            {isAnalyzing ? (
              <div className="space-y-4">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="w-16 h-16 mx-auto border-4 border-sigma-teal/20 border-t-sigma-teal rounded-full"
                />
                <p className="text-lg font-semibold text-sigma-teal">Analyzing Signal...</p>
                <p className="text-sm text-gray-400 font-mono">{uploadedFile?.name}</p>
                <div className="w-full max-w-xs mx-auto bg-[#111111] rounded-full h-2 overflow-hidden">
                  <motion.div
                    className="h-full bg-sigma-teal"
                    initial={{ width: 0 }}
                    animate={{ width: `${analysisProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-xs text-gray-500 font-mono">{analysisProgress}% Complete</p>
              </div>
            ) : error ? (
              <div className="space-y-4">
                <div className="text-6xl">⚠️</div>
                <div>
                  <p className="text-xl font-semibold text-red-400 mb-2">Upload Failed</p>
                  <p className="text-sm text-gray-400">{error}</p>
                  <motion.button
                    onClick={() => {
                      setError(null);
                      setUploadedFile(null);
                    }}
                    className="mt-4 px-6 py-2 bg-[#111111] border border-[#222222] hover:border-red-500 text-sm font-semibold rounded-lg transition-colors"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Try Again
                  </motion.button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="text-6xl"
                >
                  📡
                </motion.div>
                <div>
                  <p className="text-xl font-semibold text-white mb-2">
                    Drop your signal file here
                  </p>
                  <p className="text-sm text-gray-400 mb-4">
                    Supports .IQ and .WAV files
                  </p>
                  <motion.button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-8 py-3 bg-sigma-teal hover:bg-sigma-teal/80 text-black font-bold rounded-lg transition-colors"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Browse Files
                  </motion.button>
                </div>
              </div>
            )}
          </motion.div>

          {/* Bento Grid Layout */}
          <div className="grid grid-cols-12 gap-6">
            {/* Spectrum Viewer - Large */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="col-span-12 lg:col-span-8 h-[400px]"
            >
              <SpectrumViewer data={spectrumData} />
            </motion.div>

            {/* Data Readouts */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="col-span-12 lg:col-span-4 h-[400px]"
            >
              {signalParams ? (
                <DataReadouts {...signalParams} />
              ) : (
                <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="text-4xl">📊</div>
                    <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Waterfall Viewer */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="col-span-12 lg:col-span-6 h-[400px]"
            >
              <WaterfallViewer data={waterfallData} isLive={true} />
            </motion.div>

            {/* Constellation Viewer */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="col-span-12 lg:col-span-6 h-[400px]"
            >
              <ConstellationViewer data={constellationData} />
            </motion.div>

            {/* Waveform Viewer */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="col-span-12 lg:col-span-8 h-[350px]"
            >
              {waveformData ? (
                <WaveformViewer 
                  iData={waveformData.i}
                  qData={waveformData.q}
                  timeData={waveformData.time}
                  sampleRate={waveformData.sampleRate}
                />
              ) : (
                <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="text-4xl">〰️</div>
                    <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Diagnostics Panel */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
              className="col-span-12 lg:col-span-4 h-[350px]"
            >
              {diagnostics ? (
                <DiagnosticsPanel {...diagnostics} />
              ) : (
                <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="text-4xl">🔧</div>
                    <p className="text-gray-400 text-sm">Analysis Pending</p>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Export Toolbar */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 }}
              className="col-span-12"
            >
              <ExportToolbar
                analysisId={analysisId}
                fileName={uploadedFile?.name || ''}
                onExportReport={async () => {
                  if (!analysisId) return;
                  try {
                    const response = await fetch(`http://localhost:8000/api/v1/results/${analysisId}/report`);
                    const data = await response.json();
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `sigma_report_${analysisId.substring(0, 8)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  } catch (err) {
                    console.error('Export report failed:', err);
                  }
                }}
                onExportJSON={async () => {
                  if (!analysisId) return;
                  try {
                    const response = await fetch(`http://localhost:8000/api/v1/results/${analysisId}`);
                    const data = await response.json();
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `sigma_results_${analysisId.substring(0, 8)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  } catch (err) {
                    console.error('Export JSON failed:', err);
                  }
                }}
                onExportBits={async () => {
                  if (!analysisId) return;
                  try {
                    const response = await fetch(`http://localhost:8000/api/v1/results/${analysisId}/bitstream`);
                    const data = await response.json();
                    if (!data.available) {
                      alert(`Bitstream not available: ${data.reason}`);
                      return;
                    }
                    const content = `Bitstream Export\n${'='.repeat(50)}\nTotal Bits: ${data.total_bits}\nEntropy: ${data.entropy}\nOnes Ratio: ${data.ones_ratio}\n${'='.repeat(50)}\n\n${data.bits}`;
                    const blob = new Blob([content], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `sigma_bitstream_${analysisId.substring(0, 8)}.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                  } catch (err) {
                    console.error('Export bits failed:', err);
                  }
                }}
              />
            </motion.div>

            {/* Hypothesis Explorer - Full Width */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.9 }}
              className="col-span-12 h-[500px]"
            >
              <HypothesisExplorer hypotheses={hypotheses} />
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
