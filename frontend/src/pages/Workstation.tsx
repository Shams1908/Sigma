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
import ProcessingChain from '../components/ProcessingChain';
import RecentAnalysisSidebar from '../components/RecentAnalysisSidebar';
import InteractiveSpectrum from '../components/InteractiveSpectrum';
import SignalExplorer from '../components/SignalExplorer';
import DecoderLab from '../components/DecoderLab';
import BitstreamViewer from '../components/BitstreamViewer';
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
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);

  const sectionRefs = {
    upload: useRef<HTMLDivElement>(null),
    recent: useRef<HTMLDivElement>(null),
    spectrum: useRef<HTMLDivElement>(null),
    waterfall: useRef<HTMLDivElement>(null),
    constellation: useRef<HTMLDivElement>(null),
    waveform: useRef<HTMLDivElement>(null),
    parameters: useRef<HTMLDivElement>(null),
    processing: useRef<HTMLDivElement>(null),
    diagnostics: useRef<HTMLDivElement>(null),
    hypothesis: useRef<HTMLDivElement>(null),
    signalExplorer: useRef<HTMLDivElement>(null),
    decoderLab: useRef<HTMLDivElement>(null),
    bitstream: useRef<HTMLDivElement>(null)
  };

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
  const [recentAnalyses, setRecentAnalyses] = useState<Array<{
    id: string;
    fileName: string;
    timestamp: string;
    status: 'completed' | 'failed' | 'pending';
    snr?: number;
  }>>([]);
  const [processingStages, setProcessingStages] = useState<Array<{
    name: string;
    status: 'completed' | 'running' | 'pending' | 'failed';
    method?: string;
  }>>([
    { name: 'Ingestion', status: 'pending' },
    { name: 'DSP', status: 'pending', method: 'FFT/PSD' },
    { name: 'Modulation', status: 'pending', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'pending' },
    { name: 'Demodulation', status: 'pending' },
    { name: 'FEC', status: 'pending' }
  ]);
  const [signalRegions, setSignalRegions] = useState<Array<{
    id: string;
    frequencyStart: number;
    frequencyEnd: number;
    bandwidth: number;
    centerFreq: number;
    snr: number;
    candidateModulation: string;
    confidence: number;
  }>>([]);
  const [decoderCandidates, setDecoderCandidates] = useState<Array<{
    id: string;
    rank: number;
    modulation: string;
    symbolRate: number;
    fecType: string;
    score: number;
    stages: { sync: boolean; demod: boolean; fec: boolean };
    bitErrors?: number;
  }>>([]);
  const [bitstreamData, setBitstreamData] = useState<{
    bits: string;
    totalBits: number;
    entropy: number;
    onesRatio: number;
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
    
    // Clear all previous data to ensure fresh state
    setSpectrumData([]);
    setWaterfallData([]);
    setConstellationData([]);
    setWaveformData(null);
    setSignalParams(null);
    setHypotheses([]);
    setDiagnostics(null);
    setBitstreamData(null);
    setSignalRegions([]);
    setDecoderCandidates([]);
    setAnalysisId(null);
    
    setProcessingStages([
      { name: 'Ingestion', status: 'running' },
      { name: 'DSP', status: 'pending', method: 'FFT/PSD' },
      { name: 'Modulation', status: 'pending', method: 'ML Classifier' },
      { name: 'Synchronization', status: 'pending' },
      { name: 'Demodulation', status: 'pending' },
      { name: 'FEC', status: 'pending' }
    ]);
    
    try {
      console.log('Uploading file:', file.name);
      const uploadResponse = await api.uploadFile(file);
      console.log('Upload response:', uploadResponse);
      
      const signalId = uploadResponse.signal_id;
      const newAnalysis = {
        id: signalId,
        fileName: file.name,
        timestamp: new Date().toISOString(),
        status: 'pending' as const
      };
      setRecentAnalyses(prev => [newAnalysis, ...prev].slice(0, 10));

      setProcessingStages(prev => [
        { ...prev[0], status: 'completed' },
        { ...prev[1], status: 'running' },
        ...prev.slice(2)
      ]);

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
        
        if (params.bandwidth && params.bandwidth > 0) {
          const carrierFreq = params.carrierFrequency || 0;
          const region = {
            id: signalId,
            frequencyStart: carrierFreq - params.bandwidth / 2,
            frequencyEnd: carrierFreq + params.bandwidth / 2,
            bandwidth: params.bandwidth,
            centerFreq: carrierFreq,
            snr: params.snr,
            candidateModulation: 'Analyzing...',
            confidence: 0.5
          };
          console.log('Signal region created:', region);
          setSignalRegions([region]);
          console.log('setSignalRegions called with:', [region]);
        } else {
          console.warn('Cannot create signal region - missing or invalid bandwidth', params);
        }
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
      
      setProcessingStages(prev => [
        ...prev.slice(0, 2).map(s => ({ ...s, status: 'completed' as const })),
        { ...prev[2], status: 'running' },
        ...prev.slice(3)
      ]);
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
            
            const decoderCands = results.hypotheses.hypotheses.map((hyp: any, idx: number) => ({
              id: hyp.id,
              rank: idx + 1,
              modulation: hyp.modulation,
              symbolRate: hyp.symbolRate,
              fecType: hyp.fecType || 'None',
              score: hyp.mlConfidence,
              stages: {
                sync: hyp.validation?.syncPassed || false,
                demod: hyp.validation?.demodPassed || false,
                fec: hyp.validation?.fecPassed || false
              },
              bitErrors: hyp.bitErrors
            }));
            setDecoderCandidates(decoderCands);
            
            if (results.hypotheses.hypotheses.length > 0 && signalRegions.length > 0) {
              const topHyp = results.hypotheses.hypotheses[0];
              setSignalRegions(prev => prev.map(r => ({
                ...r,
                candidateModulation: topHyp.modulation,
                confidence: topHyp.mlConfidence
              })));
            }
          }
          
          try {
            console.log('Fetching diagnostics...');
            const diag = await api.getDiagnostics(analysisId);
            console.log('Diagnostics received:', diag);
            setDiagnostics(diag);
            
            setRecentAnalyses(prev => prev.map(a => 
              a.id === signalId 
                ? { ...a, status: 'completed' as const, snr: diag.snr } 
                : a
            ));
          } catch (diagErr) {
            console.error('Diagnostics fetch failed:', diagErr);
          }
          
          try {
            console.log('Fetching bitstream...');
            const bitstreamResp = await api.getBitstream(analysisId);
            if (bitstreamResp.available && bitstreamResp.bits) {
              console.log('Bitstream received:', bitstreamResp.total_bits, 'bits');
              setBitstreamData({
                bits: bitstreamResp.bits,
                totalBits: bitstreamResp.total_bits || 0,
                entropy: bitstreamResp.entropy || 0,
                onesRatio: bitstreamResp.ones_ratio || 0
              });
            }
          } catch (bitstreamErr) {
            console.error('Bitstream fetch failed:', bitstreamErr);
          }
          
          setProcessingStages(prev => prev.map(s => ({ ...s, status: 'completed' as const })));
        } else {
          console.warn('Analysis did not complete successfully:', finalStatus.status);
          setProcessingStages(prev => prev.map((s, i) => 
            i >= 2 ? { ...s, status: 'failed' as const } : s
          ));
          setRecentAnalyses(prev => prev.map(a => 
            a.id === signalId ? { ...a, status: 'failed' as const } : a
          ));
        }
      } catch (analysisErr) {
        console.warn('Full hypothesis analysis not available (ML model may be missing):', analysisErr);
        console.log('Showing DSP-only results without hypothesis validation');
        
        try {
          const params = await api.getSignalParameters(signalId);
          console.log('Creating fallback decoder candidate from params:', params);
          
          const fallbackCandidate = {
            id: 'dsp-only',
            rank: 1,
            modulation: 'DSP Analysis Only',
            symbolRate: params.symbolRate || 0,
            fecType: 'Unknown',
            score: 0.5,
            stages: {
              sync: false,
              demod: false,
              fec: false
            }
          };
          setDecoderCandidates([fallbackCandidate]);
          console.log('Fallback decoder candidate created:', fallbackCandidate);
        } catch (paramErr) {
          console.error('Could not create fallback candidate:', paramErr);
        }
        
        setProcessingStages(prev => [
          ...prev.slice(0, 2).map(s => ({ ...s, status: 'completed' as const })),
          ...prev.slice(2).map(s => ({ ...s, status: 'pending' as const }))
        ]);
      }

      setAnalysisProgress(100);
      setIsAnalyzing(false);
    } catch (err) {
      console.error('Upload error:', err);
      setIsAnalyzing(false);
      setProcessingStages(prev => prev.map(s => ({ ...s, status: 'failed' as const })));
      
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
      <div className="fixed inset-0 z-0">
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
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        {/* Header */}
        <motion.nav
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
          className="fixed top-6 left-0 right-0 z-50 px-6"
        >
          <motion.div 
            className="bg-[#0A0A0A] border border-[#222222] rounded-full px-8 py-4 shadow-2xl shadow-purple-900/20 max-w-[1600px] mx-auto"
            style={{
              boxShadow: '0 0 40px rgba(109, 40, 217, 0.15), 0 20px 40px rgba(0,0,0,0.5)'
            }}
          >
            <div className="flex items-center justify-between gap-8">
              <div className="flex items-center gap-8">
                <motion.div 
                  className="text-xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent"
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.2 }}
                >
                  WaveSight
                </motion.div>
                
                <div className="h-6 w-[1px] bg-[#222222]" />
                
                <div className="text-sm text-gray-400 font-mono">WORKSTATION</div>
              </div>
              
              <div className="flex items-center gap-6">
                {uploadedFile && (
                  <>
                    <div className="flex items-center gap-2 px-4 py-2 bg-[#111111] border border-[#222222] rounded-full">
                      <div className="w-2 h-2 bg-sigma-teal rounded-full animate-pulse" />
                      <span className="text-xs font-mono text-gray-400">{uploadedFile.name}</span>
                    </div>
                    <div className="h-6 w-[1px] bg-[#222222]" />
                  </>
                )}
                
                <motion.button
                  onClick={() => navigate('/')}
                  className="px-6 py-2 bg-[#111111] border border-[#222222] hover:border-sigma-purple text-sm font-semibold rounded-full transition-colors text-white"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Back to Home
                </motion.button>
              </div>
            </div>
          </motion.div>
        </motion.nav>

        {/* Main Workspace */}
        <div className="max-w-[1800px] mx-auto p-6 space-y-6 pt-28">
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
            {/* Recent Analysis Sidebar */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="col-span-12 lg:col-span-2 h-[400px]"
            >
              <RecentAnalysisSidebar 
                analyses={recentAnalyses}
                currentId={analysisId}
                onSelect={(id) => console.log('Select analysis:', id)}
              />
            </motion.div>

            {/* Interactive Spectrum Viewer with Zoom/Pan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="col-span-12 lg:col-span-6 h-[400px]"
            >
              <InteractiveSpectrum data={spectrumData} />
            </motion.div>

            {/* Data Readouts with Confidence */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="col-span-12 lg:col-span-4 h-[400px]"
            >
              {signalParams ? (
                <DataReadouts 
                  {...signalParams}
                  confidence={{
                    bandwidth: 0.78,
                    snr: 0.92,
                    symbolRate: 0.65
                  }}
                />
              ) : (
                <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="text-4xl">📊</div>
                    <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Processing Chain Visualization */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.35 }}
              className="col-span-12 h-[165px] mb-2"
            >
              <ProcessingChain stages={processingStages} />
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

            {/* Diagnostics Panel with Failure Reasons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
              className="col-span-12 lg:col-span-4 h-[350px]"
            >
              {diagnostics ? (
                <DiagnosticsPanel 
                  {...diagnostics}
                  failureReason={
                    !diagnostics.sync_locked ? 'Carrier synchronization failed - insufficient SNR' :
                    !diagnostics.demod_locked ? 'Demodulation unstable - timing recovery issues' :
                    !diagnostics.fec_valid ? 'FEC decoding failed - too many bit errors' :
                    undefined
                  }
                />
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
                    window.open(`http://localhost:8000/api/v1/results/${analysisId}/pdf`, '_blank');
                  } catch (err) {
                    console.error('Export PDF failed:', err);
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

            {/* Signal Explorer */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.0 }}
              className="col-span-12 lg:col-span-4 h-[600px]"
            >
              <SignalExplorer
                regions={signalRegions}
                onIsolate={(id) => {
                  const region = signalRegions.find(r => r.id === id);
                  if (region) {
                    alert(`Isolate Signal\n\nFrequency Range: ${(region.frequencyStart / 1e6).toFixed(3)} - ${(region.frequencyEnd / 1e6).toFixed(3)} MHz\nBandwidth: ${(region.bandwidth / 1e3).toFixed(1)} kHz\n\nThis would apply a bandpass filter to isolate this signal region.\n\n(Feature requires backend filtering endpoint)`);
                  }
                }}
                onAnalyze={(id) => {
                  const region = signalRegions.find(r => r.id === id);
                  if (region) {
                    alert(`Deep Analysis\n\nTarget: Region at ${(region.centerFreq / 1e6).toFixed(3)} MHz\nSNR: ${region.snr.toFixed(1)} dB\nCandidate: ${region.candidateModulation}\n\nThis would trigger focused hypothesis analysis on this signal region.\n\n(Feature requires backend focused analysis endpoint)`);
                  }
                }}
              />
            </motion.div>

            {/* Decoder Lab */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.1 }}
              className="col-span-12 lg:col-span-4 h-[600px]"
            >
              <DecoderLab
                candidates={decoderCandidates}
                searchProgress={analysisProgress}
                isSearching={isAnalyzing}
                onApplyBest={async () => {
                  if (decoderCandidates.length === 0) return;
                  
                  const best = decoderCandidates[0];
                  console.log('Applying best decoder:', best);
                  
                  if (best.id === 'dsp-only') {
                    alert(`Apply Decoder\n\nCannot apply DSP-only candidate.\nFull hypothesis analysis required for decoding.\n\nWaiting for ML model to generate valid decoder chains.`);
                    return;
                  }
                  
                  if (!analysisId) {
                    alert('No analysis ID available. Please complete analysis first.');
                    return;
                  }
                  
                  try {
                    console.log('Fetching bitstream with best decoder...');
                    const bitstreamResp = await api.getBitstream(analysisId);
                    if (bitstreamResp.available && bitstreamResp.bits) {
                      setBitstreamData({
                        bits: bitstreamResp.bits,
                        totalBits: bitstreamResp.total_bits || 0,
                        entropy: bitstreamResp.entropy || 0,
                        onesRatio: bitstreamResp.ones_ratio || 0
                      });
                      alert(`Decoder Applied!\n\nModulation: ${best.modulation}\nSymbol Rate: ${(best.symbolRate / 1000).toFixed(1)} ksps\nFEC: ${best.fecType}\n\nDecoded ${bitstreamResp.total_bits} bits successfully.\nCheck Bitstream Viewer for results.`);
                    } else {
                      alert(`Decoder Applied\n\nBut no bitstream available.\nReason: ${bitstreamResp.reason || 'Decoding failed'}`);
                    }
                  } catch (err) {
                    console.error('Apply decoder failed:', err);
                    alert(`Failed to apply decoder.\n\nError: ${err instanceof Error ? err.message : 'Unknown error'}`);
                  }
                }}
                onSelectCandidate={(id) => {
                  const candidate = decoderCandidates.find(c => c.id === id);
                  if (candidate) {
                    console.log('Selected decoder candidate:', candidate);
                  }
                }}
              />
            </motion.div>

            {/* Bitstream Viewer */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.2 }}
              className="col-span-12 lg:col-span-4 h-[600px]"
            >
              {bitstreamData ? (
                <BitstreamViewer
                  bits={bitstreamData.bits}
                  totalBits={bitstreamData.totalBits}
                  entropy={bitstreamData.entropy}
                  onesRatio={bitstreamData.onesRatio}
                />
              ) : (
                <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="text-4xl">⚡</div>
                    <p className="text-gray-400 text-sm">No Bitstream Available</p>
                    <p className="text-gray-600 text-xs">Complete analysis to view decoded bits</p>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
