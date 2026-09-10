import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  BarChart3, 
  Waves, 
  Radio, 
  Activity, 
  Gauge, 
  Cpu, 
  Wrench, 
  Target, 
  Search, 
  FlaskConical, 
  Binary,
  UploadCloud,
  Clock,
  FileText,
  FileJson,
  FileCode
} from 'lucide-react';
import Topography from '../components/Topography';
import SpectrumViewer from '../components/SpectrumViewer';
import WaterfallViewer from '../components/WaterfallViewer';
import ConstellationViewer from '../components/ConstellationViewer';
import DataReadouts from '../components/DataReadouts';
import HypothesisExplorer from '../components/HypothesisExplorer';
import WaveformViewer from '../components/WaveformViewer';
import DiagnosticsPanel from '../components/DiagnosticsPanel';
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

const navigationItems = [
  { id: 'spectrum', icon: BarChart3, label: 'Spectrum' },
  { id: 'waterfall', icon: Waves, label: 'Waterfall' },
  { id: 'constellation', icon: Radio, label: 'Constellation' },
  { id: 'waveform', icon: Activity, label: 'Waveform' },
  { id: 'parameters', icon: Gauge, label: 'Parameters' },
  { id: 'processing', icon: Cpu, label: 'Processing' },
  { id: 'diagnostics', icon: Wrench, label: 'Diagnostics' },
  { id: 'hypothesis', icon: Target, label: 'Hypothesis' },
  { id: 'signalExplorer', icon: Search, label: 'Signal Explorer' },
  { id: 'decoderLab', icon: FlaskConical, label: 'Decoder Lab' },
  { id: 'bitstream', icon: Binary, label: 'Bitstream' }
];

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

  const scrollToSection = (sectionId: string) => {
    const ref = sectionRefs[sectionId as keyof typeof sectionRefs];
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

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

      <div className="relative z-10">
        <motion.nav
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
          className="fixed top-6 left-0 right-0 z-50 px-6"
        >
          <motion.div 
            className="bg-[#0A0A0A] border rounded-full px-8 py-4 shadow-2xl shadow-purple-900/20 max-w-[1600px] mx-auto relative"
            style={{
              borderImage: 'linear-gradient(90deg, rgba(109, 40, 217, 0.6), rgba(0, 217, 255, 0.6), rgba(109, 40, 217, 0.6)) 1',
              boxShadow: '0 0 40px rgba(109, 40, 217, 0.3), 0 0 80px rgba(0, 217, 255, 0.2), 0 20px 40px rgba(0,0,0,0.5)',
              border: '1px solid transparent',
              background: 'linear-gradient(#0A0A0A, #0A0A0A) padding-box, linear-gradient(90deg, rgba(109, 40, 217, 0.6), rgba(0, 217, 255, 0.6), rgba(109, 40, 217, 0.6)) border-box'
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
              
              <motion.button
                onClick={() => navigate('/')}
                className="px-6 py-2 bg-[#111111] border border-[#222222] hover:border-sigma-purple text-sm font-semibold rounded-full transition-colors text-white"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Back to Home
              </motion.button>
            </div>
          </motion.div>
        </motion.nav>

        <div className="flex pt-28">
          <div
            className="fixed left-0 top-28 bottom-0 z-40 bg-[#0A0A0A] border-r border-[#222222] overflow-hidden"
            style={{ 
              width: isSidebarExpanded ? '200px' : '80px',
              transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              willChange: 'width'
            }}
            onMouseEnter={() => setIsSidebarExpanded(true)}
            onMouseLeave={() => setIsSidebarExpanded(false)}
          >
            <div className="flex flex-col gap-2 p-4 w-[200px]">
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => scrollToSection(item.id)}
                    className="flex items-center gap-4 p-3 rounded-lg hover:bg-[#111111] border border-transparent hover:border-[#222222] transition-colors text-left"
                  >
                    <IconComponent className="w-5 h-5 text-sigma-teal flex-shrink-0" strokeWidth={1.5} />
                    <span
                      className="text-sm text-gray-400 font-mono whitespace-nowrap transition-opacity duration-300"
                      style={{ 
                        opacity: isSidebarExpanded ? 1 : 0,
                        pointerEvents: isSidebarExpanded ? 'auto' : 'none'
                      }}
                    >
                      {item.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div 
            style={{ 
              marginLeft: isSidebarExpanded ? '200px' : '80px',
              marginRight: '350px',
              width: `calc(100% - ${isSidebarExpanded ? '200px' : '80px'} - 350px)`,
              transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              willChange: 'margin-left, width'
            }}
          >
            <div className="p-6 space-y-8">
              <div ref={sectionRefs.spectrum} className="h-[500px]">
                <InteractiveSpectrum data={spectrumData} />
              </div>

              <div ref={sectionRefs.waterfall} className="h-[500px]">
                <WaterfallViewer data={waterfallData} isLive={true} />
              </div>

              <div ref={sectionRefs.constellation} className="h-[500px]">
                <ConstellationViewer data={constellationData} />
              </div>

              <div ref={sectionRefs.waveform} className="h-[500px]">
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
                      <Activity className="w-12 h-12 text-gray-600 mx-auto" strokeWidth={1.5} />
                      <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
                    </div>
                  </div>
                )}
              </div>

              <div ref={sectionRefs.parameters} className="h-[450px]">
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
                      <Gauge className="w-12 h-12 text-gray-600 mx-auto" strokeWidth={1.5} />
                      <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
                    </div>
                  </div>
                )}
              </div>

              <div ref={sectionRefs.processing} className="h-[180px]">
                <ProcessingChain stages={processingStages} />
              </div>

              <div ref={sectionRefs.diagnostics} className="h-[400px]">
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
                      <Wrench className="w-12 h-12 text-gray-600 mx-auto" strokeWidth={1.5} />
                      <p className="text-gray-400 text-sm">Analysis Pending</p>
                    </div>
                  </div>
                )}
              </div>

              <div ref={sectionRefs.hypothesis} className="h-[550px]">
                <HypothesisExplorer hypotheses={hypotheses} />
              </div>

              <div ref={sectionRefs.signalExplorer} className="h-[600px]">
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
              </div>

              <div ref={sectionRefs.decoderLab} className="h-[600px]">
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
              </div>

              <div ref={sectionRefs.bitstream} className="h-[600px]">
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
                      <Binary className="w-12 h-12 text-gray-600 mx-auto" strokeWidth={1.5} />
                      <p className="text-gray-400 text-sm">No Bitstream Available</p>
                      <p className="text-gray-600 text-xs">Complete analysis to view decoded bits</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <motion.div
            initial={{ x: 350, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
            className="fixed right-0 top-28 bottom-0 w-[350px] bg-[#0A0A0A] border-l border-[#222222] overflow-y-auto z-40"
          >
            <div className="p-6 space-y-6">
              <div ref={sectionRefs.upload}>
                <motion.div
                  onDragEnter={handleDragEnter}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`
                    border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300
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
                    <div className="space-y-3">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                        className="w-12 h-12 mx-auto border-4 border-sigma-teal/20 border-t-sigma-teal rounded-full"
                      />
                      <p className="text-sm font-semibold text-sigma-teal">Analyzing...</p>
                      <p className="text-xs text-gray-400 font-mono truncate">{uploadedFile?.name}</p>
                      <div className="w-full bg-[#111111] rounded-full h-1.5 overflow-hidden">
                        <motion.div
                          className="h-full bg-sigma-teal"
                          initial={{ width: 0 }}
                          animate={{ width: `${analysisProgress}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 font-mono">{analysisProgress}%</p>
                    </div>
                  ) : error ? (
                    <div className="space-y-3">
                      <div className="text-4xl">⚠️</div>
                      <div>
                        <p className="text-sm font-semibold text-red-400 mb-1">Upload Failed</p>
                        <p className="text-xs text-gray-400 mb-3">{error}</p>
                        <motion.button
                          onClick={() => {
                            setError(null);
                            setUploadedFile(null);
                          }}
                          className="px-4 py-2 bg-[#111111] border border-[#222222] hover:border-red-500 text-xs font-semibold rounded-lg transition-colors w-full"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          Try Again
                        </motion.button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <motion.div
                        animate={{ y: [0, -8, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="flex justify-center"
                      >
                        <UploadCloud className="w-12 h-12 text-sigma-teal" strokeWidth={1.5} />
                      </motion.div>
                      <div>
                        <p className="text-sm font-semibold text-white mb-1">
                          Drop Signal File
                        </p>
                        <p className="text-xs text-gray-400 mb-3">
                          .IQ or .WAV files
                        </p>
                        <motion.button
                          onClick={() => fileInputRef.current?.click()}
                          className="px-6 py-2 bg-sigma-teal hover:bg-sigma-teal/80 text-black text-xs font-bold rounded-lg transition-colors w-full"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          Browse Files
                        </motion.button>
                      </div>
                    </div>
                  )}
                </motion.div>
              </div>

              {uploadedFile && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-[#111111] border border-[#222222] rounded-xl p-4 space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Current File</div>
                    <div className="w-2 h-2 bg-sigma-teal rounded-full animate-pulse" />
                  </div>
                  <p className="text-sm font-mono text-white truncate">{uploadedFile.name}</p>
                  {signalParams && (
                    <div className="space-y-1.5 pt-2 border-t border-[#222222]">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-500">SNR</span>
                        <span className="text-xs font-mono text-sigma-teal">{signalParams.snr.toFixed(1)} dB</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-500">Bandwidth</span>
                        <span className="text-xs font-mono text-purple-400">{(signalParams.bandwidth / 1000).toFixed(1)} kHz</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-500">Sample Rate</span>
                        <span className="text-xs font-mono text-gray-300">{(signalParams.sampleRate / 1000).toFixed(0)} ksps</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {uploadedFile && analysisId && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="space-y-3"
                >
                  <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Export Results</div>
                  <div className="space-y-2">
                    <button
                      onClick={async () => {
                        try {
                          window.open(`http://localhost:8000/api/v1/results/${analysisId}/pdf`, '_blank');
                        } catch (err) {
                          console.error('Export PDF failed:', err);
                        }
                      }}
                      className="w-full px-4 py-3 bg-[#111111] border border-[#222222] hover:border-red-500 text-xs font-semibold rounded-lg transition-colors text-white flex items-center justify-center gap-2"
                    >
                      <FileText className="w-4 h-4" strokeWidth={1.5} />
                      Export as PDF
                    </button>
                    <button
                      onClick={async () => {
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
                      className="w-full px-4 py-3 bg-[#111111] border border-[#222222] hover:border-blue-500 text-xs font-semibold rounded-lg transition-colors text-white flex items-center justify-center gap-2"
                    >
                      <FileJson className="w-4 h-4" strokeWidth={1.5} />
                      Export as JSON
                    </button>
                    <button
                      onClick={async () => {
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
                      className="w-full px-4 py-3 bg-[#111111] border border-[#222222] hover:border-cyan-500 text-xs font-semibold rounded-lg transition-colors text-white flex items-center justify-center gap-2"
                    >
                      <FileCode className="w-4 h-4" strokeWidth={1.5} />
                      Export Bitstream
                    </button>
                  </div>
                </motion.div>
              )}

              <div ref={sectionRefs.recent}>
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Recent Analyses</div>
                  {recentAnalyses.length > 0 ? (
                    <div className="space-y-2">
                      {recentAnalyses.slice(0, 5).map((analysis) => (
                        <motion.button
                          key={analysis.id}
                          onClick={() => console.log('Load analysis:', analysis.id)}
                          className={`
                            w-full p-3 rounded-lg border transition-all text-left
                            ${analysis.id === analysisId 
                              ? 'bg-sigma-teal/10 border-sigma-teal' 
                              : 'bg-[#111111] border-[#222222] hover:border-sigma-teal/50'
                            }
                          `}
                          whileHover={{ x: 2 }}
                        >
                          <div className="flex items-start justify-between gap-2 mb-1.5">
                            <p className="text-xs font-mono text-white truncate flex-1">{analysis.fileName}</p>
                            <div className={`
                              w-1.5 h-1.5 rounded-full mt-1
                              ${analysis.status === 'completed' ? 'bg-green-500' : 
                                analysis.status === 'failed' ? 'bg-red-500' : 
                                'bg-yellow-500 animate-pulse'}
                            `} />
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">{new Date(analysis.timestamp).toLocaleTimeString()}</span>
                            {analysis.snr !== undefined && (
                              <span className="text-xs font-mono text-sigma-teal">{analysis.snr.toFixed(1)} dB</span>
                            )}
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-[#111111] border border-[#222222] rounded-lg p-6 text-center">
                      <Clock className="w-10 h-10 text-gray-600 mx-auto mb-2" strokeWidth={1.5} />
                      <p className="text-xs text-gray-500">No recent analyses</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
