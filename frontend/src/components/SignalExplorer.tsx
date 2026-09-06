import { motion } from 'framer-motion';

interface SignalRegion {
  id: string;
  frequencyStart: number;
  frequencyEnd: number;
  bandwidth: number;
  centerFreq: number;
  snr: number;
  candidateModulation: string;
  confidence: number;
}

interface SignalExplorerProps {
  regions: SignalRegion[];
  onIsolate: (id: string) => void;
  onAnalyze: (id: string) => void;
  selectedId?: string;
}

export default function SignalExplorer({ regions, onIsolate, onAnalyze, selectedId }: SignalExplorerProps) {
  const formatFreq = (hz: number) => {
    if (hz >= 1e9) return `${(hz / 1e9).toFixed(3)} GHz`;
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
    if (hz >= 1e3) return `${(hz / 1e3).toFixed(3)} kHz`;
    return `${hz.toFixed(0)} Hz`;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-cyan-400';
    if (confidence >= 0.4) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">Signal Explorer</h3>
          <p className="text-xs text-gray-500 font-mono">Detected Regions</p>
        </div>
        <div className="px-3 py-1 bg-[#111111] border border-[#222222] rounded-lg">
          <span className="text-xs font-mono text-sigma-teal">{regions.length} Found</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
        {regions.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-3">
              <div className="text-4xl">🔍</div>
              <p className="text-gray-400 text-sm">No signals detected</p>
              <p className="text-gray-600 text-xs">Upload a file to begin analysis</p>
            </div>
          </div>
        ) : (
          regions.map((region, index) => (
            <motion.div
              key={region.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className={`
                bg-[#111111] border rounded-xl p-4 transition-all cursor-pointer
                ${selectedId === region.id 
                  ? 'border-sigma-teal shadow-lg shadow-sigma-teal/20' 
                  : 'border-[#222222] hover:border-sigma-teal/50'
                }
              `}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-sm font-bold text-white mb-1">
                    Region {index + 1}
                  </div>
                  <div className="text-xs text-gray-500 font-mono">
                    {formatFreq(region.centerFreq)}
                  </div>
                </div>
                <div className={`text-xs font-mono font-bold ${getConfidenceColor(region.confidence)}`}>
                  {(region.confidence * 100).toFixed(0)}%
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                <div>
                  <div className="text-gray-500 mb-1">Bandwidth</div>
                  <div className="text-white font-mono">{formatFreq(region.bandwidth)}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">SNR</div>
                  <div className="text-white font-mono">{region.snr.toFixed(1)} dB</div>
                </div>
              </div>

              <div className="mb-3 pb-3 border-b border-[#222222]">
                <div className="text-xs text-gray-500 mb-1">Candidate Modulation</div>
                <div className="text-sm text-sigma-teal font-semibold">{region.candidateModulation}</div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => onIsolate(region.id)}
                  className="flex-1 px-3 py-2 bg-[#1a1a1a] hover:bg-[#222222] border border-[#333333] hover:border-sigma-teal/50 text-xs font-semibold text-white rounded-lg transition-all"
                >
                  Isolate
                </button>
                <button
                  onClick={() => onAnalyze(region.id)}
                  className="flex-1 px-3 py-2 bg-sigma-teal hover:bg-sigma-teal/80 text-xs font-bold text-black rounded-lg transition-all"
                >
                  Analyze
                </button>
              </div>
            </motion.div>
          ))
        )}
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #111111;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #333333;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #0dd9c5;
        }
      `}</style>
    </div>
  );
}
