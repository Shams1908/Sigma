import { motion } from 'framer-motion';

interface DecoderCandidate {
  id: string;
  rank: number;
  modulation: string;
  symbolRate: number;
  fecType: string;
  score: number;
  stages: {
    sync: boolean;
    demod: boolean;
    fec: boolean;
  };
  bitErrors?: number;
}

interface DecoderLabProps {
  candidates: DecoderCandidate[];
  searchProgress: number;
  isSearching: boolean;
  onApplyBest: () => void;
  onSelectCandidate: (id: string) => void;
  selectedId?: string;
}

export default function DecoderLab({
  candidates,
  searchProgress,
  isSearching,
  onApplyBest,
  onSelectCandidate,
  selectedId
}: DecoderLabProps) {
  const getRankColor = (rank: number) => {
    if (rank === 1) return 'text-yellow-400';
    if (rank === 2) return 'text-gray-300';
    if (rank === 3) return 'text-orange-400';
    return 'text-gray-500';
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-cyan-500';
    if (score >= 0.4) return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  const StageIndicator = ({ passed }: { passed: boolean }) => (
    <div className={`w-2 h-2 rounded-full ${passed ? 'bg-green-400' : 'bg-gray-600'}`} />
  );

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">Decoder Lab</h3>
          <p className="text-xs text-gray-500 font-mono">Candidate Chains</p>
        </div>
        {candidates.length > 0 && (
          <button
            onClick={onApplyBest}
            disabled={isSearching}
            className="px-4 py-2 bg-sigma-teal hover:bg-sigma-teal/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-bold text-sm rounded-lg transition-all"
          >
            Apply Best
          </button>
        )}
      </div>

      {isSearching && (
        <div className="mb-4 p-4 bg-[#111111] border border-[#222222] rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400 font-mono">Search Progress</span>
            <span className="text-xs text-sigma-teal font-mono font-bold">{searchProgress}%</span>
          </div>
          <div className="w-full bg-[#1a1a1a] rounded-full h-2 overflow-hidden">
            <motion.div
              className="h-full bg-sigma-teal"
              initial={{ width: 0 }}
              animate={{ width: `${searchProgress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
        {candidates.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-3">
              <div className="text-4xl">🔬</div>
              <p className="text-gray-400 text-sm">
                {isSearching ? 'Searching decoder chains...' : 'No decoder candidates'}
              </p>
              {isSearching && (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="w-8 h-8 mx-auto border-4 border-sigma-teal/20 border-t-sigma-teal rounded-full"
                />
              )}
            </div>
          </div>
        ) : (
          candidates.map((candidate, index) => (
            <motion.div
              key={candidate.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              onClick={() => onSelectCandidate(candidate.id)}
              className={`
                bg-[#111111] border rounded-xl p-4 cursor-pointer transition-all
                ${selectedId === candidate.id
                  ? 'border-sigma-teal shadow-lg shadow-sigma-teal/20'
                  : 'border-[#222222] hover:border-sigma-teal/50'
                }
              `}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`text-2xl font-bold ${getRankColor(candidate.rank)}`}>
                    #{candidate.rank}
                  </div>
                  <div>
                    <div className="text-sm font-bold text-white">{candidate.modulation}</div>
                    <div className="text-xs text-gray-500 font-mono">
                      {(candidate.symbolRate / 1000).toFixed(1)} ksps
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-500 mb-1">Score</div>
                  <div className="text-lg font-bold text-white">{(candidate.score * 100).toFixed(0)}</div>
                </div>
              </div>

              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-500">Quality</span>
                  <span className="text-xs text-gray-400 font-mono">{(candidate.score * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-[#1a1a1a] rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full transition-all ${getScoreColor(candidate.score)}`}
                    style={{ width: `${candidate.score * 100}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                <div>
                  <div className="text-gray-500 mb-1">FEC Type</div>
                  <div className="text-white font-mono">{candidate.fecType}</div>
                </div>
                {candidate.bitErrors !== undefined && (
                  <div>
                    <div className="text-gray-500 mb-1">Bit Errors</div>
                    <div className="text-white font-mono">{candidate.bitErrors}</div>
                  </div>
                )}
              </div>

              <div className="pt-3 border-t border-[#222222]">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">Pipeline Status</span>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <StageIndicator passed={candidate.stages.sync} />
                      <span className="text-xs text-gray-600">Sync</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <StageIndicator passed={candidate.stages.demod} />
                      <span className="text-xs text-gray-600">Demod</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <StageIndicator passed={candidate.stages.fec} />
                      <span className="text-xs text-gray-600">FEC</span>
                    </div>
                  </div>
                </div>
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
