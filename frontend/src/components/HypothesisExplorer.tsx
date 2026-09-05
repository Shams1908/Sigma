interface ValidationStage {
  label: string;
  passed: boolean;
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

interface HypothesisExplorerProps {
  hypotheses: Hypothesis[];
}

export default function HypothesisExplorer({ hypotheses }: HypothesisExplorerProps) {
  const sortedHypotheses = [...hypotheses].sort((a, b) => {
    if (a.isWinner) return -1;
    if (b.isWinner) return 1;
    return b.mlConfidence - a.mlConfidence;
  });

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6 h-full overflow-auto">
      <div className="text-teal-400 font-mono text-xs tracking-wider mb-6">HYPOTHESIS VALIDATION CHAIN</div>
      
      <div className="space-y-4">
        {sortedHypotheses.map((hyp, idx) => {
          const stages: ValidationStage[] = [
            { label: 'ML Confidence', passed: hyp.mlConfidence > 0.5 },
            { label: 'Sync', passed: hyp.validation.syncPassed },
            { label: 'Demod', passed: hyp.validation.demodPassed },
            { label: 'Viterbi FEC', passed: hyp.validation.fecPassed }
          ];

          const allPassed = stages.every(s => s.passed);

          return (
            <div
              key={hyp.id}
              className={`border rounded-lg p-4 transition-all ${
                hyp.isWinner
                  ? 'border-teal-500 bg-teal-500/10'
                  : allPassed
                  ? 'border-green-500/50 bg-green-500/5'
                  : 'border-slate-700 bg-slate-800/30'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  {hyp.isWinner && (
                    <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse"></div>
                  )}
                  <div>
                    <div className="text-white font-mono text-sm font-bold">
                      {hyp.modulation}
                    </div>
                    <div className="text-slate-400 font-mono text-xs">
                      {hyp.symbolRate >= 1e3 ? `${(hyp.symbolRate / 1e3).toFixed(0)} ksps` : `${hyp.symbolRate} sps`}
                    </div>
                  </div>
                </div>
                {hyp.isWinner && (
                  <div className="bg-teal-400/20 text-teal-400 px-2 py-1 rounded text-xs font-mono font-bold">
                    PROVEN
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2 mb-3">
                {stages.map((stage, i) => (
                  <div key={i} className="flex items-center flex-1">
                    <div
                      className={`flex-1 h-8 rounded flex items-center justify-center text-xs font-mono font-bold transition-all ${
                        stage.passed
                          ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                          : 'bg-red-500/20 text-red-400 border border-red-500/30'
                      }`}
                    >
                      {stage.passed ? '✓' : '×'} {stage.label}
                    </div>
                    {i < stages.length - 1 && (
                      <div className="w-4 flex items-center justify-center">
                        <div className={`w-2 h-0.5 ${stage.passed ? 'bg-green-500/50' : 'bg-slate-600'}`}></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between text-xs">
                <div className="text-slate-500 font-mono">ML Confidence</div>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${hyp.mlConfidence * 100}%` }}
                    ></div>
                  </div>
                  <div className="text-slate-400 font-mono w-12 text-right">
                    {(hyp.mlConfidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {sortedHypotheses.length === 0 && (
        <div className="text-center text-slate-500 font-mono text-sm py-8">
          NO HYPOTHESES GENERATED
        </div>
      )}
    </div>
  );
}
