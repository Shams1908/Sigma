import { useMemo, type CSSProperties } from 'react';
import type { HypothesisCandidate } from '../../types';
import './HypothesisRanking.css';

type HypothesisRankingProps = {
  hypotheses?: HypothesisCandidate[];
  className?: string;
  style?: CSSProperties;
};

function getBarColorClass(confidence: number): string {
  if (confidence > 0.7) {
    return 'bg-signal-green';
  }
  if (confidence >= 0.2) {
    return 'bg-signal-amber';
  }
  return 'bg-signal-red';
}

function getConfidenceBadgeColor(confidence: number): string {
  if (confidence > 0.7) {
    return 'text-signal-green';
  }
  if (confidence >= 0.2) {
    return 'text-signal-amber';
  }
  return 'text-signal-red';
}

export function HypothesisRanking({
  hypotheses = [],
  className = '',
  style,
}: HypothesisRankingProps) {
  const sorted = useMemo(() => {
    return [...hypotheses].sort((a, b) => b.confidenceScore - a.confidenceScore);
  }, [hypotheses]);

  const animKey = useMemo(() => {
    if (hypotheses.length === 0) return 'empty';
    return hypotheses.map((h) => `${h.id}-${h.confidenceScore}`).join(':');
  }, [hypotheses]);

  return (
    <section
      className={`relative flex min-h-0 flex-col border border-grid bg-panel ${className}`}
      style={style}
    >
      <header className="flex shrink-0 items-center justify-between border-b border-grid px-3 py-2">
        <h2 className="font-sans text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">
          HYPOTHESIS ENGINE
        </h2>
        {sorted.length > 0 && (
          <span className="font-mono text-[9px] tracking-wider text-muted">
            {sorted.length} CANDIDATES
          </span>
        )}
      </header>

      <div
        key={animKey}
        className="flex min-h-0 flex-1 flex-col justify-center gap-3 overflow-y-auto p-3"
      >
        {sorted.length === 0 ? (
          <div className="flex h-full items-center justify-center font-sans text-[11px] text-muted">
            NO CANDIDATES
          </div>
        ) : (
          sorted.map((item, index) => {
            const percentage = item.confidenceScore * 100;
            const barColor = getBarColorClass(item.confidenceScore);
            const badgeColor = getConfidenceBadgeColor(item.confidenceScore);

            return (
              <div
                key={item.id}
                className="flex items-center gap-3"
                title={`${item.modulation}: ${percentage.toFixed(1)}% (${item.details})`}
              >
                {/* Modulation Name */}
                <div className="w-16 shrink-0 font-mono text-[11px] font-medium tracking-wider text-primary">
                  {item.modulation}
                </div>

                {/* Thin Horizontal Bar Track */}
                <div className="relative h-1.5 flex-1 overflow-hidden rounded-full border border-grid bg-background">
                  <div
                    className={`hypothesis-bar-fill h-full rounded-full ${barColor}`}
                    style={
                      {
                        '--target-width': `${Math.max(percentage, 0.5)}%`,
                        animationDelay: `${index * 50}ms`,
                      } as CSSProperties
                    }
                  />
                </div>

                {/* Percentage Number (fixed width, monospace) */}
                <div
                  className={`w-14 shrink-0 text-right font-mono text-[11px] font-medium tabular-nums ${
                    item.confidenceScore > 0.7 ? badgeColor : 'text-muted'
                  }`}
                >
                  {percentage.toFixed(1)}%
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
