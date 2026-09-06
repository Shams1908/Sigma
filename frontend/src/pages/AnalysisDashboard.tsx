import type { CSSProperties } from 'react';
import { generateMockSignal } from '../services/mockData';
import type { AnalyzedSignal } from '../types';

type AnalysisDashboardProps = {
  signal: AnalyzedSignal | null;
  onLoadSignal: (signal: AnalyzedSignal) => void;
  onClearSignal: () => void;
};

function EmptyPanel({
  label,
  className = '',
  style,
}: {
  label: string;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <section
      className={`relative min-h-0 border border-grid bg-panel ${className}`}
      style={style}
    >
      <h2 className="absolute left-3 top-2 font-mono text-[10px] font-medium uppercase tracking-[0.22em] text-muted">
        {label}
      </h2>
    </section>
  );
}

function EmptyState({
  onLoadSignal,
}: {
  onLoadSignal: (signal: AnalyzedSignal) => void;
}) {
  return (
    <div className="relative h-full w-full bg-background">
      <div className="absolute left-6 top-6">
        <div className="font-mono text-sm tracking-[0.42em] text-primary">SIGMA</div>
        <p className="mt-1.5 font-sans text-[10px] tracking-[0.28em] text-muted">
          SIGNAL INTELLIGENCE WORKSTATION
        </p>
      </div>

      <div className="flex h-full items-center justify-center px-6">
        <div className="flex w-full max-w-xl flex-col items-center">
          <div className="flex w-full flex-col items-center border border-dashed border-grid bg-panel px-10 py-16">
            <p className="font-mono text-sm tracking-[0.28em] text-primary">
              DROP SIGNAL FILE
            </p>
            <p className="mt-3 font-mono text-[11px] tracking-[0.18em] text-muted">
              .IQ &nbsp; .WAV &nbsp; .RAW
            </p>
            <button
              type="button"
              className="mt-8 rounded border border-grid px-4 py-1.5 font-mono text-[11px] tracking-[0.16em] text-primary transition-colors hover:bg-primary hover:text-background"
            >
              Browse Files
            </button>
          </div>

          <div className="mt-8 flex w-full max-w-sm items-center gap-4">
            <div className="h-px flex-1 bg-grid" />
            <span className="font-mono text-[10px] tracking-[0.24em] text-muted">OR</span>
            <div className="h-px flex-1 bg-grid" />
          </div>

          <button
            type="button"
            onClick={() => onLoadSignal(generateMockSignal())}
            className="mt-6 font-mono text-[11px] tracking-[0.14em] text-muted underline decoration-grid underline-offset-4 hover:text-primary"
          >
            Load demonstration signal
          </button>
        </div>
      </div>
    </div>
  );
}

function LoadedState({
  signal,
  onClearSignal,
}: {
  signal: AnalyzedSignal;
  onClearSignal: () => void;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <header
        className="flex h-14 shrink-0 items-center justify-between border-b border-grid px-5"
        aria-label={`Loaded ${signal.metadata.fileName}`}
      >
        <button
          type="button"
          onClick={onClearSignal}
          className="font-mono text-sm tracking-[0.42em] text-primary"
        >
          SIGMA
        </button>
        <div className="flex items-center gap-2 font-mono text-[11px] tracking-[0.16em] text-signal-green">
          <span className="text-[8px] leading-none" aria-hidden>
            ●
          </span>
          <span>SYSTEM ONLINE</span>
        </div>
      </header>

      <div className="workstation-grid min-h-0 flex-1">
        <aside
          className="flex min-h-0 flex-col gap-2"
          style={{ gridArea: 'sidebar' }}
        >
          <EmptyPanel label="SIGNAL FILE" className="flex-1" />
          <EmptyPanel label="PARAMETERS" className="flex-1" />
        </aside>
        <EmptyPanel label="SIGNAL SPECTRUM" style={{ gridArea: 'spectrum' }} />
        <EmptyPanel label="WATERFALL" style={{ gridArea: 'waterfall' }} />
        <EmptyPanel
          label="HYPOTHESIS ENGINE"
          style={{ gridArea: 'hypothesis' }}
        />
        <EmptyPanel
          label="CONSTELLATION"
          style={{ gridArea: 'constellation' }}
        />
      </div>
    </div>
  );
}

export function AnalysisDashboard({
  signal,
  onLoadSignal,
  onClearSignal,
}: AnalysisDashboardProps) {
  if (!signal) {
    return <EmptyState onLoadSignal={onLoadSignal} />;
  }

  return <LoadedState signal={signal} onClearSignal={onClearSignal} />;
}
