import type { CSSProperties } from 'react';
import type { PipelineStage } from '../../types';
import './PipelineStatus.css';

type PipelineStatusProps = {
  stages?: PipelineStage[];
  className?: string;
  style?: CSSProperties;
};

export function PipelineStatus({
  stages = [],
  className = '',
  style,
}: PipelineStatusProps) {
  return (
    <section
      className={`relative flex min-h-0 flex-col border border-grid bg-panel ${className}`}
      style={style}
    >
      <header className="flex shrink-0 items-center justify-between border-b border-grid px-3 py-2">
        <h2 className="font-sans text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">
          PIPELINE STATUS
        </h2>
      </header>

      <div className="flex min-h-0 flex-1 flex-col justify-center px-4 py-3">
        {stages.length === 0 ? (
          <div className="font-sans text-[10px] uppercase tracking-wider text-muted">
            NO ACTIVE PIPELINE
          </div>
        ) : (
          <div className="flex flex-col">
            {stages.map((stage, index) => {
              const isLast = index === stages.length - 1;
              const isComplete = stage.status === 'complete';
              const isActive = stage.status === 'active';
              const isPending = stage.status === 'pending';

              return (
                <div
                  key={stage.label}
                  className="relative flex items-center gap-3 py-1.5"
                >
                  {/* Thin vertical connecting line to next stage */}
                  {!isLast && (
                    <div
                      className="absolute left-[4.5px] top-4 -bottom-1.5 w-px bg-grid"
                      aria-hidden="true"
                    />
                  )}

                  {/* Stage Node Indicator */}
                  <div className="relative z-10 flex h-2.5 w-2.5 shrink-0 items-center justify-center">
                    {isComplete && (
                      <div className="h-2 w-2 rounded-full bg-signal-green" />
                    )}
                    {isActive && (
                      <div className="pipeline-active-dot h-2 w-2 rounded-full bg-signal-cyan" />
                    )}
                    {isPending && (
                      <div className="h-2 w-2 rounded-full border border-grid bg-transparent" />
                    )}
                  </div>

                  {/* Stage Name */}
                  <div className="flex items-center gap-2 font-sans text-[10px] font-medium uppercase tracking-[0.14em]">
                    <span
                      className={
                        isActive
                          ? 'text-signal-cyan'
                          : isComplete
                          ? 'text-primary'
                          : 'text-muted'
                      }
                    >
                      {stage.label}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
