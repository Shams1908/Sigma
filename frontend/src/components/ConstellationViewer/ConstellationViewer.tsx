import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { ConstellationPoint } from '../../types';
import './ConstellationViewer.css';

type ConstellationViewerProps = {
  constellation: ConstellationPoint[];
  className?: string;
  style?: CSSProperties;
};

const PLOT_PAD = 24;
const DOT_R = 2;

function maxAbsIQ(points: ConstellationPoint[]): number {
  let max = 0;
  for (const p of points) {
    const ai = Math.abs(p.i);
    const aq = Math.abs(p.q);
    if (ai > max) max = ai;
    if (aq > max) max = aq;
  }
  return max;
}

export function ConstellationViewer({
  constellation,
  className = '',
  style,
}: ConstellationViewerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setDimensions({
        width: Math.floor(width),
        height: Math.floor(height),
      });
    });

    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const side = Math.max(0, Math.min(dimensions.width, dimensions.height));

  const chart = useMemo(() => {
    if (side < 48) return null;

    const maxVal = maxAbsIQ(constellation);
    const extent = Math.max(maxVal * 1.2, 1.0);
    const mid = side / 2;
    const plotRadius = (side - PLOT_PAD * 2) / 2;

    const toX = (i: number) => mid + (i / extent) * plotRadius;
    const toY = (q: number) => mid - (q / extent) * plotRadius;

    const ox = mid;
    const oy = mid;

    const dots = constellation.map((p, index) => {
      const x = toX(p.i);
      const y = toY(p.q);
      return {
        key: `${index}-${p.i.toFixed(4)}-${p.q.toFixed(4)}`,
        x,
        y,
        dx: x - ox,
        dy: y - oy,
        delay: Math.min((index % 25) * 10, 200),
      };
    });

    return { ox, oy, dots, extent };
  }, [side, constellation]);

  const animKey = useMemo(() => {
    if (constellation.length === 0) return 'empty';
    return `${constellation.length}-${constellation[0].i.toFixed(3)}-${constellation[0].q.toFixed(3)}`;
  }, [constellation]);

  return (
    <section
      className={`relative flex min-h-0 flex-col border border-grid bg-panel ${className}`}
      style={style}
    >
      <header className="flex shrink-0 items-center justify-between px-3 pt-2.5 pb-1">
        <h2 className="font-sans text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">
          CONSTELLATION
        </h2>
        {constellation.length > 0 && (
          <span className="font-mono text-[9px] tracking-wider text-muted">
            {constellation.length} PTS
          </span>
        )}
      </header>

      <div
        ref={hostRef}
        className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden p-2"
      >
        {side >= 48 && (
          <svg
            key={animKey}
            width={side}
            height={side}
            viewBox={`0 0 ${side} ${side}`}
            className="block aspect-square select-none"
            role="img"
            aria-label="I/Q Constellation scatter plot"
          >
            {/* Minimal Quadrant Crosshair */}
            <line
              x1={PLOT_PAD}
              y1={chart?.oy ?? side / 2}
              x2={side - PLOT_PAD}
              y2={chart?.oy ?? side / 2}
              stroke="var(--color-border)"
              strokeWidth={1}
              shapeRendering="crispEdges"
            />
            <line
              x1={chart?.ox ?? side / 2}
              y1={PLOT_PAD}
              x2={chart?.ox ?? side / 2}
              y2={side - PLOT_PAD}
              stroke="var(--color-border)"
              strokeWidth={1}
              shapeRendering="crispEdges"
            />

            {/* Monospace I and Q Axis Labels */}
            <text
              x={side - PLOT_PAD + 5}
              y={(chart?.oy ?? side / 2) + 3}
              dominantBaseline="middle"
              textAnchor="start"
              fill="var(--color-text-muted)"
              className="select-none font-mono text-[10px]"
              fontFamily="var(--font-mono)"
              fontSize="10"
            >
              I
            </text>

            <text
              x={chart?.ox ?? side / 2}
              y={PLOT_PAD - 8}
              dominantBaseline="auto"
              textAnchor="middle"
              fill="var(--color-text-muted)"
              className="select-none font-mono text-[10px]"
              fontFamily="var(--font-mono)"
              fontSize="10"
            >
              Q
            </text>

            {/* Constellation Scatter Points */}
            {chart?.dots.map((dot) => (
              <circle
                key={dot.key}
                cx={dot.x}
                cy={dot.y}
                r={DOT_R}
                className="constellation-dot"
                stroke="none"
                style={
                  {
                    '--dx': `${dot.dx}px`,
                    '--dy': `${dot.dy}px`,
                    animationDelay: `${dot.delay}ms`,
                  } as CSSProperties
                }
              />
            ))}
          </svg>
        )}
      </div>
    </section>
  );
}

