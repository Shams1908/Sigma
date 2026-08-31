import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { ConstellationPoint } from '../../types';
import './ConstellationViewer.css';

type ConstellationViewerProps = {
  constellation: ConstellationPoint[];
  className?: string;
  style?: CSSProperties;
};

const PLOT_PAD = 22;
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
  const [side, setSide] = useState(0);

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const update = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      setSide(Math.max(0, Math.floor(Math.min(w, h))));
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const chart = useMemo(() => {
    if (side < 48 || constellation.length === 0) return null;

    const extent = Math.max(maxAbsIQ(constellation) * 1.25, 0.5);
    const mid = side / 2;
    const radius = (side - PLOT_PAD * 2) / 2;
    const toX = (i: number) => mid + (i / extent) * radius;
    const toY = (q: number) => mid - (q / extent) * radius;
    const ox = toX(0);
    const oy = toY(0);

    const dots = constellation.map((p, index) => {
      const x = toX(p.i);
      const y = toY(p.q);
      return {
        key: `${index}-${p.i}-${p.q}`,
        dx: x - ox,
        dy: y - oy,
        delay: (index % 25) * 12,
      };
    });

    return { ox, oy, dots, extent };
  }, [side, constellation]);

  const animKey =
    constellation.length === 0
      ? 'empty'
      : `${constellation.length}:${constellation[0].i}:${constellation[0].q}`;

  return (
    <section
      className={`relative min-h-0 border border-grid bg-panel ${className}`}
      style={style}
    >
      <h2 className="absolute left-3 top-2 z-10 font-mono text-[10px] font-medium uppercase tracking-[0.22em] text-muted">
        CONSTELLATION
      </h2>

      <div ref={hostRef} className="absolute inset-0 flex min-h-0 items-center justify-center pt-7">
        {chart ? (
          <svg
            key={animKey}
            width={side}
            height={side}
            viewBox={`0 0 ${side} ${side}`}
            className="block aspect-square"
            role="img"
            aria-label="I Q constellation scatter plot"
          >
            <line
              x1={PLOT_PAD}
              y1={chart.oy}
              x2={side - PLOT_PAD}
              y2={chart.oy}
              stroke="var(--color-border)"
              strokeWidth={1}
              shapeRendering="crispEdges"
            />
            <line
              x1={chart.ox}
              y1={PLOT_PAD}
              x2={chart.ox}
              y2={side - PLOT_PAD}
              stroke="var(--color-border)"
              strokeWidth={1}
              shapeRendering="crispEdges"
            />

            <text
              x={side - PLOT_PAD + 2}
              y={chart.oy}
              dominantBaseline="middle"
              fill="var(--color-text-muted)"
              fontFamily="var(--font-mono)"
              fontSize={10}
            >
              I
            </text>
            <text
              x={chart.ox}
              y={PLOT_PAD - 6}
              textAnchor="middle"
              fill="var(--color-text-muted)"
              fontFamily="var(--font-mono)"
              fontSize={10}
            >
              Q
            </text>

            {chart.dots.map((dot) => (
              <circle
                key={dot.key}
                className="constellation-dot"
                cx={chart.ox}
                cy={chart.oy}
                r={DOT_R}
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
        ) : null}
      </div>
    </section>
  );
}
