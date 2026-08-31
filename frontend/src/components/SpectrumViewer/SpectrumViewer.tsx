import {
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react';
import type { SpectrumPoint } from '../../types';

type SpectrumViewerProps = {
  spectrum: SpectrumPoint[];
  className?: string;
  style?: CSSProperties;
};

const TRACE_MS = 400;
const PLOT_PAD = { top: 36, right: 16, bottom: 36, left: 52 };

type FreqUnit = {
  divisor: number;
  suffix: string;
};

function freqUnitFor(maxAbsHz: number): FreqUnit {
  if (maxAbsHz >= 1e6) return { divisor: 1e6, suffix: 'MHz' };
  if (maxAbsHz >= 1e3) return { divisor: 1e3, suffix: 'kHz' };
  return { divisor: 1, suffix: 'Hz' };
}

function formatFixed1(value: number): string {
  return value.toFixed(1);
}

function formatFrequency(hz: number, unit: FreqUnit): string {
  return `${formatFixed1(hz / unit.divisor)} ${unit.suffix}`;
}

function niceNum(range: number, round: boolean): number {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const exp = Math.floor(Math.log10(range));
  const f = range / 10 ** exp;
  let nf: number;
  if (round) {
    if (f < 1.5) nf = 1;
    else if (f < 3) nf = 2;
    else if (f < 7) nf = 5;
    else nf = 10;
  } else if (f <= 1) nf = 1;
  else if (f <= 2) nf = 2;
  else if (f <= 5) nf = 5;
  else nf = 10;
  return nf * 10 ** exp;
}

function ticks(min: number, max: number, maxTicks = 6): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) {
    return [min];
  }
  const range = niceNum(span, false);
  const step = niceNum(range / Math.max(maxTicks - 1, 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const values: number[] = [];
  const limit = niceMax + step * 0.5;
  for (let v = niceMin; v <= limit; v += step) {
    values.push(Number(v.toPrecision(12)));
  }
  return values;
}

function peakOf(spectrum: SpectrumPoint[]): SpectrumPoint | null {
  if (spectrum.length === 0) return null;
  let peak = spectrum[0];
  for (let i = 1; i < spectrum.length; i += 1) {
    if (spectrum[i].magnitudeDb > peak.magnitudeDb) {
      peak = spectrum[i];
    }
  }
  return peak;
}

export function SpectrumViewer({
  spectrum,
  className = '',
  style,
}: SpectrumViewerProps) {
  const clipId = `spectrum-clip-${useId().replace(/:/g, '')}`;
  const hostRef = useRef<HTMLDivElement>(null);
  const pathRef = useRef<SVGPathElement>(null);
  const animatedKey = useRef<string | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const update = () => {
      const rect = host.getBoundingClientRect();
      setSize({
        width: Math.max(0, Math.floor(host.clientWidth || rect.width)),
        height: Math.max(0, Math.floor(host.clientHeight || rect.height)),
      });
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const chart = useMemo(() => {
    const { width, height } = size;
    const innerW = width - PLOT_PAD.left - PLOT_PAD.right;
    const innerH = height - PLOT_PAD.top - PLOT_PAD.bottom;
    if (width < 32 || height < 32 || innerW < 8 || innerH < 8 || spectrum.length === 0) {
      return null;
    }

    let minF = spectrum[0].frequency;
    let maxF = spectrum[0].frequency;
    let minDb = spectrum[0].magnitudeDb;
    let maxDb = spectrum[0].magnitudeDb;
    for (const point of spectrum) {
      if (point.frequency < minF) minF = point.frequency;
      if (point.frequency > maxF) maxF = point.frequency;
      if (point.magnitudeDb < minDb) minDb = point.magnitudeDb;
      if (point.magnitudeDb > maxDb) maxDb = point.magnitudeDb;
    }

    if (maxF === minF) maxF = minF + 1;
    const dbPad = Math.max((maxDb - minDb) * 0.08, 1);
    let yMin = minDb - dbPad;
    let yMax = maxDb + dbPad;
    if (yMax === yMin) yMax = yMin + 1;

    const xTicks = ticks(minF, maxF, 6);
    const yTicks = ticks(yMin, yMax, 5);
    const xMin = Math.min(minF, xTicks[0]);
    const xMax = Math.max(maxF, xTicks[xTicks.length - 1]);
    yMin = Math.min(yMin, yTicks[0]);
    yMax = Math.max(yMax, yTicks[yTicks.length - 1]);

    const xScale = (f: number) =>
      PLOT_PAD.left + ((f - xMin) / (xMax - xMin)) * innerW;
    const yScale = (db: number) =>
      PLOT_PAD.top + ((yMax - db) / (yMax - yMin)) * innerH;

    const d = spectrum
      .map((point, i) => {
        const cmd = i === 0 ? 'M' : 'L';
        return `${cmd}${xScale(point.frequency).toFixed(2)},${yScale(point.magnitudeDb).toFixed(2)}`;
      })
      .join(' ');

    const maxAbsHz = Math.max(Math.abs(xMin), Math.abs(xMax));
    const unit = freqUnitFor(maxAbsHz);

    return { innerW, innerH, xTicks, yTicks, xScale, yScale, d, unit };
  }, [size, spectrum]);

  const peak = useMemo(() => peakOf(spectrum), [spectrum]);
  const peakUnit = freqUnitFor(Math.abs(peak?.frequency ?? 0));
  const spectrumKey = `${spectrum.length}:${spectrum[0]?.frequency}:${spectrum[spectrum.length - 1]?.frequency}:${spectrum[0]?.magnitudeDb}`;

  useLayoutEffect(() => {
    const path = pathRef.current;
    if (!path || !chart) return;

    const length = path.getTotalLength();
    if (length <= 0) return;

    if (animatedKey.current === spectrumKey) {
      path.style.transition = 'none';
      path.style.strokeDasharray = 'none';
      path.style.strokeDashoffset = '0';
      return;
    }

    animatedKey.current = spectrumKey;
    path.style.transition = 'none';
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;
    path.getBoundingClientRect();
    path.style.transition = `stroke-dashoffset ${TRACE_MS}ms linear`;
    path.style.strokeDashoffset = '0';
  }, [chart, spectrumKey]);

  return (
    <section
      className={`relative min-h-0 border border-grid bg-panel ${className}`}
      style={style}
    >
      <h2 className="absolute left-3 top-2 z-10 font-sans text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">
        SIGNAL SPECTRUM
      </h2>

      {peak ? (
        <div
          className="absolute right-3 top-2 z-10 text-right font-mono text-[10px] tabular-nums leading-4 text-muted"
          data-mono
        >
          <div>PK {formatFrequency(peak.frequency, peakUnit)}</div>
          <div>{formatFixed1(peak.magnitudeDb)} dB</div>
        </div>
      ) : null}

      <div ref={hostRef} className="absolute inset-0 min-h-0">
        {chart ? (
          <svg
            width={size.width}
            height={size.height}
            viewBox={`0 0 ${size.width} ${size.height}`}
            className="block overflow-visible"
            role="img"
            aria-label="Signal spectrum, frequency versus magnitude in decibels"
          >
            <clipPath id={clipId}>
              <rect
                x={PLOT_PAD.left}
                y={PLOT_PAD.top}
                width={chart.innerW}
                height={chart.innerH}
              />
            </clipPath>

            {chart.xTicks.map((tick) => {
              const x = chart.xScale(tick);
              return (
                <line
                  key={`vx-${tick}`}
                  x1={x}
                  y1={PLOT_PAD.top}
                  x2={x}
                  y2={PLOT_PAD.top + chart.innerH}
                  stroke="var(--color-border)"
                  strokeWidth={1}
                  shapeRendering="crispEdges"
                />
              );
            })}
            {chart.yTicks.map((tick) => {
              const y = chart.yScale(tick);
              return (
                <line
                  key={`hy-${tick}`}
                  x1={PLOT_PAD.left}
                  y1={y}
                  x2={PLOT_PAD.left + chart.innerW}
                  y2={y}
                  stroke="var(--color-border)"
                  strokeWidth={1}
                  shapeRendering="crispEdges"
                />
              );
            })}

            {chart.yTicks.map((tick) => (
              <text
                key={`yl-${tick}`}
                x={PLOT_PAD.left - 6}
                y={chart.yScale(tick)}
                textAnchor="end"
                dominantBaseline="middle"
                fill="var(--color-text-muted)"
                fontFamily="var(--font-mono)"
                fontSize={10}
              >
                {formatFixed1(tick)}
              </text>
            ))}
            {chart.xTicks.map((tick) => (
              <text
                key={`xl-${tick}`}
                x={chart.xScale(tick)}
                y={PLOT_PAD.top + chart.innerH + 14}
                textAnchor="middle"
                dominantBaseline="hanging"
                fill="var(--color-text-muted)"
                fontFamily="var(--font-mono)"
                fontSize={10}
              >
                {formatFixed1(tick / chart.unit.divisor)}
              </text>
            ))}

            <text
              x={PLOT_PAD.left + chart.innerW / 2}
              y={size.height - 4}
              textAnchor="middle"
              fill="var(--color-text-muted)"
              fontFamily="var(--font-sans)"
              fontSize={9}
              className="uppercase tracking-[0.18em]"
            >
              FREQUENCY ({chart.unit.suffix})
            </text>
            <text
              x={12}
              y={PLOT_PAD.top + chart.innerH / 2}
              textAnchor="middle"
              fill="var(--color-text-muted)"
              fontFamily="var(--font-sans)"
              fontSize={9}
              className="uppercase tracking-[0.18em]"
              transform={`rotate(-90 12 ${PLOT_PAD.top + chart.innerH / 2})`}
            >
              MAGNITUDE (dB)
            </text>

            <path
              ref={pathRef}
              d={chart.d}
              clipPath={`url(#${clipId})`}
              fill="none"
              stroke="var(--color-signal-cyan)"
              strokeWidth={1.5}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
        ) : null}
      </div>
    </section>
  );
}
