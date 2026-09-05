import { useEffect, useRef, useMemo } from 'react';

interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

interface SpectrumViewerProps {
  data: SpectrumPoint[];
}

export default function SpectrumViewer({ data }: SpectrumViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { width, height, xScale, yScale, path, ticks, peakInfo } = useMemo(() => {
    if (!data.length) return { width: 800, height: 400, xScale: (x: number) => x, yScale: (y: number) => y, path: '', ticks: { x: [], y: [] }, peakInfo: null };

    const w = 800;
    const h = 400;
    const margin = { top: 20, right: 80, bottom: 60, left: 80 };
    const innerWidth = w - margin.left - margin.right;
    const innerHeight = h - margin.top - margin.bottom;

    const freqs = data.map(d => d.frequency);
    const mags = data.map(d => d.magnitudeDb);
    const minFreq = Math.min(...freqs);
    const maxFreq = Math.max(...freqs);
    const minMag = Math.min(...mags);
    const maxMag = Math.max(...mags);

    const xScale = (freq: number) => margin.left + ((freq - minFreq) / (maxFreq - minFreq)) * innerWidth;
    const yScale = (mag: number) => margin.top + innerHeight - ((mag - minMag) / (maxMag - minMag)) * innerHeight;

    const pathData = data.map((d, i) => {
      const x = xScale(d.frequency);
      const y = yScale(d.magnitudeDb);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    }).join(' ');

    const niceNum = (range: number, round: boolean): number => {
      const exponent = Math.floor(Math.log10(range));
      const fraction = range / Math.pow(10, exponent);
      let niceFraction: number;
      if (round) {
        if (fraction < 1.5) niceFraction = 1;
        else if (fraction < 3) niceFraction = 2;
        else if (fraction < 7) niceFraction = 5;
        else niceFraction = 10;
      } else {
        if (fraction <= 1) niceFraction = 1;
        else if (fraction <= 2) niceFraction = 2;
        else if (fraction <= 5) niceFraction = 5;
        else niceFraction = 10;
      }
      return niceFraction * Math.pow(10, exponent);
    };

    const xRange = maxFreq - minFreq;
    const xTickSpacing = niceNum(xRange / 5, true);
    const xTickValues: number[] = [];
    const xStart = Math.ceil(minFreq / xTickSpacing) * xTickSpacing;
    for (let i = xStart; i <= maxFreq; i += xTickSpacing) {
      xTickValues.push(i);
    }

    const yRange = maxMag - minMag;
    const yTickSpacing = niceNum(yRange / 5, true);
    const yTickValues: number[] = [];
    const yStart = Math.ceil(minMag / yTickSpacing) * yTickSpacing;
    for (let i = yStart; i <= maxMag; i += yTickSpacing) {
      yTickValues.push(i);
    }

    const formatFreq = (f: number): string => {
      if (Math.abs(f) >= 1e6) return `${(f / 1e6).toFixed(1)}M`;
      if (Math.abs(f) >= 1e3) return `${(f / 1e3).toFixed(1)}k`;
      return `${f.toFixed(0)}`;
    };

    const xTicks = xTickValues.map(val => ({
      value: val,
      position: xScale(val),
      label: formatFreq(val)
    }));

    const yTicks = yTickValues.map(val => ({
      value: val,
      position: yScale(val),
      label: val.toFixed(1)
    }));

    const peakIndex = mags.indexOf(Math.max(...mags));
    const peakFreq = freqs[peakIndex];
    const peakMag = mags[peakIndex];

    return {
      width: w,
      height: h,
      xScale,
      yScale,
      path: pathData,
      ticks: { x: xTicks, y: yTicks },
      peakInfo: { freq: peakFreq, mag: peakMag, x: xScale(peakFreq), y: yScale(peakMag) }
    };
  }, [data]);

  useEffect(() => {
    const path = svgRef.current?.querySelector('.spectrum-trace') as SVGPathElement;
    if (path) {
      const length = path.getTotalLength();
      path.style.strokeDasharray = `${length}`;
      path.style.strokeDashoffset = `${length}`;
      requestAnimationFrame(() => {
        path.style.transition = 'stroke-dashoffset 0.4s linear';
        path.style.strokeDashoffset = '0';
      });
    }
  }, [path]);

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-slate-900/50 rounded-xl border border-slate-700">
        <p className="text-slate-500 font-mono text-sm">NO SPECTRUM DATA</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full bg-slate-900/50 rounded-xl border border-slate-700 p-4 relative">
      <div className="absolute top-4 left-4 z-10">
        <div className="text-teal-400 font-mono text-xs tracking-wider">SPECTRUM</div>
      </div>
      {peakInfo && (
        <div className="absolute top-4 right-4 z-10 bg-slate-800/80 backdrop-blur-sm px-3 py-1.5 rounded border border-teal-500/30">
          <div className="text-xs font-mono text-slate-400">PEAK</div>
          <div className="text-sm font-mono text-teal-400">
            {peakInfo.freq >= 1e6 ? `${(peakInfo.freq / 1e6).toFixed(2)} MHz` : `${(peakInfo.freq / 1e3).toFixed(2)} kHz`} / {peakInfo.mag.toFixed(1)} dB
          </div>
        </div>
      )}
      <svg ref={svgRef} width={width} height={height} className="w-full h-auto">
        <defs>
          <linearGradient id="spectrumGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgb(20, 184, 166)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(20, 184, 166)" stopOpacity="0" />
          </linearGradient>
        </defs>

        <g>
          {ticks.y.map((tick, i) => (
            <g key={i}>
              <line
                x1={80}
                y1={tick.position}
                x2={width - 80}
                y2={tick.position}
                stroke="rgb(51, 65, 85)"
                strokeWidth="0.5"
                strokeDasharray="2,2"
              />
              <text
                x={70}
                y={tick.position}
                textAnchor="end"
                alignmentBaseline="middle"
                className="text-xs font-mono fill-slate-400"
              >
                {tick.label}
              </text>
            </g>
          ))}
        </g>

        <g>
          {ticks.x.map((tick, i) => (
            <g key={i}>
              <line
                x1={tick.position}
                y1={20}
                x2={tick.position}
                y2={height - 60}
                stroke="rgb(51, 65, 85)"
                strokeWidth="0.5"
                strokeDasharray="2,2"
              />
              <text
                x={tick.position}
                y={height - 40}
                textAnchor="middle"
                className="text-xs font-mono fill-slate-400"
              >
                {tick.label}
              </text>
            </g>
          ))}
        </g>

        <text
          x={width / 2}
          y={height - 10}
          textAnchor="middle"
          className="text-xs font-mono fill-slate-400"
        >
          FREQUENCY (Hz)
        </text>

        <text
          x={20}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90, 20, ${height / 2})`}
          className="text-xs font-mono fill-slate-400"
        >
          MAGNITUDE (dB)
        </text>

        <path
          className="spectrum-trace"
          d={path}
          fill="none"
          stroke="rgb(20, 184, 166)"
          strokeWidth="2"
        />

        {peakInfo && (
          <circle
            cx={peakInfo.x}
            cy={peakInfo.y}
            r="4"
            fill="rgb(20, 184, 166)"
            stroke="rgb(20, 184, 166)"
            strokeWidth="2"
            opacity="0.8"
          >
            <animate
              attributeName="r"
              values="4;6;4"
              dur="2s"
              repeatCount="indefinite"
            />
          </circle>
        )}
      </svg>
    </div>
  );
}
