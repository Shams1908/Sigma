import { useEffect, useRef, useMemo, useState } from 'react';
import FocusMode from './FocusMode';

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
  const [isFocused, setIsFocused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const { width, height, xScale, yScale, path, ticks, peakInfo } = useMemo(() => {
    if (!data.length) return { width: 800, height: 400, xScale: (x: number) => x, yScale: (y: number) => y, path: '', ticks: { x: [], y: [] }, peakInfo: null };

    const w = isFocused ? 1600 : 800;
    const h = isFocused ? 800 : 400;
    const margin = { top: 20, right: 80, bottom: 60, left: 80 };
    const innerWidth = w - margin.left - margin.right;
    const innerHeight = h - margin.top - margin.bottom;

    const freqs = data.map(d => d.frequency);
    const mags = data.map(d => d.magnitudeDb);
    const minFreq = Math.min(...freqs);
    const maxFreq = Math.max(...freqs);
    const minMag = Math.min(...mags);
    const maxMag = Math.max(...mags);

    const freqRange = (maxFreq - minFreq) / zoomLevel;
    const magRange = (maxMag - minMag) / zoomLevel;
    const centerFreq = (maxFreq + minFreq) / 2 + panOffset.x * (maxFreq - minFreq);
    const centerMag = (maxMag + minMag) / 2 + panOffset.y * (maxMag - minMag);

    const viewMinFreq = centerFreq - freqRange / 2;
    const viewMaxFreq = centerFreq + freqRange / 2;
    const viewMinMag = centerMag - magRange / 2;
    const viewMaxMag = centerMag + magRange / 2;

    const xScale = (freq: number) => margin.left + ((freq - viewMinFreq) / (viewMaxFreq - viewMinFreq)) * innerWidth;
    const yScale = (mag: number) => margin.top + innerHeight - ((mag - viewMinMag) / (viewMaxMag - viewMinMag)) * innerHeight;

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

    const xRange = viewMaxFreq - viewMinFreq;
    const xTickSpacing = niceNum(xRange / (isFocused ? 10 : 5), true);
    const xTickValues: number[] = [];
    const xStart = Math.ceil(viewMinFreq / xTickSpacing) * xTickSpacing;
    for (let i = xStart; i <= viewMaxFreq; i += xTickSpacing) {
      xTickValues.push(i);
    }

    const yRange = viewMaxMag - viewMinMag;
    const yTickSpacing = niceNum(yRange / (isFocused ? 10 : 5), true);
    const yTickValues: number[] = [];
    const yStart = Math.ceil(viewMinMag / yTickSpacing) * yTickSpacing;
    for (let i = yStart; i <= viewMaxMag; i += yTickSpacing) {
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
  }, [data, isFocused, zoomLevel, panOffset]);

  const handleWheel = (e: React.WheelEvent) => {
    if (!isFocused) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel(prev => Math.max(1, Math.min(10, prev * delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isFocused) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isFocused) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCursor({ x, y });

    if (isDragging) {
      const dx = (e.clientX - dragStart.x) / width;
      const dy = (e.clientY - dragStart.y) / height;
      setPanOffset(prev => ({
        x: prev.x - dx * 2,
        y: prev.y + dy * 2
      }));
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setCursor(null);
    setIsDragging(false);
  };

  const getCursorValues = () => {
    if (!cursor || !data.length) return null;
    const freqs = data.map(d => d.frequency);
    const mags = data.map(d => d.magnitudeDb);
    const minFreq = Math.min(...freqs);
    const maxFreq = Math.max(...freqs);
    const minMag = Math.min(...mags);
    const maxMag = Math.max(...mags);

    const freqRange = (maxFreq - minFreq) / zoomLevel;
    const magRange = (maxMag - minMag) / zoomLevel;
    const centerFreq = (maxFreq + minFreq) / 2 + panOffset.x * (maxFreq - minFreq);
    const centerMag = (maxMag + minMag) / 2 + panOffset.y * (maxMag - minMag);

    const viewMinFreq = centerFreq - freqRange / 2;
    const viewMaxFreq = centerFreq + freqRange / 2;
    const viewMinMag = centerMag - magRange / 2;
    const viewMaxMag = centerMag + magRange / 2;

    const margin = { top: 20, right: 80, bottom: 60, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const relX = (cursor.x - margin.left) / innerWidth;
    const relY = 1 - (cursor.y - margin.top) / innerHeight;

    const freq = viewMinFreq + relX * (viewMaxFreq - viewMinFreq);
    const mag = viewMinMag + relY * (viewMaxMag - viewMinMag);

    return { freq, mag };
  };

  const cursorValues = getCursorValues();

  const renderSpectrum = () => (
    <div className="w-full h-full relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: isFocused ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
      >
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

        {isFocused && cursor && (
          <>
            <line
              x1={cursor.x}
              y1={20}
              x2={cursor.x}
              y2={height - 60}
              stroke="rgb(168, 85, 247)"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.6"
            />
            <line
              x1={80}
              y1={cursor.y}
              x2={width - 80}
              y2={cursor.y}
              stroke="rgb(168, 85, 247)"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.6"
            />
          </>
        )}
      </svg>
      {isFocused && cursor && cursorValues && (
        <div
          className="absolute bg-[#0A0A0A] border border-sigma-purple text-xs font-mono text-white px-3 py-2 rounded pointer-events-none"
          style={{
            left: cursor.x + 10,
            top: cursor.y + 10
          }}
        >
          <div className="text-sigma-teal">
            {cursorValues.freq >= 1e6 ? `${(cursorValues.freq / 1e6).toFixed(3)} MHz` : `${(cursorValues.freq / 1e3).toFixed(3)} kHz`}
          </div>
          <div className="text-sigma-purple">{cursorValues.mag.toFixed(2)} dB</div>
        </div>
      )}
      {isFocused && (
        <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded">
          <div className="text-xs font-mono text-slate-400">Zoom: {zoomLevel.toFixed(1)}x</div>
        </div>
      )}
    </div>
  );

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO SPECTRUM DATA</p>
      </div>
    );
  }

  return (
    <>
      <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal-900 transition-colors duration-300">
        <div className="absolute top-4 left-4 z-10">
          <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">SPECTRUM</div>
        </div>
        {peakInfo && (
          <div className="absolute top-4 right-16 z-10 bg-[#0A0A0A] border border-sigma-teal/30 px-3 py-1.5 rounded">
            <div className="text-xs font-mono text-slate-400 uppercase">PEAK</div>
            <div className="text-sm font-mono text-sigma-teal">
              {peakInfo.freq >= 1e6 ? `${(peakInfo.freq / 1e6).toFixed(2)} MHz` : `${(peakInfo.freq / 1e3).toFixed(2)} kHz`} / {peakInfo.mag.toFixed(1)} dB
            </div>
          </div>
        )}
        <button
          onClick={() => setIsFocused(true)}
          className="absolute top-4 right-4 z-10 p-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-sigma-teal rounded-lg transition-all"
          title="Expand to Focus Mode"
        >
          <svg className="w-4 h-4 text-sigma-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        {renderSpectrum()}
      </div>
      <FocusMode isOpen={isFocused} onClose={() => { setIsFocused(false); setZoomLevel(1); setPanOffset({ x: 0, y: 0 }); }} title="SPECTRUM ANALYZER">
        {renderSpectrum()}
      </FocusMode>
    </>
  );
}
