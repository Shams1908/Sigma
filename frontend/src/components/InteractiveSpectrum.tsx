import { useEffect, useRef, useState } from 'react';
import FocusMode from './FocusMode';

interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

interface InteractiveSpectrumProps {
  data: SpectrumPoint[];
}

export default function InteractiveSpectrum({ data }: InteractiveSpectrumProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!data.length || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = isFocused ? 80 : 40;

    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, width, height);

    const visibleData = data.slice(
      Math.floor((0.5 - 0.5 / zoom + pan) * data.length),
      Math.ceil((0.5 + 0.5 / zoom + pan) * data.length)
    );

    if (visibleData.length === 0) return;

    const minDb = Math.min(...visibleData.map(p => p.magnitudeDb));
    const maxDb = Math.max(...visibleData.map(p => p.magnitudeDb));

    const scaleX = (width - 2 * padding) / visibleData.length;
    const scaleY = (height - 2 * padding) / (maxDb - minDb || 1);

    if (isFocused) {
      const gridLines = 10;
      ctx.strokeStyle = '#1a1a1a';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= gridLines; i++) {
        const y = padding + (i / gridLines) * (height - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();

        const x = padding + (i / gridLines) * (width - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
      }

      ctx.fillStyle = '#666666';
      ctx.font = '11px monospace';
      for (let i = 0; i <= gridLines; i++) {
        const dbVal = maxDb - (i / gridLines) * (maxDb - minDb);
        ctx.fillText(`${dbVal.toFixed(1)}dB`, 5, padding + (i / gridLines) * (height - 2 * padding) + 4);
      }
    }

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = isFocused ? 2 : 1.5;
    ctx.beginPath();

    visibleData.forEach((point, i) => {
      const x = padding + i * scaleX;
      const y = height - padding - (point.magnitudeDb - minDb) * scaleY;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();

    ctx.strokeStyle = '#222222';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, width - 2 * padding, height - 2 * padding);

    if (!isFocused) {
      ctx.fillStyle = '#888888';
      ctx.font = '10px monospace';
      ctx.fillText(`${minDb.toFixed(1)} dB`, 5, height - padding + 5);
      ctx.fillText(`${maxDb.toFixed(1)} dB`, 5, padding + 10);
    }

    if (isFocused && cursor) {
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursor.x, padding);
      ctx.lineTo(cursor.x, height - padding);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding, cursor.y);
      ctx.lineTo(width - padding, cursor.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
  }, [data, zoom, pan, isFocused, cursor]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(1, Math.min(10, prev + (e.deltaY > 0 ? -0.2 : 0.2))));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart(e.clientX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isFocused) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        setCursor({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        });
      }
    }
    
    if (!isDragging) return;
    const delta = (e.clientX - dragStart) / (canvasRef.current?.width || 1);
    setPan(prev => Math.max(-0.5, Math.min(0.5, prev + delta)));
    setDragStart(e.clientX);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setCursor(null);
  };

  const getCursorValues = () => {
    if (!cursor || !canvasRef.current || !data.length) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    const padding = isFocused ? 80 : 40;

    const relX = (cursor.x - padding) / (rect.width - 2 * padding);
    
    const visibleData = data.slice(
      Math.floor((0.5 - 0.5 / zoom + pan) * data.length),
      Math.ceil((0.5 + 0.5 / zoom + pan) * data.length)
    );

    if (visibleData.length === 0) return null;

    const dataIndex = Math.floor(relX * visibleData.length);
    if (dataIndex < 0 || dataIndex >= visibleData.length) return null;

    const point = visibleData[dataIndex];
    return {
      freq: point.frequency,
      mag: point.magnitudeDb
    };
  };

  const cursorValues = getCursorValues();

  const renderSpectrum = () => (
    <div className="w-full h-full relative">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ cursor: isFocused ? (isDragging ? 'grabbing' : 'grab') : 'move' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {isFocused && cursor && cursorValues && (
        <div
          className="absolute bg-[#0A0A0A] border border-sigma-purple text-xs font-mono text-white px-3 py-2 rounded pointer-events-none z-10"
          style={{
            left: cursor.x + 10,
            top: cursor.y + 10
          }}
        >
          <div className="text-cyan-400">
            {cursorValues.freq >= 1e6 ? `${(cursorValues.freq / 1e6).toFixed(3)} MHz` : `${(cursorValues.freq / 1e3).toFixed(3)} kHz`}
          </div>
          <div className="text-purple-400">{cursorValues.mag.toFixed(2)} dB</div>
        </div>
      )}
      {isFocused && (
        <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Zoom: {zoom.toFixed(1)}x</div>
        </div>
      )}
    </div>
  );

  if (!data.length) {
    return (
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
        <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">SPECTRUM</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3">
            <div className="text-4xl">📡</div>
            <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900 transition-colors duration-300 relative">
        <div className="flex items-center justify-between mb-4">
          <div className="text-cyan-500 font-mono text-xs tracking-wider uppercase">SPECTRUM (INTERACTIVE)</div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setZoom(1); setPan(0); }}
              className="text-xs text-gray-400 hover:text-cyan-400 font-mono px-2 py-1 border border-[#222222] rounded hover:border-cyan-900"
            >
              RESET
            </button>
            <div className="text-xs text-gray-500 font-mono">
              {zoom.toFixed(1)}x
            </div>
          </div>
        </div>
        <button
          onClick={() => setIsFocused(true)}
          className="absolute top-6 right-6 z-10 p-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-cyan-500 rounded-lg transition-all"
          title="Expand to Focus Mode"
        >
          <svg className="w-4 h-4 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        <div className="flex-1 relative">
          {renderSpectrum()}
        </div>
        <div className="mt-2 text-xs text-gray-500 font-mono text-center">
          Scroll to zoom • Drag to pan
        </div>
      </div>
      <FocusMode
        isOpen={isFocused}
        onClose={() => {
          setIsFocused(false);
          setZoom(1);
          setPan(0);
          setCursor(null);
        }}
        title="SPECTRUM ANALYZER (INTERACTIVE)"
      >
        {renderSpectrum()}
        <div className="mt-4 text-xs text-gray-500 font-mono text-center">
          Scroll to zoom • Drag to pan
        </div>
      </FocusMode>
    </>
  );
}
