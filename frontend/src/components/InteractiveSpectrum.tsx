import { useEffect, useRef, useState } from 'react';

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
    const padding = 40;

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

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.5;
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

    ctx.fillStyle = '#888888';
    ctx.font = '10px monospace';
    ctx.fillText(`${minDb.toFixed(1)} dB`, 5, height - padding + 5);
    ctx.fillText(`${maxDb.toFixed(1)} dB`, 5, padding + 10);
    
  }, [data, zoom, pan]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(1, Math.min(10, prev + (e.deltaY > 0 ? -0.2 : 0.2))));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart(e.clientX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const delta = (e.clientX - dragStart) / (canvasRef.current?.width || 1);
    setPan(prev => Math.max(-0.5, Math.min(0.5, prev + delta)));
    setDragStart(e.clientX);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

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
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900 transition-colors duration-300">
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
      <div className="flex-1 relative">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-move"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>
      <div className="mt-2 text-xs text-gray-500 font-mono text-center">
        Scroll to zoom • Drag to pan
      </div>
    </div>
  );
}
