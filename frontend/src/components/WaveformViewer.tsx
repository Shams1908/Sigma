import { useEffect, useRef } from 'react';

interface WaveformViewerProps {
  iData: number[];
  qData: number[];
  timeData: number[];
  sampleRate: number;
}

export default function WaveformViewer({ iData, qData, timeData, sampleRate }: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!iData.length || !qData.length || !canvasRef.current) return;

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
    const midY = height / 2;

    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, width, height);

    const iMax = Math.max(...iData.map(Math.abs));
    const qMax = Math.max(...qData.map(Math.abs));
    const ampMax = Math.max(iMax, qMax) || 1;

    const scaleY = (height * 0.4) / ampMax;

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1;
    ctx.beginPath();
    iData.forEach((val, idx) => {
      const x = (idx / iData.length) * width;
      const y = midY - val * scaleY;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 1;
    ctx.beginPath();
    qData.forEach((val, idx) => {
      const x = (idx / qData.length) * width;
      const y = midY + val * scaleY;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = '#222222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

  }, [iData, qData, timeData]);

  if (!iData.length) {
    return (
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
        <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">TIME DOMAIN</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3">
            <div className="text-4xl">〰️</div>
            <p className="text-gray-400 text-sm">Awaiting Signal Data</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900 transition-colors duration-300">
      <div className="flex items-center justify-between mb-4">
        <div className="text-cyan-500 font-mono text-xs tracking-wider uppercase">TIME DOMAIN</div>
        <div className="text-xs text-gray-500 font-mono">
          {(sampleRate / 1000).toFixed(1)} ksps
        </div>
      </div>
      <div className="flex-1 relative">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />
      </div>
      <div className="mt-2 flex gap-4 text-xs font-mono">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-cyan-500"></div>
          <span className="text-gray-400">I (In-Phase)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-purple-400"></div>
          <span className="text-gray-400">Q (Quadrature)</span>
        </div>
      </div>
    </div>
  );
}
