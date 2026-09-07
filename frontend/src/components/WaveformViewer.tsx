import { useEffect, useRef, useState } from 'react';
import FocusMode from './FocusMode';

interface WaveformViewerProps {
  iData: number[];
  qData: number[];
  timeData: number[];
  sampleRate: number;
}

export default function WaveformViewer({ iData, qData, timeData, sampleRate }: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);
  const [selectedRange, setSelectedRange] = useState<{ start: number; end: number } | null>(null);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);

  const drawWaveform = (canvas: HTMLCanvasElement, focused: boolean) => {
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

    const samplesPerView = Math.floor(iData.length / zoomLevel);
    const centerSample = Math.floor(iData.length / 2 + panOffset * iData.length);
    const startSample = Math.max(0, centerSample - samplesPerView / 2);
    const endSample = Math.min(iData.length, startSample + samplesPerView);

    const viewIData = iData.slice(startSample, endSample);
    const viewQData = qData.slice(startSample, endSample);
    const viewTimeData = timeData.slice(startSample, endSample);

    const iMax = Math.max(...viewIData.map(Math.abs));
    const qMax = Math.max(...viewQData.map(Math.abs));
    const ampMax = Math.max(iMax, qMax) || 1;

    const scaleY = (height * 0.4) / ampMax;

    if (focused) {
      const gridLines = 10;
      ctx.strokeStyle = '#1a1a1a';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= gridLines; i++) {
        const y = (i / gridLines) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        const x = (i / gridLines) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      ctx.fillStyle = '#666666';
      ctx.font = '10px monospace';
      for (let i = 0; i <= gridLines; i++) {
        const amp = ampMax - (i / gridLines) * 2 * ampMax;
        ctx.fillText(amp.toFixed(3), 5, (i / gridLines) * height + 12);
      }

      for (let i = 0; i <= gridLines; i++) {
        const timeIdx = Math.floor((i / gridLines) * viewTimeData.length);
        const time = viewTimeData[timeIdx] || 0;
        ctx.fillText(`${(time * 1000).toFixed(2)}ms`, (i / gridLines) * width - 20, height - 5);
      }
    }

    if (selectedRange) {
      const startX = ((selectedRange.start - startSample) / (endSample - startSample)) * width;
      const endX = ((selectedRange.end - startSample) / (endSample - startSample)) * width;
      ctx.fillStyle = 'rgba(20, 184, 166, 0.1)';
      ctx.fillRect(startX, 0, endX - startX, height);
      ctx.strokeStyle = 'rgba(20, 184, 166, 0.5)';
      ctx.lineWidth = 2;
      ctx.strokeRect(startX, 0, endX - startX, height);
    }

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = focused ? 2 : 1;
    ctx.beginPath();
    viewIData.forEach((val, idx) => {
      const x = (idx / viewIData.length) * width;
      const y = midY - val * scaleY;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = focused ? 2 : 1;
    ctx.beginPath();
    viewQData.forEach((val, idx) => {
      const x = (idx / viewQData.length) * width;
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

    if (focused && cursor) {
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursor.x, 0);
      ctx.lineTo(cursor.x, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, cursor.y);
      ctx.lineTo(width, cursor.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    if (!iData.length || !qData.length || !canvasRef.current) return;
    drawWaveform(canvasRef.current, isFocused);
  }, [iData, qData, timeData, isFocused, zoomLevel, panOffset, cursor, selectedRange]);

  const handleWheel = (e: React.WheelEvent) => {
    if (!isFocused) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel(prev => Math.max(1, Math.min(20, prev * delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isFocused) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const relX = x / rect.width;
    
    if (e.shiftKey) {
      const samplesPerView = Math.floor(iData.length / zoomLevel);
      const centerSample = Math.floor(iData.length / 2 + panOffset * iData.length);
      const startSample = Math.max(0, centerSample - samplesPerView / 2);
      const sampleIdx = Math.floor(startSample + relX * samplesPerView);
      setSelectionStart(sampleIdx);
      setSelectedRange(null);
    } else {
      setIsDragging(true);
      setDragStart(e.clientX);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isFocused) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCursor({ x, y });

    if (selectionStart !== null && e.shiftKey) {
      const relX = x / rect.width;
      const samplesPerView = Math.floor(iData.length / zoomLevel);
      const centerSample = Math.floor(iData.length / 2 + panOffset * iData.length);
      const startSample = Math.max(0, centerSample - samplesPerView / 2);
      const sampleIdx = Math.floor(startSample + relX * samplesPerView);
      setSelectedRange({
        start: Math.min(selectionStart, sampleIdx),
        end: Math.max(selectionStart, sampleIdx)
      });
    } else if (isDragging) {
      const dx = (e.clientX - dragStart) / rect.width;
      setPanOffset(prev => Math.max(-0.5, Math.min(0.5, prev - dx * 2)));
      setDragStart(e.clientX);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setSelectionStart(null);
  };

  const handleMouseLeave = () => {
    setCursor(null);
    setIsDragging(false);
    setSelectionStart(null);
  };

  const getCursorValues = () => {
    if (!cursor || !iData.length) return null;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return null;

    const relX = cursor.x / rect.width;
    const relY = (rect.height / 2 - cursor.y) / (rect.height * 0.4);

    const samplesPerView = Math.floor(iData.length / zoomLevel);
    const centerSample = Math.floor(iData.length / 2 + panOffset * iData.length);
    const startSample = Math.max(0, centerSample - samplesPerView / 2);
    const sampleIdx = Math.floor(startSample + relX * samplesPerView);

    if (sampleIdx < 0 || sampleIdx >= iData.length) return null;

    const time = timeData[sampleIdx];
    const iVal = iData[sampleIdx];
    const qVal = qData[sampleIdx];

    return { time, iVal, qVal };
  };

  const cursorValues = getCursorValues();

  const renderWaveform = () => (
    <div className="w-full h-full relative">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: isFocused ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
      />
      {isFocused && cursor && cursorValues && (
        <div
          className="absolute bg-[#0A0A0A] border border-sigma-purple text-xs font-mono text-white px-3 py-2 rounded pointer-events-none z-10"
          style={{
            left: cursor.x + 10,
            top: cursor.y + 10
          }}
        >
          <div className="text-cyan-400">I: {cursorValues.iVal.toFixed(4)}</div>
          <div className="text-purple-400">Q: {cursorValues.qVal.toFixed(4)}</div>
          <div className="text-gray-400">{(cursorValues.time * 1000).toFixed(3)} ms</div>
        </div>
      )}
      {isFocused && (
        <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Zoom: {zoomLevel.toFixed(1)}x</div>
          {selectedRange && (
            <div className="text-xs font-mono text-sigma-teal mt-1">
              Selected: {selectedRange.end - selectedRange.start} samples
            </div>
          )}
        </div>
      )}
      {isFocused && (
        <div className="absolute top-4 right-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Hold Shift + Drag to select time range</div>
        </div>
      )}
    </div>
  );

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
    <>
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900 transition-colors duration-300 relative">
        <div className="flex items-center justify-between mb-4">
          <div className="text-cyan-500 font-mono text-xs tracking-wider uppercase">TIME DOMAIN</div>
          <div className="text-xs text-gray-500 font-mono">
            {(sampleRate / 1000).toFixed(1)} ksps
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
          {renderWaveform()}
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
      <FocusMode 
        isOpen={isFocused} 
        onClose={() => { 
          setIsFocused(false); 
          setZoomLevel(1); 
          setPanOffset(0); 
          setSelectedRange(null);
        }} 
        title="TIME DOMAIN WAVEFORM"
      >
        {renderWaveform()}
        <div className="mt-4 flex gap-6 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-cyan-500"></div>
            <span className="text-gray-400">I (In-Phase)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-purple-400"></div>
            <span className="text-gray-400">Q (Quadrature)</span>
          </div>
          <div className="text-gray-500">Sample Rate: {(sampleRate / 1000).toFixed(1)} ksps</div>
        </div>
      </FocusMode>
    </>
  );
}
