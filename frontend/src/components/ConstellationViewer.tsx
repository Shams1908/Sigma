import { useEffect, useRef, useMemo, useState, memo } from 'react';
import FocusMode from './FocusMode';

interface ConstellationPoint {
  i: number;
  q: number;
}

interface ConstellationViewerProps {
  data: ConstellationPoint[];
}

const MAX_DISPLAY_POINTS = 2500;

const ConstellationViewer = memo(function ConstellationViewer({ data }: ConstellationViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const displayData = useMemo(() => {
    if (data.length <= MAX_DISPLAY_POINTS) return data;
    const stride = Math.ceil(data.length / MAX_DISPLAY_POINTS);
    const downsampled: ConstellationPoint[] = [];
    for (let i = 0; i < data.length; i += stride) {
      downsampled.push(data[i]);
    }
    return downsampled;
  }, [data]);

  const { width, height, extent } = useMemo(() => {
    if (!displayData.length) return { width: 400, height: 400, extent: 1 };

    const w = isFocused ? 1000 : 400;
    const h = isFocused ? 1000 : 400;

    const iValues = displayData.map(d => d.i);
    const qValues = displayData.map(d => d.q);
    const maxI = Math.max(...iValues.map(Math.abs));
    const maxQ = Math.max(...qValues.map(Math.abs));
    const baseExtent = Math.max(maxI, maxQ) * 1.25;

    return { width: w, height: h, extent: baseExtent };
  }, [displayData, isFocused]);

  useEffect(() => {
    if (!isFocused) {
      setZoomLevel(1);
      setPanOffset({ x: 0, y: 0 });
    }
  }, [data.length, isFocused]);

  const drawConstellation = (canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const margin = isFocused ? 80 : 40;
    const innerSize = Math.min(w, h) - 2 * margin;
    const centerX = w / 2;
    const centerY = h / 2;

    const viewExtent = extent / zoomLevel;
    const centerI = panOffset.x * viewExtent;
    const centerQ = panOffset.y * viewExtent;

    const scaleX = (v: number) => {
      const relativeV = v - centerI;
      return centerX + (relativeV / viewExtent) * (innerSize / 2);
    };

    const scaleY = (v: number) => {
      const relativeV = v - centerQ;
      return centerY - (relativeV / viewExtent) * (innerSize / 2);
    };

    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, w, h);

    if (isFocused) {
      ctx.strokeStyle = 'rgb(30, 30, 30)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      for (let i = 0; i <= 20; i++) {
        const val = -extent + (i / 20) * 2 * extent + panOffset.x * extent * zoomLevel;
        const x = scaleX(val);
        ctx.moveTo(x, margin);
        ctx.lineTo(x, h - margin);
      }
      for (let i = 0; i <= 20; i++) {
        const val = -extent + (i / 20) * 2 * extent + panOffset.y * extent * zoomLevel;
        const y = scaleY(val);
        ctx.moveTo(margin, y);
        ctx.lineTo(w - margin, y);
      }
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgb(71, 85, 105)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(margin, centerY);
    ctx.lineTo(w - margin, centerY);
    ctx.moveTo(centerX, margin);
    ctx.lineTo(centerX, h - margin);
    ctx.stroke();

    ctx.fillStyle = 'rgb(71, 85, 105)';
    ctx.font = '10px monospace';
    ctx.fillText('I', w - margin + 20, centerY - 8);
    ctx.fillText('Q', centerX + 8, margin - 10);

    const tickValues = isFocused ? [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] : [-0.5, 0.5];
    ctx.strokeStyle = 'rgb(71, 85, 105)';
    ctx.lineWidth = 1;
    ctx.fillStyle = 'rgb(71, 85, 105)';
    ctx.font = '9px monospace';
    
    tickValues.forEach(val => {
      const adjustedValI = val * extent + panOffset.x * extent * zoomLevel;
      const posX = scaleX(adjustedValI);
      ctx.beginPath();
      ctx.moveTo(posX, centerY - 3);
      ctx.lineTo(posX, centerY + 3);
      ctx.stroke();
      if (isFocused) {
        ctx.fillText(adjustedValI.toFixed(2), posX - 12, centerY + 15);
      }

      const adjustedValQ = val * extent + panOffset.y * extent * zoomLevel;
      const posY = scaleY(adjustedValQ);
      ctx.beginPath();
      ctx.moveTo(centerX - 3, posY);
      ctx.lineTo(centerX + 3, posY);
      ctx.stroke();
      if (isFocused) {
        ctx.fillText(adjustedValQ.toFixed(2), centerX - 35, posY + 4);
      }
    });

    const pointRadius = isFocused ? Math.min(6, 2.5 * zoomLevel) : 2.5;
    ctx.fillStyle = 'rgba(20, 184, 166, 0.7)';
    
    displayData.forEach(point => {
      const x = scaleX(point.i);
      const y = scaleY(point.q);
      if (x >= margin && x <= w - margin && y >= margin && y <= h - margin) {
        ctx.fillRect(x - pointRadius / 2, y - pointRadius / 2, pointRadius, pointRadius);
      }
    });

    if (isFocused && cursor) {
      ctx.strokeStyle = 'rgb(168, 85, 247)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursor.x, margin);
      ctx.lineTo(cursor.x, h - margin);
      ctx.moveTo(margin, cursor.y);
      ctx.lineTo(w - margin, cursor.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    if (!displayData.length || !canvasRef.current) return;
    
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
    }

    animFrameRef.current = requestAnimationFrame(() => {
      if (canvasRef.current) {
        drawConstellation(canvasRef.current);
      }
    });

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [displayData, isFocused, zoomLevel, panOffset, cursor, width, height, extent]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      if (canvasRef.current) {
        drawConstellation(canvasRef.current);
      }
    });

    resizeObserver.observe(canvasRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [displayData, isFocused, zoomLevel, panOffset, extent]);

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

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
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
    if (!cursor) return null;
    const margin = isFocused ? 80 : 40;
    const innerSize = Math.min(width, height) - 2 * margin;
    const centerX = width / 2;
    const centerY = height / 2;

    const viewExtent = extent / zoomLevel;
    const centerI = panOffset.x * viewExtent;
    const centerQ = panOffset.y * viewExtent;

    const iVal = ((cursor.x - centerX) / (innerSize / 2)) * viewExtent + centerI;
    const qVal = -((cursor.y - centerY) / (innerSize / 2)) * viewExtent + centerQ;

    return { i: iVal, q: qVal };
  };

  const cursorValues = getCursorValues();

  const renderConstellation = () => (
    <div className="w-full h-full relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
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
          <div className="text-sigma-teal">I: {cursorValues.i.toFixed(4)}</div>
          <div className="text-sigma-purple">Q: {cursorValues.q.toFixed(4)}</div>
        </div>
      )}
      {isFocused && (
        <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Zoom: {zoomLevel.toFixed(1)}x</div>
        </div>
      )}
    </div>
  );

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO CONSTELLATION DATA</p>
      </div>
    );
  }

  return (
    <>
      <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal/30 transition-colors duration-300">
        <div className="absolute top-4 left-4 z-10">
          <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">CONSTELLATION</div>
        </div>
        <div className="absolute top-4 right-14 z-10 bg-[#0A0A0A] border border-sigma-teal/30 px-3 py-1.5 rounded">
          <div className="text-xs font-mono text-slate-400 uppercase">POINTS</div>
          <div className="text-sm font-mono text-sigma-teal">{data.length}</div>
        </div>
        <button
          onClick={() => setIsFocused(true)}
          className="absolute top-4 right-4 z-10 p-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-sigma-teal rounded-lg transition-all"
          title="Expand to Focus Mode"
        >
          <svg className="w-4 h-4 text-sigma-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        {renderConstellation()}
      </div>
      <FocusMode 
        isOpen={isFocused} 
        onClose={() => { 
          setIsFocused(false); 
          setZoomLevel(1); 
          setPanOffset({ x: 0, y: 0 }); 
        }} 
        title="CONSTELLATION DIAGRAM"
      >
        {renderConstellation()}
      </FocusMode>
    </>
  );
});

export default ConstellationViewer;
