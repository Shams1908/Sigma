import { useEffect, useRef, useMemo, useState } from 'react';
import FocusMode from './FocusMode';

interface ConstellationPoint {
  i: number;
  q: number;
}

interface ConstellationViewerProps {
  data: ConstellationPoint[];
}

export default function ConstellationViewer({ data }: ConstellationViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const { width, height, scale, extent } = useMemo(() => {
    if (!data.length) return { width: 400, height: 400, scale: (v: number) => v, extent: 1 };

    const w = isFocused ? 1000 : 400;
    const h = isFocused ? 1000 : 400;
    const margin = isFocused ? 80 : 40;
    const innerSize = Math.min(w, h) - 2 * margin;

    const iValues = data.map(d => d.i);
    const qValues = data.map(d => d.q);
    const maxI = Math.max(...iValues.map(Math.abs));
    const maxQ = Math.max(...qValues.map(Math.abs));
    const baseExtent = Math.max(maxI, maxQ) * 1.25;
    
    const viewExtent = baseExtent / zoomLevel;
    const centerI = panOffset.x * baseExtent;
    const centerQ = panOffset.y * baseExtent;

    const scale = (v: number, isQ: boolean = false) => {
      const offset = isQ ? centerQ : centerI;
      const relativeV = v - offset;
      return margin + innerSize / 2 + (relativeV / viewExtent) * (innerSize / 2);
    };

    return { width: w, height: h, scale, extent: viewExtent };
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
    if (!cursor) return null;
    const margin = isFocused ? 80 : 40;
    const innerSize = Math.min(width, height) - 2 * margin;
    const centerX = width / 2;
    const centerY = height / 2;

    const iVal = ((cursor.x - centerX) / (innerSize / 2)) * extent + panOffset.x * extent * zoomLevel;
    const qVal = -((cursor.y - centerY) / (innerSize / 2)) * extent + panOffset.y * extent * zoomLevel;

    return { i: iVal, q: qVal };
  };

  const cursorValues = getCursorValues();
  const pointRadius = isFocused ? Math.max(1.5, 3 / zoomLevel) : 2.5;

  useEffect(() => {
    const circles = svgRef.current?.querySelectorAll('.constellation-point');
    circles?.forEach((circle, i) => {
      const delay = Math.floor(i / 25) * 12;
      (circle as SVGElement).style.animationDelay = `${delay}ms`;
    });
  }, [data]);

  const centerX = width / 2;
  const centerY = height / 2;
  const margin = isFocused ? 80 : 40;

  const renderConstellation = () => (
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
          <style>
            {`
              @keyframes constellation-emerge {
                0% {
                  opacity: 0;
                  transform: scale(0);
                }
                100% {
                  opacity: 0.7;
                  transform: scale(1);
                }
              }
              .constellation-point {
                animation: constellation-emerge 0.4s ease-out forwards;
                transform-origin: center;
                opacity: 0;
              }
            `}
          </style>
        </defs>

        {isFocused && (
          <>
            {Array.from({ length: 21 }, (_, i) => {
              const val = -extent + (i / 20) * 2 * extent + panOffset.x * extent * zoomLevel;
              const x = scale(val, false);
              return (
                <line
                  key={`vgrid-${i}`}
                  x1={x}
                  y1={margin}
                  x2={x}
                  y2={height - margin}
                  stroke="rgb(30, 30, 30)"
                  strokeWidth="0.5"
                />
              );
            })}
            {Array.from({ length: 21 }, (_, i) => {
              const val = -extent + (i / 20) * 2 * extent + panOffset.y * extent * zoomLevel;
              const y = scale(val, true);
              return (
                <line
                  key={`hgrid-${i}`}
                  x1={margin}
                  y1={y}
                  x2={width - margin}
                  y2={y}
                  stroke="rgb(30, 30, 30)"
                  strokeWidth="0.5"
                />
              );
            })}
          </>
        )}

        <line
          x1={margin}
          y1={centerY}
          x2={width - margin}
          y2={centerY}
          stroke="rgb(71, 85, 105)"
          strokeWidth="1.5"
        />
        <line
          x1={centerX}
          y1={margin}
          x2={centerX}
          y2={height - margin}
          stroke="rgb(71, 85, 105)"
          strokeWidth="1.5"
        />

        <text
          x={width - margin + 20}
          y={centerY - 8}
          className="text-xs font-mono fill-slate-500"
          textAnchor="middle"
        >
          I
        </text>
        <text
          x={centerX + 8}
          y={margin - 10}
          className="text-xs font-mono fill-slate-500"
          textAnchor="middle"
        >
          Q
        </text>

        {(isFocused ? [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] : [-0.5, 0.5]).map(val => {
          const adjustedVal = val * extent + panOffset.x * extent * zoomLevel;
          const pos = scale(adjustedVal, false);
          const qAdjustedVal = val * extent + panOffset.y * extent * zoomLevel;
          const qPos = scale(qAdjustedVal, true);
          
          return (
            <g key={val}>
              <line
                x1={pos}
                y1={centerY - 3}
                x2={pos}
                y2={centerY + 3}
                stroke="rgb(71, 85, 105)"
                strokeWidth="1"
              />
              {isFocused && (
                <text
                  x={pos}
                  y={centerY + 15}
                  className="text-[10px] font-mono fill-slate-500"
                  textAnchor="middle"
                >
                  {adjustedVal.toFixed(2)}
                </text>
              )}
              <line
                x1={centerX - 3}
                y1={qPos}
                x2={centerX + 3}
                y2={qPos}
                stroke="rgb(71, 85, 105)"
                strokeWidth="1"
              />
              {isFocused && (
                <text
                  x={centerX - 20}
                  y={qPos + 4}
                  className="text-[10px] font-mono fill-slate-500"
                  textAnchor="end"
                >
                  {(-qAdjustedVal).toFixed(2)}
                </text>
              )}
            </g>
          );
        })}

        {data.map((point, i) => (
          <circle
            key={i}
            className="constellation-point"
            cx={scale(point.i, false)}
            cy={scale(-point.q, true)}
            r={pointRadius}
            fill="rgb(20, 184, 166)"
            fillOpacity="0.7"
          />
        ))}

        {isFocused && cursor && (
          <>
            <line
              x1={cursor.x}
              y1={margin}
              x2={cursor.x}
              y2={height - margin}
              stroke="rgb(168, 85, 247)"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.6"
            />
            <line
              x1={margin}
              y1={cursor.y}
              x2={width - margin}
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
      <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal-900 transition-colors duration-300">
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
}
