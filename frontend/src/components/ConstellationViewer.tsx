import { useEffect, useRef, useMemo, useState, memo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface ConstellationPoint {
  i: number;
  q: number;
}

interface ConstellationViewerProps {
  data: ConstellationPoint[];
}

const MAX_DISPLAY_POINTS = 2500;
const MAX_3D_POINTS = 15000;

type TraceMode = 'scatter' | 'density' | 'trajectory' | '3d_helix';
type TargetModulation = 'none' | 'bpsk' | 'qpsk' | '16qam' | '64qam';

const Constellation3D = memo(({ iqData, showUnitCircle, onHover, onHoverEnd }: { 
  iqData: ConstellationPoint[]; 
  showUnitCircle: boolean;
  onHover: (data: { x: number; y: number; i: number; q: number; index: number }) => void;
  onHoverEnd: () => void;
}) => {
  const { positions, pointCount, timeDepth, safeData } = useMemo(() => {
    const safe = iqData.length > 15000 ? iqData.slice(-15000) : iqData;
    const count = safe.length;
    const posArray = new Float32Array(count * 3);
    const timeScale = count > 0 ? 10 / count : 0.001;
    const depth = count * timeScale;

    for (let i = 0; i < count; i++) {
      posArray[i * 3] = safe[i].i;
      posArray[i * 3 + 1] = safe[i].q;
      posArray[i * 3 + 2] = -(i * timeScale);
    }

    return { positions: posArray, pointCount: count, timeDepth: depth, safeData: safe };
  }, [iqData]);

  const geometryRef = useRef<THREE.BufferGeometry>(null);
  const lastHoveredIndexRef = useRef<number | null>(null);

  useEffect(() => {
    if (geometryRef.current) {
      geometryRef.current.attributes.position.needsUpdate = true;
      geometryRef.current.computeBoundingSphere();
      geometryRef.current.computeBoundingBox();
    }
  }, [positions]);

  const handlePointerMove = (e: any) => {
    e.stopPropagation();
    
    if (e.index === undefined || e.index >= safeData.length) return;
    
    if (lastHoveredIndexRef.current === e.index) return;
    
    lastHoveredIndexRef.current = e.index;
    
    const realPoint = safeData[e.index];
    
    if (realPoint && realPoint.i !== undefined && realPoint.q !== undefined) {
      onHover({
        x: e.clientX,
        y: e.clientY,
        i: realPoint.i,
        q: realPoint.q,
        index: e.index
      });
    }
  };

  const handlePointerOut = (e: any) => {
    e.stopPropagation();
    lastHoveredIndexRef.current = null;
    onHoverEnd();
  };

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <axesHelper args={[2]} />
      <line raycast={() => null}>
        <bufferGeometry ref={geometryRef}>
          <bufferAttribute
            attach="attributes-position"
            count={pointCount}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#00E5FF"
          transparent={true}
          opacity={0.5}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </line>
      <points 
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      >
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={pointCount}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          color="#00E5FF"
          sizeAttenuation={true}
          transparent={true}
          opacity={1.0}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
      {showUnitCircle && (
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -timeDepth / 2]} raycast={() => null}>
          <cylinderGeometry args={[1, 1, timeDepth, 32, 1, true]} />
          <meshBasicMaterial
            color="#8d5e50"
            wireframe={true}
            transparent={true}
            opacity={0.6}
            depthWrite={false}
          />
        </mesh>
      )}
      <OrbitControls dampingFactor={0.05} enableDamping makeDefault />
    </>
  );
});

const ConstellationViewer = memo(function ConstellationViewer({ data }: ConstellationViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const [traceMode, setTraceMode] = useState<TraceMode>('scatter');
  const [targetModulation, setTargetModulation] = useState<TargetModulation>('none');
  const [showUnitCircle, setShowUnitCircle] = useState(false);
  const [showEvmVectors, setShowEvmVectors] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; i: number; q: number; index: number } | null>(null);

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

    const w = isExpanded ? 1000 : 400;
    const h = isExpanded ? 1000 : 400;

    const iValues = displayData.map(d => d.i);
    const qValues = displayData.map(d => d.q);
    const maxI = Math.max(...iValues.map(Math.abs));
    const maxQ = Math.max(...qValues.map(Math.abs));
    const baseExtent = Math.max(maxI, maxQ) * 1.25;

    return { width: w, height: h, extent: baseExtent };
  }, [displayData, isExpanded]);

  useEffect(() => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  }, [data]);

  const getIdealPoints = (mod: TargetModulation): Array<{ i: number; q: number }> => {
    if (mod === 'bpsk') {
      return [{ i: -1, q: 0 }, { i: 1, q: 0 }];
    } else if (mod === 'qpsk') {
      const s = Math.sqrt(2) / 2;
      return [
        { i: s, q: s },
        { i: s, q: -s },
        { i: -s, q: s },
        { i: -s, q: -s }
      ];
    } else if (mod === '16qam') {
      const pts: Array<{ i: number; q: number }> = [];
      for (let i = -3; i <= 3; i += 2) {
        for (let q = -3; q <= 3; q += 2) {
          pts.push({ i: i / 3, q: q / 3 });
        }
      }
      return pts;
    } else if (mod === '64qam') {
      const pts: Array<{ i: number; q: number }> = [];
      for (let i = -7; i <= 7; i += 2) {
        for (let q = -7; q <= 7; q += 2) {
          pts.push({ i: i / 7, q: q / 7 });
        }
      }
      return pts;
    }
    return [];
  };

  const findNearestIdeal = (point: ConstellationPoint, idealPoints: Array<{ i: number; q: number }>) => {
    let minDist = Infinity;
    let nearest = idealPoints[0];
    for (const ideal of idealPoints) {
      const dist = Math.sqrt((point.i - ideal.i) ** 2 + (point.q - ideal.q) ** 2);
      if (dist < minDist) {
        minDist = dist;
        nearest = ideal;
      }
    }
    return nearest;
  };

  const drawConstellationExpanded = (canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const centerX = w / 2;
    const centerY = h / 2;

    const viewExtent = extent / zoomLevel;
    const centerI = panOffset.x * viewExtent;
    const centerQ = panOffset.y * viewExtent;

    const scaleX = (v: number) => {
      const relativeV = v - centerI;
      return centerX + (relativeV / viewExtent) * (Math.min(w, h) / 2);
    };

    const scaleY = (v: number) => {
      const relativeV = v - centerQ;
      return centerY - (relativeV / viewExtent) * (Math.min(w, h) / 2);
    };

    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, h);
    ctx.stroke();

    if (showUnitCircle) {
      const radius = (1.0 / viewExtent) * (Math.min(w, h) / 2);
      ctx.strokeStyle = '#444444';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    const idealPoints = getIdealPoints(targetModulation);
    if (idealPoints.length > 0) {
      ctx.fillStyle = 'rgba(255, 50, 50, 0.6)';
      ctx.strokeStyle = 'rgba(255, 50, 50, 0.8)';
      ctx.lineWidth = 1;
      idealPoints.forEach(pt => {
        const x = scaleX(pt.i);
        const y = scaleY(pt.q);
        ctx.beginPath();
        ctx.moveTo(x - 6, y);
        ctx.lineTo(x + 6, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y - 6);
        ctx.lineTo(x, y + 6);
        ctx.stroke();
      });
    }

    if (traceMode === 'scatter') {
      const basePointSize = 2;
      const dynamicSize = Math.max(1.5, basePointSize * zoomLevel);
      ctx.fillStyle = 'rgba(20, 184, 166, 0.8)';
      displayData.forEach(point => {
        const x = scaleX(point.i);
        const y = scaleY(point.q);
        ctx.fillRect(x - dynamicSize / 2, y - dynamicSize / 2, dynamicSize, dynamicSize);
      });
    } else if (traceMode === 'density') {
      const basePointSize = 3;
      const dynamicSize = Math.max(2, basePointSize * zoomLevel);
      ctx.globalAlpha = 0.08;
      ctx.fillStyle = '#ffffff';
      displayData.forEach(point => {
        const x = scaleX(point.i);
        const y = scaleY(point.q);
        ctx.fillRect(x - dynamicSize / 2, y - dynamicSize / 2, dynamicSize, dynamicSize);
      });
      ctx.globalAlpha = 1.0;
    } else if (traceMode === 'trajectory') {
      const dynamicLineWidth = Math.max(0.5, 0.5 * zoomLevel);
      ctx.strokeStyle = 'rgba(20, 184, 166, 0.4)';
      ctx.lineWidth = dynamicLineWidth;
      ctx.beginPath();
      displayData.forEach((point, i) => {
        const x = scaleX(point.i);
        const y = scaleY(point.q);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    if (showEvmVectors && idealPoints.length > 0) {
      ctx.strokeStyle = 'rgba(255, 50, 50, 0.3)';
      ctx.lineWidth = 1;
      displayData.forEach(point => {
        const nearest = findNearestIdeal(point, idealPoints);
        const x1 = scaleX(point.i);
        const y1 = scaleY(point.q);
        const x2 = scaleX(nearest.i);
        const y2 = scaleY(nearest.q);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      });
    }

    if (cursor) {
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursor.x, 0);
      ctx.lineTo(cursor.x, h);
      ctx.moveTo(0, cursor.y);
      ctx.lineTo(w, cursor.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  const drawConstellationBase = (canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const margin = 40;
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

    const tickValues = [-0.5, 0.5];
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

      const adjustedValQ = val * extent + panOffset.y * extent * zoomLevel;
      const posY = scaleY(adjustedValQ);
      ctx.beginPath();
      ctx.moveTo(centerX - 3, posY);
      ctx.lineTo(centerX + 3, posY);
      ctx.stroke();
    });

    const basePointSize = 2;
    const dynamicPointRadius = Math.max(1.5, basePointSize * zoomLevel);
    ctx.fillStyle = 'rgba(20, 184, 166, 0.7)';

    displayData.forEach(point => {
      const x = scaleX(point.i);
      const y = scaleY(point.q);
      if (x >= margin && x <= w - margin && y >= margin && y <= h - margin) {
        ctx.fillRect(x - dynamicPointRadius / 2, y - dynamicPointRadius / 2, dynamicPointRadius, dynamicPointRadius);
      }
    });
  };

  useEffect(() => {
    if (!displayData.length || !canvasRef.current) return;

    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
    }

    animFrameRef.current = requestAnimationFrame(() => {
      if (canvasRef.current) {
        if (isExpanded) {
          drawConstellationExpanded(canvasRef.current);
        } else {
          drawConstellationBase(canvasRef.current);
        }
      }
    });

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [displayData, isExpanded, zoomLevel, panOffset, cursor, width, height, extent, traceMode, targetModulation, showUnitCircle, showEvmVectors]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      if (canvasRef.current) {
        if (isExpanded) {
          drawConstellationExpanded(canvasRef.current);
        } else {
          drawConstellationBase(canvasRef.current);
        }
      }
    });

    resizeObserver.observe(canvasRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [displayData, isExpanded, zoomLevel, panOffset, extent, traceMode, targetModulation, showUnitCircle, showEvmVectors]);

  const handleWheel = (e: React.WheelEvent) => {
    if (!isExpanded) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel(prev => Math.max(1, Math.min(10, prev * delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isExpanded) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isExpanded) return;

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
    if (!cursor || !isExpanded) return null;
    const centerX = width / 2;
    const centerY = height / 2;
    const viewExtent = extent / zoomLevel;
    const centerI = panOffset.x * viewExtent;
    const centerQ = panOffset.y * viewExtent;
    const iVal = ((cursor.x - centerX) / (Math.min(width, height) / 2)) * viewExtent + centerI;
    const qVal = -((cursor.y - centerY) / (Math.min(width, height) / 2)) * viewExtent + centerQ;
    return { i: iVal, q: qVal };
  };

  const cursorValues = getCursorValues();

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO CONSTELLATION DATA</p>
      </div>
    );
  }

  if (isExpanded) {
    return (
      <div className="fixed inset-0 z-50 bg-[#050505] flex flex-col w-screen h-screen overflow-hidden text-[11px] text-gray-300 font-mono">
        <div className="h-12 bg-[#0A0A0A] border-b border-[#222] flex items-center justify-between px-4 shrink-0">
          <div className="text-sm font-bold text-sigma-teal uppercase tracking-wider">
            CONSTELLATION FOCUS MODE
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setTraceMode('scatter')}
              className={`px-3 py-1 border rounded transition-all ${
                traceMode === 'scatter'
                  ? 'border-sigma-teal text-sigma-teal bg-sigma-teal/10'
                  : 'border-[#222] text-gray-400 hover:border-sigma-teal/50'
              }`}
            >
              SCATTER
            </button>
            <button
              onClick={() => setTraceMode('density')}
              className={`px-3 py-1 border rounded transition-all ${
                traceMode === 'density'
                  ? 'border-sigma-teal text-sigma-teal bg-sigma-teal/10'
                  : 'border-[#222] text-gray-400 hover:border-sigma-teal/50'
              }`}
            >
              DENSITY
            </button>
            <button
              onClick={() => setTraceMode('trajectory')}
              className={`px-3 py-1 border rounded transition-all ${
                traceMode === 'trajectory'
                  ? 'border-sigma-teal text-sigma-teal bg-sigma-teal/10'
                  : 'border-[#222] text-gray-400 hover:border-sigma-teal/50'
              }`}
            >
              TRAJECTORY
            </button>
            <button
              onClick={() => setTraceMode('3d_helix')}
              className={`px-3 py-1 border rounded transition-all ${
                traceMode === '3d_helix'
                  ? 'text-[#00E5FF] border-[#00E5FF] bg-[#00E5FF]/10 shadow-[0_0_8px_rgba(0,229,255,0.4)]'
                  : 'border-[#222] text-gray-400 bg-[#0A0A0A] hover:border-[#00E5FF]/50'
              }`}
            >
              3D HELIX
            </button>
          </div>
          <button
            onClick={() => {
              setIsExpanded(false);
              setZoomLevel(1);
              setPanOffset({ x: 0, y: 0 });
              setCursor(null);
            }}
            className="px-4 py-1 border border-[#222] hover:border-red-500 text-gray-400 hover:text-red-400 rounded transition-all"
          >
            CLOSE [ESC]
          </button>
        </div>

        <div className="flex-1 flex flex-row overflow-hidden">
          <div className="w-64 bg-[#0A0A0A] border-r border-[#222] flex flex-col p-4 shrink-0 space-y-6">
            <div className="bg-[#111] border border-[#222] rounded">
              <div className="bg-[#0A0A0A] px-3 py-2 border-b border-[#222] text-xs uppercase tracking-wider text-gray-400">
                Reference Overlays
              </div>
              <div className="p-3 space-y-3">
                <div>
                  <label className="block text-[10px] uppercase text-gray-500 mb-1">Target Modulation</label>
                  <select
                    value={targetModulation}
                    onChange={(e) => setTargetModulation(e.target.value as TargetModulation)}
                    className="w-full bg-[#0A0A0A] border border-[#222] text-gray-300 px-2 py-1 rounded text-xs focus:outline-none focus:border-sigma-teal"
                  >
                    <option value="none">None</option>
                    <option value="bpsk">BPSK</option>
                    <option value="qpsk">QPSK</option>
                    <option value="16qam">16-QAM</option>
                    <option value="64qam">64-QAM</option>
                  </select>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showUnitCircle}
                    onChange={(e) => setShowUnitCircle(e.target.checked)}
                    className="w-3 h-3"
                  />
                  <span className="text-xs">Show Unit Circle</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showEvmVectors}
                    onChange={(e) => setShowEvmVectors(e.target.checked)}
                    className="w-3 h-3"
                  />
                  <span className="text-xs">Show EVM Vectors</span>
                </label>
              </div>
            </div>

            <div className="bg-[#111] border border-[#222] rounded">
              <div className="bg-[#0A0A0A] px-3 py-2 border-b border-[#222] text-xs uppercase tracking-wider text-gray-400">
                Signal Info
              </div>
              <div className="p-3 space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Total Points</span>
                  <span className="text-sigma-teal font-bold">{data.length}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Rendered</span>
                  <span className="text-sigma-teal font-bold">{displayData.length}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-1 flex bg-[#050505] relative items-center justify-center p-4">
            <div className="aspect-square h-full max-h-[85vh] border border-[#222] bg-[#0A0A0A] relative">
              {traceMode === '3d_helix' ? (
                <>
                  <Canvas
                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
                    camera={{ position: [5, 5, 10], fov: 45 }}
                  >
                    <Constellation3D 
                      iqData={data} 
                      showUnitCircle={showUnitCircle}
                      onHover={setHoverInfo}
                      onHoverEnd={() => setHoverInfo(null)}
                    />
                  </Canvas>
                  {hoverInfo && (
                    <div
                      style={{
                        position: 'fixed',
                        top: hoverInfo.y + 15,
                        left: hoverInfo.x + 15,
                        pointerEvents: 'none',
                        zIndex: 9999
                      }}
                      className="bg-[#050505] border border-[#00E5FF] shadow-[0_0_10px_rgba(0,229,255,0.2)] p-2 font-mono text-[10px] text-gray-300 flex flex-col space-y-1 min-w-[120px]"
                    >
                      <div className="text-[#00E5FF] font-bold border-b border-[#222] pb-1 mb-1">DATA POINT</div>
                      <div>I: {hoverInfo.i.toFixed(4)}</div>
                      <div>Q: {hoverInfo.q.toFixed(4)}</div>
                      <div>IDX: {hoverInfo.index}</div>
                    </div>
                  )}
                  <div className="absolute bottom-4 right-4 z-10 bg-[#0A0A0A] border border-[#222] p-3 flex flex-col space-y-2 font-mono text-[10px] text-gray-400 shadow-2xl pointer-events-none">
                    <div className="flex items-center gap-2">
                      <div className="bg-red-500 w-2 h-2"></div>
                      <span>X-AXIS : IN-PHASE (I)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="bg-green-500 w-2 h-2"></div>
                      <span>Y-AXIS : QUADRATURE (Q)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="bg-blue-500 w-2 h-2"></div>
                      <span>Z-AXIS : TIME (t)</span>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <canvas
                    ref={canvasRef}
                    className="w-full h-full"
                    onWheel={handleWheel}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                    style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
                  />
                  {cursor && cursorValues && (
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
                  <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222] px-3 py-2 rounded z-10">
                    <div className="text-xs font-mono text-gray-400">Zoom: {zoomLevel.toFixed(1)}x</div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal/30 transition-colors duration-300">
      <div className="absolute top-4 left-4 z-10">
        <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">CONSTELLATION</div>
      </div>
      <div className="absolute top-4 right-14 z-10 bg-[#0A0A0A] border border-sigma-teal/30 px-3 py-1.5 rounded">
        <div className="text-xs font-mono text-slate-400 uppercase">POINTS</div>
        <div className="text-sm font-mono text-sigma-teal">{data.length}</div>
      </div>
      <button
        onClick={() => setIsExpanded(true)}
        className="absolute top-4 right-4 z-10 p-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-sigma-teal rounded-lg transition-all"
        title="Expand to Focus Mode"
      >
        <svg className="w-4 h-4 text-sigma-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
      />
    </div>
  );
});

export default ConstellationViewer;
