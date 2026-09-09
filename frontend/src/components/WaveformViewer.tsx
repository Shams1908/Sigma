import { useEffect, useRef, useState, memo, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import FocusMode from './FocusMode';

interface WaveformViewerProps {
  iData: number[];
  qData: number[];
  timeData: number[];
  sampleRate: number;
}

interface IQDataPoint {
  i: number;
  q: number;
}

interface TimeDomain3DProps {
  iqData: IQDataPoint[];
  onHoverChange: (hover: { 
    x: number; 
    y: number; 
    timeVal: number; 
    iVal: number; 
    qVal: number; 
    index: number; 
    coord3D: [number, number, number] 
  } | null) => void;
}

const TimeDomain3D = memo(function TimeDomain3D({ iqData, onHoverChange }: TimeDomain3DProps) {
  const hoverIndexRef = useRef<number | null>(null);
  const [hover3DLocal, setHover3DLocal] = useState<{ coord3D: [number, number, number] } | null>(null);

  const geometries = useMemo(() => {
    const safeData = iqData.length > 5000 ? iqData.slice(-5000) : iqData;
    const timeScale = 20 / safeData.length;
    
    const iArray = new Float32Array(safeData.length * 3);
    const qArray = new Float32Array(safeData.length * 3);
    const compositeArray = new Float32Array(safeData.length * 3);
    
    for (let i = 0; i < safeData.length; i++) {
      const idx = i * 3;
      const xPos = (i - safeData.length / 2) * timeScale;
      
      iArray[idx] = xPos;
      iArray[idx + 1] = safeData[i].i;
      iArray[idx + 2] = 0;
      
      qArray[idx] = xPos;
      qArray[idx + 1] = 0;
      qArray[idx + 2] = safeData[i].q;
      
      compositeArray[idx] = xPos;
      compositeArray[idx + 1] = safeData[i].i;
      compositeArray[idx + 2] = safeData[i].q;
    }
    
    return { iArray, qArray, compositeArray, timeScale, dataLength: safeData.length, safeData };
  }, [iqData]);

  const axisGeometries = useMemo(() => {
    const halfTime = (geometries.dataLength / 2) * geometries.timeScale;
    
    const xAxisPositions = new Float32Array([-halfTime, 0, 0, halfTime, 0, 0]);
    const yAxisPositions = new Float32Array([0, -1.5, 0, 0, 1.5, 0]);
    const zAxisPositions = new Float32Array([0, 0, -1.5, 0, 0, 1.5]);
    
    return { 
      xAxis: xAxisPositions, 
      yAxis: yAxisPositions, 
      zAxis: zAxisPositions,
      startPos: [-halfTime, 0, 0] as [number, number, number],
      endPos: [halfTime, 0, 0] as [number, number, number]
    };
  }, [geometries.dataLength, geometries.timeScale]);

  const handlePointerMoveLocal = (e: any) => {
    e.stopPropagation();
    if (e.index === undefined) return;
    if (hoverIndexRef.current === e.index) return;
    
    hoverIndexRef.current = e.index;
    const realPoint = geometries.safeData[e.index];
    const xPos = (e.index - geometries.safeData.length / 2) * geometries.timeScale;
    const coord3D: [number, number, number] = [xPos, realPoint.i, realPoint.q];
    
    setHover3DLocal({ coord3D });
    
    onHoverChange({
      x: e.clientX,
      y: e.clientY,
      timeVal: e.index,
      iVal: realPoint.i,
      qVal: realPoint.q,
      index: e.index,
      coord3D
    });
  };

  const handlePointerOutLocal = () => {
    hoverIndexRef.current = null;
    setHover3DLocal(null);
    onHoverChange(null);
  };

  const laserGeometry = useMemo(() => {
    if (!hover3DLocal) return null;
    const positions = new Float32Array([
      hover3DLocal.coord3D[0], hover3DLocal.coord3D[1], 0,
      hover3DLocal.coord3D[0], hover3DLocal.coord3D[1], hover3DLocal.coord3D[2],
      hover3DLocal.coord3D[0], 0, hover3DLocal.coord3D[2]
    ]);
    return positions;
  }, [hover3DLocal]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={axisGeometries.xAxis.length / 3}
            array={axisGeometries.xAxis}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFFFF" opacity={0.6} transparent blending={THREE.AdditiveBlending} />
      </line>
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={axisGeometries.yAxis.length / 3}
            array={axisGeometries.yAxis}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00E5FF" opacity={0.4} transparent />
      </line>
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={axisGeometries.zAxis.length / 3}
            array={axisGeometries.zAxis}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#B200FF" opacity={0.4} transparent />
      </line>
      
      <mesh position={axisGeometries.startPos} raycast={() => null}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#00FF00" emissive="#00FF00" emissiveIntensity={0.5} />
      </mesh>
      <Html position={axisGeometries.startPos} center>
        <div className="font-mono text-[10px] text-green-400 bg-[#050505]/80 px-1 border border-green-500/50">START</div>
      </Html>
      
      <mesh position={axisGeometries.endPos} raycast={() => null}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#FF0044" emissive="#FF0044" emissiveIntensity={0.5} />
      </mesh>
      <Html position={axisGeometries.endPos} center>
        <div className="font-mono text-[10px] text-red-400 bg-[#050505]/80 px-1 border border-red-500/50">END</div>
      </Html>
      
      <Html position={[0, 1.1, 0]} center>
        <div className="font-mono text-[9px] text-[#00E5FF]">+1.0 (I)</div>
      </Html>
      <Html position={[0, -1.1, 0]} center>
        <div className="font-mono text-[9px] text-[#00E5FF]">-1.0 (I)</div>
      </Html>
      <Html position={[0, 0, 1.1]} center>
        <div className="font-mono text-[9px] text-[#B200FF]">+1.0 (Q)</div>
      </Html>
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.iArray.length / 3}
            array={geometries.iArray}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00E5FF" opacity={0.8} transparent />
      </line>
      
      <points raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.iArray.length / 3}
            array={geometries.iArray}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial color="#00E5FF" size={0.05} />
      </points>
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.qArray.length / 3}
            array={geometries.qArray}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#B200FF" opacity={0.8} transparent />
      </line>
      
      <points raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.qArray.length / 3}
            array={geometries.qArray}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial color="#B200FF" size={0.05} />
      </points>
      
      <line raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.compositeArray.length / 3}
            array={geometries.compositeArray}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFFFF" opacity={0.2} transparent blending={THREE.AdditiveBlending} />
      </line>
      
      <points onPointerMove={handlePointerMoveLocal} onPointerOut={handlePointerOutLocal}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={geometries.compositeArray.length / 3}
            array={geometries.compositeArray}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial color="#FFFFFF" size={0.08} transparent opacity={0.0} />
      </points>
      
      {hover3DLocal && laserGeometry && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={laserGeometry.length / 3}
              array={laserGeometry}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#FFFFFF" transparent opacity={0.8} />
        </lineSegments>
      )}
      
      <OrbitControls makeDefault />
      
      <gridHelper args={[40, 40, '#444444', '#222222']} raycast={() => null} />
    </>
  );
});

const WaveformViewer = memo(function WaveformViewer({ iData, qData, timeData, sampleRate }: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);
  const [selectedRange, setSelectedRange] = useState<{ start: number; end: number } | null>(null);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');
  const [hover3D, setHover3D] = useState<{ 
    x: number; 
    y: number; 
    timeVal: number; 
    iVal: number; 
    qVal: number; 
    index: number; 
    coord3D: [number, number, number] 
  } | null>(null);

  useEffect(() => {
    // Reset zoom, pan, and selection whenever data changes
    setZoomLevel(1);
    setPanOffset(0);
    setSelectedRange(null);
  }, [iData]);

  const drawWaveform = (canvas: HTMLCanvasElement, focused: boolean) => {
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
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
      ctx.beginPath();
      for (let i = 0; i <= gridLines; i++) {
        const y = (i / gridLines) * height;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        const x = (i / gridLines) * width;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      ctx.stroke();

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

    const pixelWidth = Math.floor(width);
    const samplesPerPixel = viewIData.length / pixelWidth;
    const showVertices = focused && zoomLevel > 4.0;

    if (samplesPerPixel > 2) {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = focused ? 2 : 1;
      ctx.beginPath();
      
      for (let px = 0; px < pixelWidth; px++) {
        const startIdx = Math.floor(px * samplesPerPixel);
        const endIdx = Math.floor((px + 1) * samplesPerPixel);
        
        let minI = Infinity;
        let maxI = -Infinity;
        
        for (let i = startIdx; i < endIdx && i < viewIData.length; i++) {
          const val = viewIData[i];
          if (val < minI) minI = val;
          if (val > maxI) maxI = val;
        }
        
        const yMin = midY - maxI * scaleY;
        const yMax = midY - minI * scaleY;
        
        ctx.moveTo(px, yMin);
        ctx.lineTo(px, yMax);
      }
      ctx.stroke();

      ctx.strokeStyle = '#a78bfa';
      ctx.lineWidth = focused ? 2 : 1;
      ctx.beginPath();
      
      for (let px = 0; px < pixelWidth; px++) {
        const startIdx = Math.floor(px * samplesPerPixel);
        const endIdx = Math.floor((px + 1) * samplesPerPixel);
        
        let minQ = Infinity;
        let maxQ = -Infinity;
        
        for (let i = startIdx; i < endIdx && i < viewQData.length; i++) {
          const val = viewQData[i];
          if (val < minQ) minQ = val;
          if (val > maxQ) maxQ = val;
        }
        
        const yMin = midY + minQ * scaleY;
        const yMax = midY + maxQ * scaleY;
        
        ctx.moveTo(px, yMin);
        ctx.lineTo(px, yMax);
      }
      ctx.stroke();
    } else {
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

      if (showVertices) {
        ctx.fillStyle = '#FFFFFF';
        viewIData.forEach((val, idx) => {
          const x = (idx / viewIData.length) * width;
          const y = midY - val * scaleY;
          ctx.fillRect(x - 1, y - 1, 2, 2);
        });

        ctx.fillStyle = '#00FFFF';
        viewQData.forEach((val, idx) => {
          const x = (idx / viewQData.length) * width;
          const y = midY + val * scaleY;
          ctx.fillRect(x - 1, y - 1, 2, 2);
        });
      }
    }

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
      ctx.moveTo(0, cursor.y);
      ctx.lineTo(width, cursor.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    if (!iData.length || !qData.length || !canvasRef.current) return;
    
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
    }

    animFrameRef.current = requestAnimationFrame(() => {
      if (canvasRef.current) {
        drawWaveform(canvasRef.current, isFocused);
      }
    });

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [iData, qData, timeData, isFocused, zoomLevel, panOffset, cursor, selectedRange]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      if (canvasRef.current) {
        drawWaveform(canvasRef.current, isFocused);
      }
    });

    resizeObserver.observe(canvasRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [iData, qData, isFocused, zoomLevel, panOffset]);

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

  const iqDataMemo = useMemo<IQDataPoint[]>(() => {
    return iData.map((i, idx) => ({ i, q: qData[idx] }));
  }, [iData, qData]);

  const renderWaveform = () => (
    <div className="w-full h-full relative">
      {isFocused && viewMode === '3d' ? (
        <>
          <Canvas
            camera={{ position: [15, 10, 15], fov: 50 }}
            style={{ background: '#0A0A0A' }}
          >
            <TimeDomain3D iqData={iqDataMemo} onHoverChange={setHover3D} />
          </Canvas>
          {hover3D && (
            <div 
              style={{ 
                position: 'fixed', 
                top: hover3D.y + 15, 
                left: hover3D.x + 15, 
                pointerEvents: 'none', 
                zIndex: 9999 
              }}
              className="bg-[#050505] border border-[#B200FF] shadow-[0_0_10px_rgba(178,0,255,0.2)] p-2 font-mono text-[10px] text-gray-300 flex flex-col space-y-1 min-w-[130px]"
            >
              <div className="text-[#B200FF] font-bold border-b border-[#222] pb-1 mb-1">SAMPLE INDEX: {hover3D.index}</div>
              <div>I (In-Phase): {hover3D.iVal.toFixed(4)}</div>
              <div>Q (Quadrature): {hover3D.qVal.toFixed(4)}</div>
              <div>Mag: {Math.sqrt(hover3D.iVal ** 2 + hover3D.qVal ** 2).toFixed(4)}</div>
            </div>
          )}
          <div className="absolute bottom-4 left-4 z-10 bg-[#0A0A0A] border border-[#222] p-3 flex flex-col space-y-2 font-mono text-[10px] text-gray-400 shadow-xl pointer-events-none">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[#00E5FF]"></div>
              <span>IN-PHASE (I) : XY PLANE</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[#B200FF]"></div>
              <span>QUADRATURE (Q) : XZ PLANE</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-white"></div>
              <span>COMPOSITE TRAJECTORY</span>
            </div>
          </div>
        </>
      ) : (
        <>
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
        </>
      )}
      {isFocused && viewMode === '2d' && (
        <div className="absolute bottom-4 left-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Zoom: {zoomLevel.toFixed(1)}x</div>
          {selectedRange && (
            <div className="text-xs font-mono text-sigma-teal mt-1">
              Selected: {selectedRange.end - selectedRange.start} samples
            </div>
          )}
        </div>
      )}
      {isFocused && viewMode === '2d' && (
        <div className="absolute top-4 right-4 bg-[#0A0A0A] border border-[#222222] px-3 py-2 rounded z-10">
          <div className="text-xs font-mono text-slate-400">Hold Shift + Drag to select time range</div>
        </div>
      )}
      {isFocused && (
        <div className="absolute top-4 left-4 bg-[#0A0A0A] border border-[#222222] rounded-lg overflow-hidden z-10">
          <div className="flex">
            <button
              onClick={() => setViewMode('2d')}
              className={`px-4 py-2 text-xs font-mono transition-colors ${
                viewMode === '2d' 
                  ? 'bg-cyan-500 text-black' 
                  : 'bg-transparent text-gray-400 hover:text-white'
              }`}
            >
              2D
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-4 py-2 text-xs font-mono transition-colors ${
                viewMode === '3d' 
                  ? 'bg-cyan-500 text-black' 
                  : 'bg-transparent text-gray-400 hover:text-white'
              }`}
            >
              3D RIBBON
            </button>
          </div>
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
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900/30 transition-colors duration-300 relative">
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
          setViewMode('2d');
        }} 
        title="TIME DOMAIN WAVEFORM"
      >
        {renderWaveform()}
        {viewMode === '2d' && (
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
        )}
        {viewMode === '3d' && (
          <div className="mt-4 flex gap-6 text-xs font-mono">
            <div className="flex items-center gap-2">
              <div className="w-4 h-1 bg-cyan-500"></div>
              <span className="text-gray-400">I/Q Ribbon in 3D Space</span>
            </div>
            <div className="text-gray-500">Use mouse to rotate • Scroll to zoom</div>
          </div>
        )}
      </FocusMode>
    </>
  );
});

export default WaveformViewer;
