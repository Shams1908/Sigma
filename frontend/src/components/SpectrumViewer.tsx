import { useEffect, useRef, useState, useMemo, memo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

interface SpectrumViewerProps {
  data: SpectrumPoint[];
}

interface Spectrum3DProps {
  fftData: number[];
  contrast: number;
  noiseFloor: number;
}

const HISTORY_ROWS = 50;
const DATA_BINS = 256;

const Spectrum3D = memo(function Spectrum3D({ fftData, contrast, noiseFloor }: Spectrum3DProps) {
  const positionsRef = useRef(new Float32Array(DATA_BINS * HISTORY_ROWS * 3));
  const geometryRef = useRef<THREE.BufferGeometry>(null);

  useMemo(() => {
    const positions = positionsRef.current;
    for (let z = 0; z < HISTORY_ROWS; z++) {
      for (let x = 0; x < DATA_BINS; x++) {
        const idx = (z * DATA_BINS + x) * 3;
        positions[idx] = (x / DATA_BINS) * 20 - 10;
        positions[idx + 1] = 0;
        positions[idx + 2] = -(z / HISTORY_ROWS) * 20;
      }
    }
  }, []);

  useEffect(() => {
    if (!fftData || fftData.length === 0) return;

    const positions = positionsRef.current;
    
    for (let z = HISTORY_ROWS - 1; z > 0; z--) {
      for (let x = 0; x < DATA_BINS; x++) {
        const currentIdx = (z * DATA_BINS + x) * 3 + 1;
        const prevIdx = ((z - 1) * DATA_BINS + x) * 3 + 1;
        positions[currentIdx] = positions[prevIdx];
      }
    }

    for (let x = 0; x < DATA_BINS; x++) {
      const dataIdx = Math.floor((x / DATA_BINS) * fftData.length);
      const dbValue = fftData[dataIdx];
      const normalizedY = ((dbValue - noiseFloor) / (contrast - noiseFloor)) * 5;
      const idx = x * 3 + 1;
      positions[idx] = Math.max(0, Math.min(5, normalizedY));
    }

    if (geometryRef.current) {
      geometryRef.current.attributes.position.needsUpdate = true;
    }
  }, [fftData, contrast, noiseFloor]);

  const indices = useMemo(() => {
    const idx: number[] = [];
    for (let z = 0; z < HISTORY_ROWS - 1; z++) {
      for (let x = 0; x < DATA_BINS - 1; x++) {
        const topLeft = z * DATA_BINS + x;
        const topRight = topLeft + 1;
        const bottomLeft = (z + 1) * DATA_BINS + x;
        const bottomRight = bottomLeft + 1;

        idx.push(topLeft, bottomLeft);
        idx.push(bottomLeft, bottomRight);
        idx.push(bottomRight, topRight);
        idx.push(topRight, topLeft);
      }
    }
    return new Uint32Array(idx);
  }, []);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 10, 5]} intensity={1} />
      
      <lineSegments>
        <bufferGeometry ref={geometryRef}>
          <bufferAttribute
            attach="attributes-position"
            count={DATA_BINS * HISTORY_ROWS}
            array={positionsRef.current}
            itemSize={3}
          />
          <bufferAttribute
            attach="index"
            count={indices.length}
            array={indices}
            itemSize={1}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00E5FF" transparent opacity={0.6} blending={THREE.AdditiveBlending} />
      </lineSegments>
      
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={DATA_BINS * HISTORY_ROWS}
            array={positionsRef.current}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial color="#00E5FF" size={0.05} transparent opacity={0.8} />
      </points>
      
      <OrbitControls 
        makeDefault 
        minPolarAngle={0.1} 
        maxPolarAngle={Math.PI / 2} 
        enablePan={false}
      />
      
      <gridHelper args={[30, 30, '#333333', '#111111']} position={[0, 0, -10]} />
    </>
  );
});

export default function SpectrumViewer({ data }: SpectrumViewerProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const spectrumCanvasRef = useRef<HTMLCanvasElement>(null);
  const maxHoldRef = useRef<Float32Array | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [isExpanded, setIsExpanded] = useState(false);
  const [showMaxHold, setShowMaxHold] = useState(false);
  const [showGradient, setShowGradient] = useState(false);
  const [contrast, setContrast] = useState(-20);
  const [noiseFloor, setNoiseFloor] = useState(-100);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');

  const fftData = data.map(d => d.magnitudeDb);

  useEffect(() => {
    if (isExpanded) return;
    
    const canvas = baseCanvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const rect = parent.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, w, h);

    const dataLen = fftData.length;
    if (dataLen === 0) return;

    const minDb = Math.min(...fftData);
    const maxDb = Math.max(...fftData);
    const dbRange = maxDb - minDb || 1;

    ctx.beginPath();
    ctx.strokeStyle = '#14B8A6';
    ctx.lineWidth = 2;

    for (let i = 0; i < w; i++) {
      const dataIdx = Math.floor((i / w) * dataLen);
      const dbValue = fftData[dataIdx];
      let y = h - ((dbValue - minDb) / dbRange) * h;
      y = Math.max(0, Math.min(h, y));
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  }, [data, fftData, isExpanded]);

  useEffect(() => {
    if (!isExpanded || viewMode !== '2d') return;

    const canvas = spectrumCanvasRef.current;
    if (!canvas || !fftData || fftData.length === 0) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const rect = parent.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, w, h);

    const dataLen = fftData.length;

    if (!maxHoldRef.current || maxHoldRef.current.length !== dataLen) {
      maxHoldRef.current = new Float32Array(dataLen).fill(-1000);
    }
    for (let i = 0; i < dataLen; i++) {
      if (fftData[i] > maxHoldRef.current[i]) {
        maxHoldRef.current[i] = fftData[i];
      }
    }

    if (showMaxHold && maxHoldRef.current) {
      ctx.beginPath();
      ctx.strokeStyle = '#B200FF';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      for (let i = 0; i < w; i++) {
        const dataIdx = Math.floor((i / w) * dataLen);
        const dbValue = maxHoldRef.current[dataIdx];
        let y = h - ((dbValue - noiseFloor) / (contrast - noiseFloor)) * h;
        y = Math.max(0, Math.min(h, y));
        if (i === 0) ctx.moveTo(i, y);
        else ctx.lineTo(i, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.strokeStyle = '#00E5FF';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < w; i++) {
      const dataIdx = Math.floor((i / w) * dataLen);
      const dbValue = fftData[dataIdx];
      let y = h - ((dbValue - noiseFloor) / (contrast - noiseFloor)) * h;
      y = Math.max(0, Math.min(h, y));
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }

    if (showGradient) {
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(0, 229, 255, 0.4)');
      grad.addColorStop(1, 'rgba(0, 17, 51, 0.8)');

      ctx.fillStyle = grad;
      ctx.fill();
    }

    ctx.stroke();
  }, [fftData, isExpanded, showMaxHold, showGradient, contrast, noiseFloor, viewMode]);

  const resetMaxHold = () => {
    if (maxHoldRef.current) {
      maxHoldRef.current.fill(-1000);
    }
  };

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO SPECTRUM DATA</p>
      </div>
    );
  }

  if (!isExpanded) {
    return (
      <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-cyan-900/30 transition-colors duration-300">
        <div className="absolute top-4 left-4 z-10">
          <div className="text-cyan-500 font-mono text-xs tracking-wider uppercase">SPECTRUM</div>
        </div>
        <button
          onClick={() => setIsExpanded(true)}
          className="absolute top-4 right-4 z-10 p-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-cyan-500 rounded-lg transition-all"
          title="Expand to Focus Mode"
        >
          <svg className="w-4 h-4 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
        <canvas ref={baseCanvasRef} className="w-full h-full" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-[#050505] font-mono text-[11px] text-gray-300 overflow-hidden">
      
      <div className="absolute inset-0 z-10">
        {viewMode === '2d' ? (
          <canvas ref={spectrumCanvasRef} className="absolute inset-0 block w-full h-full" />
        ) : (
          <Canvas
            camera={{ position: [0, 5, 5], fov: 60 }}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
          >
            <Spectrum3D fftData={fftData} contrast={contrast} noiseFloor={noiseFloor} />
          </Canvas>
        )}
      </div>

      <div className="absolute top-0 left-0 right-0 h-10 bg-[#0A0A0A]/90 border-b border-[#222] backdrop-blur-sm flex items-center justify-between px-4 z-50">
        <div className="font-bold text-white tracking-widest">SPECTRUM ANALYZER - FOCUS MODE</div>
        <button onClick={() => setIsExpanded(false)} className="border border-[#222] bg-black/50 px-3 py-1 hover:bg-[#222] transition-colors text-white">Close (ESC)</button>
      </div>

      <div className="absolute top-14 left-4 w-64 bg-[#0A0A0A]/85 border border-[#222] backdrop-blur-md p-4 flex flex-col space-y-6 z-50 shadow-2xl rounded-sm">
        
        <div className="font-bold text-[#00E5FF] tracking-widest border-b border-[#222] pb-1">- VISUALIZATION</div>
        <div className="flex flex-col space-y-2">
          <button onClick={() => setViewMode('2d')} className={`p-2 border text-left transition-colors ${viewMode === '2d' ? 'border-[#00E5FF] text-[#00E5FF] bg-[#00E5FF]/20' : 'border-[#222] text-gray-400 hover:border-[#444] bg-black/50'}`}>2D CANVAS</button>
          <button onClick={() => setViewMode('3d')} className={`p-2 border text-left transition-colors ${viewMode === '3d' ? 'border-[#00E5FF] text-[#00E5FF] bg-[#00E5FF]/20' : 'border-[#222] text-gray-400 hover:border-[#444] bg-black/50'}`}>3D SURFACE</button>
        </div>

        {viewMode === '2d' && (
          <>
            <div className="font-bold text-[#00E5FF] tracking-widest border-b border-[#222] pb-1 mt-6">- ANALYTICS</div>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={showMaxHold} onChange={e => setShowMaxHold(e.target.checked)} className="accent-[#00E5FF]" /> 
              <span>MAX HOLD TRACE</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={showGradient} onChange={e => setShowGradient(e.target.checked)} className="accent-[#00E5FF]" /> 
              <span>GRADIENT FILL</span>
            </label>
          </>
        )}

        <div className="font-bold text-[#00E5FF] tracking-widest border-b border-[#222] pb-1 mt-6">- PARAMETERS</div>
        <div className="flex flex-col space-y-4">
          <div className="flex flex-col space-y-1">
            <div className="flex justify-between text-gray-400"><span>CONTRAST (MAX)</span><span>{contrast} dB</span></div>
            <input type="range" min="-80" max="0" value={contrast} onChange={e => setContrast(Number(e.target.value))} className="w-full accent-[#00E5FF]" />
          </div>
          <div className="flex flex-col space-y-1">
            <div className="flex justify-between text-gray-400"><span>NOISE FLOOR</span><span>{noiseFloor} dB</span></div>
            <input type="range" min="-140" max="-40" value={noiseFloor} onChange={e => setNoiseFloor(Number(e.target.value))} className="w-full accent-[#00E5FF]" />
          </div>
        </div>
      </div>
      
    </div>
  );
}
