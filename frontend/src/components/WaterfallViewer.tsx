import { useEffect, useRef, useState } from 'react';

interface WaterfallViewerProps {
  data: number[][];
  isLive?: boolean;
}

export default function WaterfallViewer({ data, isLive = false }: WaterfallViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);

  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!data.length || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = dimensions.width;
    canvas.height = dimensions.height;

    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const offscreen = offscreenRef.current;
    const offscreenCtx = offscreen.getContext('2d');
    if (!offscreenCtx) return;

    const rows = data.length;
    const cols = data[0]?.length || 0;

    offscreen.width = cols;
    offscreen.height = rows;

    const allValues = data.flat();
    allValues.sort((a, b) => a - b);
    const p8 = allValues[Math.floor(allValues.length * 0.08)] || 0;
    const p99 = allValues[Math.floor(allValues.length * 0.995)] || 1;

    const createLUT = (): Uint8ClampedArray => {
      const lut = new Uint8ClampedArray(256 * 4);
      for (let i = 0; i < 256; i++) {
        const t = i / 255;
        if (t < 0.2) {
          lut[i * 4] = 15;
          lut[i * 4 + 1] = 23;
          lut[i * 4 + 2] = 42;
        } else if (t < 0.5) {
          const local = (t - 0.2) / 0.3;
          lut[i * 4] = Math.floor(15 + local * 5);
          lut[i * 4 + 1] = Math.floor(23 + local * 161);
          lut[i * 4 + 2] = Math.floor(42 + local * 124);
        } else {
          const local = (t - 0.5) / 0.5;
          lut[i * 4] = Math.floor(20 + local * 235);
          lut[i * 4 + 1] = Math.floor(184 + local * 71);
          lut[i * 4 + 2] = Math.floor(166 + local * 89);
        }
        lut[i * 4 + 3] = 255;
      }
      return lut;
    };

    const lut = createLUT();

    const imageData = offscreenCtx.createImageData(cols, rows);
    const pixels = imageData.data;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const value = data[row][col];
        const normalized = Math.max(0, Math.min(1, (value - p8) / (p99 - p8)));
        const index = Math.floor(normalized * 255);
        const pixelIndex = (row * cols + col) * 4;
        pixels[pixelIndex] = lut[index * 4];
        pixels[pixelIndex + 1] = lut[index * 4 + 1];
        pixels[pixelIndex + 2] = lut[index * 4 + 2];
        pixels[pixelIndex + 3] = 255;
      }
    }

    offscreenCtx.putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, canvas.width, canvas.height);

  }, [data, dimensions]);

  useEffect(() => {
    if (!isLive || !canvasRef.current || !data.length) return;

    const animate = (timestamp: number) => {
      if (timestamp - lastUpdateRef.current < 150) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      lastUpdateRef.current = timestamp;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx || !offscreenRef.current) return;

      const offscreen = offscreenRef.current;
      const offscreenCtx = offscreen.getContext('2d');
      if (!offscreenCtx) return;

      const rows = offscreen.height;
      const cols = offscreen.width;

      const existingData = offscreenCtx.getImageData(0, 0, cols, rows);
      const newImageData = offscreenCtx.createImageData(cols, rows);

      for (let row = 1; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const srcIdx = ((row - 1) * cols + col) * 4;
          const dstIdx = (row * cols + col) * 4;
          newImageData.data[dstIdx] = existingData.data[srcIdx];
          newImageData.data[dstIdx + 1] = existingData.data[srcIdx + 1];
          newImageData.data[dstIdx + 2] = existingData.data[srcIdx + 2];
          newImageData.data[dstIdx + 3] = existingData.data[srcIdx + 3];
        }
      }

      for (let col = 0; col < cols; col++) {
        const variation = Math.sin(timestamp * 0.001 + col * 0.1) * 0.3 + 0.7;
        const baseIdx = (Math.floor(rows * 0.3) * cols + col) * 4;
        const r = existingData.data[baseIdx];
        const g = existingData.data[baseIdx + 1];
        const b = existingData.data[baseIdx + 2];

        const dstIdx = col * 4;
        newImageData.data[dstIdx] = Math.floor(r * variation);
        newImageData.data[dstIdx + 1] = Math.floor(g * variation);
        newImageData.data[dstIdx + 2] = Math.floor(b * variation);
        newImageData.data[dstIdx + 3] = 255;
      }

      offscreenCtx.putImageData(newImageData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, canvas.width, canvas.height);

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isLive, data]);

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO WATERFALL DATA</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal-900 transition-colors duration-300">
      <div className="absolute top-4 left-4 z-10">
        <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">WATERFALL</div>
      </div>
      {isLive && (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
          <div className="w-2 h-2 bg-sigma-teal rounded-full animate-pulse"></div>
          <span className="text-xs font-mono text-sigma-teal uppercase">LIVE</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
}
