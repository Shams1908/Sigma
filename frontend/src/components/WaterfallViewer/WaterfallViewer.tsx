import {
  useEffect,
  useLayoutEffect,
  useRef,
  type CSSProperties,
} from 'react';

type WaterfallViewerProps = {
  waterfall: number[][];
  isLive: boolean;
  className?: string;
  style?: CSSProperties;
};

const SCROLL_MS = 150;
const BG = [0x0a, 0x0e, 0x12] as const;
const CYAN = [0x22, 0xd3, 0xee] as const;
const PEAK = [0xe6, 0xed, 0xf3] as const;

type Rgb = readonly [number, number, number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mixRgb(a: Rgb, b: Rgb, t: number): [number, number, number] {
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
  ];
}

/** Background → cyan (mid) → text-primary. No other hues. */
function colorAt(t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, t));
  if (u < 0.5) return mixRgb(BG, CYAN, u / 0.5);
  return mixRgb(CYAN, PEAK, (u - 0.5) / 0.5);
}

function buildLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const [r, g, b] = colorAt(i / 255);
    const o = i * 3;
    lut[o] = r;
    lut[o + 1] = g;
    lut[o + 2] = b;
  }
  return lut;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[i];
}

function scaleFrom(rows: number[][]): { floor: number; span: number } {
  const values: number[] = [];
  for (const row of rows) {
    for (let i = 0; i < row.length; i += 1) values.push(row[i]);
  }
  values.sort((a, b) => a - b);
  const floor = percentile(values, 0.08);
  const ceil = percentile(values, 0.995);
  const span = Math.max(ceil - floor, 1e-6);
  return { floor, span };
}

function cloneRows(src: number[][]): number[][] {
  const out: number[][] = new Array(src.length);
  for (let i = 0; i < src.length; i += 1) {
    out[i] = src[i].slice();
  }
  return out;
}

/** Newest time at the top so live rows insert at index 0 and older rows fall. */
function displayOrder(waterfall: number[][]): number[][] {
  const copy = cloneRows(waterfall);
  copy.reverse();
  return copy;
}

function varyRow(source: number[], tick: number): number[] {
  const next = new Array<number>(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const n =
      Math.sin(tick * 0.37 + i * 0.051) * 1.15 +
      Math.sin(tick * 0.11 + i * 0.013) * 0.55;
    next[i] = source[i] + n;
  }
  return next;
}

const LUT = buildLut();

export function WaterfallViewer({
  waterfall,
  isLive,
  className = '',
  style,
}: WaterfallViewerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLCanvasElement | null>(null);
  const rowsRef = useRef<number[][]>([]);
  const sourceRef = useRef<number[][]>([]);
  const scaleRef = useRef({ floor: 0, span: 1 });
  const tickRef = useRef(0);
  const paintRef = useRef<() => void>(() => {});

  paintRef.current = () => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    const rows = rowsRef.current;
    if (!canvas || !host || rows.length === 0 || rows[0].length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, Math.floor(host.clientWidth));
    const cssH = Math.max(1, Math.floor(host.clientHeight));
    const pixelW = Math.max(1, Math.floor(cssW * dpr));
    const pixelH = Math.max(1, Math.floor(cssH * dpr));

    if (canvas.width !== pixelW || canvas.height !== pixelH) {
      canvas.width = pixelW;
      canvas.height = pixelH;
    }

    const nRows = rows.length;
    const nCols = rows[0].length;
    let frame = frameRef.current;
    if (!frame) {
      frame = document.createElement('canvas');
      frameRef.current = frame;
    }
    if (frame.width !== nCols || frame.height !== nRows) {
      frame.width = nCols;
      frame.height = nRows;
    }

    const frameCtx = frame.getContext('2d', { alpha: false });
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!frameCtx || !ctx) return;

    const image = frameCtx.createImageData(nCols, nRows);
    const data = image.data;
    const { floor, span } = scaleRef.current;

    let p = 0;
    for (let y = 0; y < nRows; y += 1) {
      const row = rows[y];
      for (let x = 0; x < nCols; x += 1) {
        const t = (row[x] - floor) / span;
        const idx = Math.min(255, Math.max(0, Math.round(t * 255))) * 3;
        data[p] = LUT[idx];
        data[p + 1] = LUT[idx + 1];
        data[p + 2] = LUT[idx + 2];
        data[p + 3] = 255;
        p += 4;
      }
    }

    frameCtx.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(frame, 0, 0, pixelW, pixelH);
  };

  useLayoutEffect(() => {
    sourceRef.current = waterfall;
    rowsRef.current = displayOrder(waterfall);
    scaleRef.current = scaleFrom(waterfall);
    tickRef.current = 0;
    paintRef.current();
  }, [waterfall]);

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const observer = new ResizeObserver(() => {
      paintRef.current();
    });
    observer.observe(host);
    paintRef.current();
    return () => observer.disconnect();
  }, [waterfall]);

  useEffect(() => {
    if (!isLive) return;

    let raf = 0;
    let last = performance.now();

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (now - last < SCROLL_MS) return;
      last = now;

      const source = sourceRef.current;
      const rows = rowsRef.current;
      if (source.length === 0 || rows.length === 0) return;

      tickRef.current += 1;
      const srcRow = source[tickRef.current % source.length];
      rows.pop();
      rows.unshift(varyRow(srcRow, tickRef.current));
      paintRef.current();
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [isLive, waterfall]);

  const cols = waterfall[0]?.length ?? 1;
  const rows = waterfall.length || 1;

  return (
    <section
      className={`relative min-h-0 border border-grid bg-panel ${className}`}
      style={style}
    >
      <h2 className="absolute left-3 top-2 z-10 font-mono text-[10px] font-medium uppercase tracking-[0.22em] text-muted">
        WATERFALL
      </h2>

      <div className="absolute inset-0 min-h-0 px-2 pb-2 pt-7">
        <div
          ref={hostRef}
          className="h-full min-h-0 w-full border border-grid bg-background shadow-none"
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Waterfall spectrogram of frequency versus time"
            width={cols}
            height={rows}
          />
        </div>
      </div>
    </section>
  );
}
