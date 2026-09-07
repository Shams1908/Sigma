// LandingView.tsx — full replacement per landing-page-redesign spec
// Tasks T5 (shell), T6 (MagneticButton), T7 (ReticleCursor),
//       T8 (Navbar), T9 (Hero), T10 (Features),
//       T11a (Pipeline), T11b (Upload), T11c (Impact), T11d (Footer)

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  RefObject,
} from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useScanReveal } from '../hooks/useScanReveal';
import WebThreads from '../components/WebThreads/WebThreads';

// ─────────────────────────────────────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────────────────────────────────────
interface LandingViewProps {
  onLaunch: () => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// T5 — Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

function smoothScrollTo(id: string) {
  document.querySelector(id)?.scrollIntoView({ behavior: 'smooth' });
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [query]);
  return matches;
}

// ─────────────────────────────────────────────────────────────────────────────
// T3 — ScanRevealBlock (defined here alongside its consumers)
// ─────────────────────────────────────────────────────────────────────────────

interface ScanRevealBlockProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}

function ScanRevealBlock({ children, delay, duration = 800, className }: ScanRevealBlockProps) {
  const { ref, isRevealed, style } = useScanReveal({ delay, duration });
  const prefersReduced = useMediaQuery('(prefers-reduced-motion: reduce)');

  return (
    <div
      ref={ref}
      style={{ position: 'relative', overflow: 'hidden', ...style }}
      className={className}
    >
      {children}
      {/* Radar sweep line — only shown when not reduced-motion */}
      {!prefersReduced && (
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            right: isRevealed ? '0%' : '100%',
            width: '2px',
            background: '#22D3EE',
            boxShadow: isRevealed ? 'none' : '0 0 8px #22D3EE',
            opacity: isRevealed ? 0 : 1,
            transition: isRevealed
              ? `right ${duration}ms cubic-bezier(0.16,1,0.3,1), opacity 50ms ease ${duration}ms`
              : 'none',
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T6 — useMagneticHover hook
// ─────────────────────────────────────────────────────────────────────────────

function useMagneticHover(ref: RefObject<HTMLElement>) {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const el = ref.current;
    if (!el) return;

    let rafId = 0;
    let ox = 0, oy = 0;
    let txOx = 0, txOy = 0;

    const clamp = (v: number, lo: number, hi: number) =>
      Math.max(lo, Math.min(hi, v));

    const onMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      txOx = clamp((e.clientX - cx) * 0.15, -7, 7);
      txOy = clamp((e.clientY - cy) * 0.15, -7, 7);
    };
    const onLeave = () => { txOx = 0; txOy = 0; };

    const loop = () => {
      ox += (txOx - ox) * 0.18;
      oy += (txOy - oy) * 0.18;
      el.style.transform = `translate(${ox.toFixed(2)}px, ${oy.toFixed(2)}px)`;
      if (
        Math.abs(ox) > 0.05 || Math.abs(oy) > 0.05 ||
        Math.abs(txOx) > 0.05 || Math.abs(txOy) > 0.05
      ) {
        rafId = requestAnimationFrame(loop);
      } else {
        rafId = 0;
        el.style.transform = 'translate(0px, 0px)';
      }
    };
    const onEnter = () => {
      if (rafId === 0) rafId = requestAnimationFrame(loop);
    };

    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseenter', onEnter);
    el.addEventListener('mouseleave', onLeave);
    return () => {
      cancelAnimationFrame(rafId);
      el.removeEventListener('mousemove', onMove);
      el.removeEventListener('mouseenter', onEnter);
      el.removeEventListener('mouseleave', onLeave);
    };
  }, [ref]);
}

// ─────────────────────────────────────────────────────────────────────────────
// T6 — MagneticButton
// ─────────────────────────────────────────────────────────────────────────────

interface MagneticButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
}

function MagneticButton({ variant = 'secondary', children, style, ...props }: MagneticButtonProps) {
  const ref = useRef<HTMLButtonElement>(null);
  useMagneticHover(ref as RefObject<HTMLElement>);

  const base: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    borderRadius: '2px',
    padding: '10px 24px',
    cursor: 'pointer',
    transition: 'background-color 150ms ease, color 150ms ease, border-color 150ms ease',
    border: 'none',
    outline: 'none',
  };

  const variantStyle: React.CSSProperties =
    variant === 'primary'
      ? { backgroundColor: '#22D3EE', color: '#0A0E12', border: 'none' }
      : {
          backgroundColor: 'transparent',
          border: '1px solid #1E262E',
          color: '#8A939D',
        };

  return (
    <button
      ref={ref}
      style={{ ...base, ...variantStyle, ...style }}
      onMouseEnter={(e) => {
        const btn = e.currentTarget;
        if (variant === 'primary') {
          btn.style.backgroundColor = '#34D399';
        } else {
          btn.style.borderColor = '#22D3EE';
          btn.style.color = '#E6EDF3';
        }
      }}
      onMouseLeave={(e) => {
        const btn = e.currentTarget;
        if (variant === 'primary') {
          btn.style.backgroundColor = '#22D3EE';
        } else {
          btn.style.borderColor = '#1E262E';
          btn.style.color = '#8A939D';
        }
      }}
      {...props}
    >
      {children}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7 — ReticleCursor
// ─────────────────────────────────────────────────────────────────────────────

function ReticleCursor({ heroRef }: { heroRef: RefObject<HTMLElement> }) {
  const isTouch = typeof window !== 'undefined' &&
    (window.matchMedia('(hover: none)').matches || 'ontouchstart' in window);
  const prefersReduced = typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const containerRef = useRef<HTMLDivElement>(null);
  const [isInHero, setIsInHero] = useState(false);

  useEffect(() => {
    if (isTouch || prefersReduced) return;

    let posX = 0, posY = 0;
    let mouseX = 0, mouseY = 0;
    let rafId = 0;

    const loop = () => {
      posX += (mouseX - posX) * 0.12;
      posY += (mouseY - posY) * 0.12;
      if (containerRef.current) {
        containerRef.current.style.transform =
          `translate(${(posX - 16).toFixed(1)}px, ${(posY - 16).toFixed(1)}px)`;
      }
      rafId = requestAnimationFrame(loop);
    };

    const onMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    window.addEventListener('mousemove', onMouseMove);
    rafId = requestAnimationFrame(loop);

    const hero = heroRef.current;
    const onEnter = () => setIsInHero(true);
    const onLeave = () => setIsInHero(false);
    hero?.addEventListener('mouseenter', onEnter);
    hero?.addEventListener('mouseleave', onLeave);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('mousemove', onMouseMove);
      hero?.removeEventListener('mouseenter', onEnter);
      hero?.removeEventListener('mouseleave', onLeave);
    };
  }, [heroRef, isTouch, prefersReduced]);

  // Apply cursor: none to hero when reticle is active
  useEffect(() => {
    const hero = heroRef.current;
    if (!hero || isTouch || prefersReduced) return;
    hero.style.cursor = isInHero ? 'none' : 'auto';
    return () => { hero.style.cursor = 'auto'; };
  }, [isInHero, heroRef, isTouch, prefersReduced]);

  if (isTouch || prefersReduced) return null;

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '32px',
        height: '32px',
        pointerEvents: 'none',
        zIndex: 9999,
        opacity: isInHero ? 1 : 0,
        transition: 'opacity 100ms ease',
      }}
    >
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        {/* Four tick marks with 4px centre void */}
        <line x1="16" y1="4"  x2="16" y2="12" stroke="#22D3EE" strokeWidth="1"/>
        <line x1="16" y1="20" x2="16" y2="28" stroke="#22D3EE" strokeWidth="1"/>
        <line x1="4"  y1="16" x2="12" y2="16" stroke="#22D3EE" strokeWidth="1"/>
        <line x1="20" y1="16" x2="28" y2="16" stroke="#22D3EE" strokeWidth="1"/>
        {/* Outer ring */}
        <circle cx="16" cy="16" r="8" stroke="#22D3EE" strokeWidth="1" opacity="0.4"/>
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8 — Navbar
// ─────────────────────────────────────────────────────────────────────────────

function Navbar() {
  const [activeSection, setActiveSection] = useState('');

  useEffect(() => {
    const sections = ['features', 'pipeline', 'upload', 'impact'];
    const observers: IntersectionObserver[] = [];

    sections.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => { if (entry.isIntersecting) setActiveSection(id); },
        { threshold: 0.4 }
      );
      obs.observe(el);
      observers.push(obs);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, []);

  const navLinks = [
    { id: 'features', label: 'Features' },
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'impact',   label: 'Impact' },
  ];

  return (
    <motion.nav
      aria-label="Main navigation"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        height: '64px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 2rem',
        backgroundColor: 'rgba(17, 23, 29, 0.85)',
        backdropFilter: 'blur(4px)',
        WebkitBackdropFilter: 'blur(4px)',
        borderBottom: '1px solid #1E262E',
      }}
    >
      {/* Wordmark */}
      <span
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontWeight: 700,
          letterSpacing: '0.2em',
          color: '#E6EDF3',
          fontSize: '0.875rem',
        }}
      >
        SIGMA
      </span>

      {/* Nav links — hidden below 768px */}
      <div
        className="hidden md:flex"
        style={{ gap: '2rem', alignItems: 'center' }}
      >
        {navLinks.map(({ id, label }) => (
          <a
            key={id}
            href={`#${id}`}
            onClick={(e) => { e.preventDefault(); smoothScrollTo(`#${id}`); }}
            className={`nav-link${activeSection === id ? ' active' : ''}`}
          >
            <span className="nav-link-ul-left" />
            {label}
            <span className="nav-link-ul-right" />
          </a>
        ))}
      </div>

      {/* Launch Workstation button */}
      <button
        onClick={() => smoothScrollTo('#upload')}
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '0.75rem',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          padding: '6px 16px',
          borderRadius: '2px',
          border: '1px solid #22D3EE',
          color: '#22D3EE',
          backgroundColor: 'transparent',
          cursor: 'pointer',
          transition: 'background-color 150ms ease, color 150ms ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#22D3EE';
          e.currentTarget.style.color = '#0A0E12';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent';
          e.currentTarget.style.color = '#22D3EE';
        }}
      >
        Launch Workstation
      </button>
    </motion.nav>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T10 — Feature mini-visuals (static inline SVGs / canvas)
// ─────────────────────────────────────────────────────────────────────────────

// Spectrum Analysis — polyline with Gaussian peak
function SpectrumMini() {
  // y = 65 - 50 * exp(-((x-60)^2) / 200) sampled at x=5,10,...,120
  const pts: string[] = [];
  for (let x = 5; x <= 120; x += 5) {
    const y = 65 - 50 * Math.exp(-Math.pow(x - 60, 2) / 200);
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return (
    <svg width="120" height="80" viewBox="0 0 120 80" aria-hidden="true"
         style={{ flexShrink: 0 }}>
      <rect width="120" height="80" rx="2" fill="#0A0E12"/>
      <line x1="0" y1="20" x2="120" y2="20" stroke="#1E262E" strokeWidth="0.5" strokeDasharray="3,3"/>
      <line x1="0" y1="60" x2="120" y2="60" stroke="#1E262E" strokeWidth="0.5" strokeDasharray="3,3"/>
      <polyline points={pts.join(' ')} fill="none" stroke="#22D3EE" strokeWidth="1.5"/>
    </svg>
  );
}

// Waterfall View — canvas drawn once
function WaterfallMini() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = 120, H = 80;
    const img = ctx.createImageData(W, H);
    for (let row = 0; row < H; row++) {
      for (let col = 0; col < W; col++) {
        const t = (Math.sin(col / 8) * 0.5 + 0.5) * (1 - Math.abs(row / 40 - 1));
        let r, g, b;
        if (t < 0.2) {
          const l = t / 0.2;
          r = Math.round(15 + l * 5);
          g = Math.round(23 + l * 161);
          b = Math.round(42 + l * 124);
        } else if (t < 0.5) {
          const l = (t - 0.2) / 0.3;
          r = Math.round(20 + l * 235);
          g = Math.round(184 + l * 71);
          b = Math.round(166 + l * 89);
        } else {
          const l = (t - 0.5) / 0.5;
          r = Math.round(255);
          g = Math.round(255 - l * 89);
          b = Math.round(255 - l * 89);
        }
        const i = (row * W + col) * 4;
        img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, []);
  return (
    <canvas
      ref={canvasRef}
      width={120}
      height={80}
      aria-hidden="true"
      style={{ flexShrink: 0, imageRendering: 'pixelated' }}
    />
  );
}

// Modulation Classification — confidence bars
function ClassificationMini() {
  const bars = [
    { width: 104, opacity: 0.9 },  // QPSK 87%
    { width: 13,  opacity: 0.5 },  // BPSK 11%
    { width: 2,   opacity: 0.25 }, // 8PSK 2%
    { width: 1,   opacity: 0.1 },  // QAM16 1%
  ];
  return (
    <svg width="120" height="80" viewBox="0 0 120 80" aria-hidden="true"
         style={{ flexShrink: 0 }}>
      <rect width="120" height="80" rx="2" fill="#0A0E12"/>
      {bars.map((b, i) => (
        <g key={i} transform={`translate(0, ${8 + i * 18})`}>
          <rect x="0" y="0" width="120" height="12" fill="#1E262E" rx="1"/>
          <rect x="0" y="0" width={b.width} height="12" fill="#FBBF24"
                fillOpacity={b.opacity} rx="1"/>
        </g>
      ))}
    </svg>
  );
}

// Constellation Mapping — QPSK dot clusters
function ConstellationMini() {
  // Four clusters at (18,22), (102,22), (18,58), (102,58)
  // Each cluster: 8 dots with hardcoded offsets
  const offsets = [
    [-3,-2], [2,-3], [-1,3], [3,2], [0,-4], [4,0], [-2,4], [1,1],
  ];
  const centres = [[18,22],[102,22],[18,58],[102,58]] as [number,number][];
  return (
    <svg width="120" height="80" viewBox="0 0 120 80" aria-hidden="true"
         style={{ flexShrink: 0 }}>
      <rect width="120" height="80" rx="2" fill="#0A0E12"/>
      {/* Crosshair axes */}
      <line x1="0" y1="40" x2="120" y2="40" stroke="#1E262E" strokeWidth="0.5"/>
      <line x1="60" y1="0" x2="60"  y2="80" stroke="#1E262E" strokeWidth="0.5"/>
      {centres.map(([cx, cy], ci) =>
        offsets.map(([dx, dy], di) => (
          <circle key={`${ci}-${di}`}
            cx={cx + dx} cy={cy + dy} r="2"
            fill="#22D3EE" fillOpacity="0.6"/>
        ))
      )}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T10 — Features section
// ─────────────────────────────────────────────────────────────────────────────

interface FeatureCardProps {
  accent: string;
  label: string;
  name: string;
  body: string;
  mini: React.ReactNode;
  delay: number;
  wide?: boolean;
}

function FeatureCard({ accent, label, name, body, mini, delay, wide }: FeatureCardProps) {
  const [hovered, setHovered] = useState(false);
  return (
    <ScanRevealBlock delay={delay} duration={700}>
      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          backgroundColor: '#11171D',
          border: `1px solid ${hovered ? accent : '#1E262E'}`,
          borderTop: `2px solid ${accent}`,
          borderRadius: '4px',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: wide ? 'row' : 'column',
          gap: '1.5rem',
          alignItems: wide ? 'center' : 'flex-start',
          transition: 'border-color 150ms ease',
          height: '100%',
        }}
      >
        <div style={{ flex: 1 }}>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.65rem',
                      color: '#8A939D', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            {label}
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                      color: '#E6EDF3', fontSize: '1rem', marginTop: '4px' }}>
            {name}
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem',
                      color: '#8A939D', lineHeight: 1.6, marginTop: '8px' }}>
            {body}
          </p>
        </div>
        <div style={{ flexShrink: 0 }}>{mini}</div>
      </div>
    </ScanRevealBlock>
  );
}

function FeaturesSection() {
  return (
    <section
      id="features"
      style={{ padding: '6rem 1.5rem', maxWidth: '80rem', margin: '0 auto' }}
    >
      <ScanRevealBlock>
        <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                    color: '#8A939D', textTransform: 'uppercase', letterSpacing: '0.2em' }}>
          // FEATURES
        </p>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginTop: '0.5rem',
                     marginBottom: '3rem' }}>
          From raw capture to validated signal
        </h2>
      </ScanRevealBlock>

      {/* Row 1: wide (7fr) | narrow (5fr) */}
      <div style={{ display: 'grid', gridTemplateColumns: '7fr 5fr',
                    gap: '1.5rem', marginBottom: '1.5rem' }}
           className="max-md:!grid-cols-1">
        <FeatureCard accent="#22D3EE" label="Spectrum Analysis"
          name="Frequency-domain decomposition"
          body="FFT-derived power spectral density with peak detection and bandwidth estimation across the full capture."
          mini={<SpectrumMini />} delay={0} wide />
        <FeatureCard accent="#34D399" label="Waterfall View"
          name="Time-frequency intensity map"
          body="Scrolling spectrogram rendered row-by-row, revealing signal persistence, drift, and spectral occupancy over time."
          mini={<WaterfallMini />} delay={100} />
      </div>

      {/* Row 2: narrow (5fr) | wide (7fr) */}
      <div style={{ display: 'grid', gridTemplateColumns: '5fr 7fr', gap: '1.5rem' }}
           className="max-md:!grid-cols-1">
        <FeatureCard accent="#FBBF24" label="Modulation Classification"
          name="ML-ranked hypothesis candidates"
          body="Convolutional classifier ranks modulation schemes by confidence. Each candidate undergoes physical demodulation and FEC verification before acceptance."
          mini={<ClassificationMini />} delay={200} />
        <FeatureCard accent="#22D3EE" label="Constellation Mapping"
          name="IQ scatter diagram"
          body="Phase-space plot of demodulated symbols. Tight clusters indicate successful synchronization and low EVM; scatter indicates sync failure."
          mini={<ConstellationMini />} delay={300} wide />
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11a — Pipeline section
// ─────────────────────────────────────────────────────────────────────────────

type StageStatus = 'pending' | 'running' | 'completed';

const PIPELINE_STAGES = [
  { key: 'INGESTION',    label: 'STAGE 01' },
  { key: 'DSP ANALYSIS', label: 'STAGE 02' },
  { key: 'MODULATION',   label: 'STAGE 03' },
  { key: 'DEMODULATION', label: 'STAGE 04' },
  { key: 'FEC',          label: 'STAGE 05' },
  { key: 'VALIDATION',   label: 'STAGE 06' },
];

function stageCircleColor(status: StageStatus): string {
  if (status === 'completed') return '#34D399';
  if (status === 'running')   return '#22D3EE';
  return '#1E262E';
}

function PipelineSection() {
  const [statuses, setStatuses] = useState<StageStatus[]>(
    Array(6).fill('pending') as StageStatus[]
  );
  const [hasPlayed, setHasPlayed] = useState(false);
  const { ref: sectionRef, isRevealed } = useScanReveal({ threshold: 0.2 });
  const prefersReduced = useMediaQuery('(prefers-reduced-motion: reduce)');
  const isMobile = useMediaQuery('(max-width: 767px)');

  useEffect(() => {
    if (!isRevealed || hasPlayed) return;
    setHasPlayed(true);
    if (prefersReduced) {
      setStatuses(Array(6).fill('completed') as StageStatus[]);
      return;
    }
    let i = 0;
    const id = setInterval(() => {
      setStatuses((prev) => {
        const next = [...prev] as StageStatus[];
        if (i > 0) next[i - 1] = 'completed';
        if (i < 6) next[i] = 'running';
        return next;
      });
      i++;
      if (i > 6) {
        clearInterval(id);
        setStatuses(Array(6).fill('completed') as StageStatus[]);
      }
    }, 600);
    return () => clearInterval(id);
  }, [isRevealed, hasPlayed, prefersReduced]);

  const CHIP_PARAMS = ['fs', 'fc', 'modulation', 'symbol_rate', 'fec_scheme', 'bandwidth'];

  return (
    <section id="pipeline"
      style={{ padding: '6rem 1.5rem', maxWidth: '80rem', margin: '0 auto' }}>

      {/* Section header */}
      <ScanRevealBlock>
        <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                    color: '#8A939D', textTransform: 'uppercase', letterSpacing: '0.2em' }}>
          // PIPELINE
        </p>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginTop: '0.5rem',
                     marginBottom: '3rem' }}>
          Six stages, fully traced
        </h2>
      </ScanRevealBlock>

      {/* Stage timeline */}
      <div ref={sectionRef as React.RefObject<HTMLDivElement>}
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          alignItems: isMobile ? 'flex-start' : 'flex-start',
          gap: isMobile ? '0' : '0',
          marginBottom: '3rem',
          overflowX: 'auto',
        }}>
        {PIPELINE_STAGES.map((stage, i) => (
          <ScanRevealBlock key={stage.key} delay={i * 150} duration={600}
            style={{ display: 'flex', flex: isMobile ? undefined : 1,
                     flexDirection: isMobile ? 'row' : 'column',
                     alignItems: 'center' } as React.CSSProperties}>
            <div style={{ display: 'flex',
                          flexDirection: isMobile ? 'column' : 'column',
                          alignItems: 'center', flex: 'none' }}>
              {/* Circle */}
              <div
                className={statuses[i] === 'running' ? 'stage-running' : ''}
                style={{
                  width: '40px', height: '40px', borderRadius: '50%',
                  backgroundColor: stageCircleColor(statuses[i]),
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1rem', color: '#0A0E12', fontWeight: 700,
                  border: statuses[i] === 'pending' ? '1px solid #1E262E' : 'none',
                  transition: 'background-color 300ms ease',
                }}
              >
                {statuses[i] === 'completed' ? '✓' :
                 statuses[i] === 'running'   ? '⟳' : '○'}
              </div>
              {/* Labels */}
              <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.6rem',
                          color: '#8A939D', textTransform: 'uppercase',
                          letterSpacing: '0.1em', marginTop: '6px', textAlign: 'center' }}>
                {stage.label}
              </p>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.7rem',
                          color: '#E6EDF3', textAlign: 'center', marginTop: '2px',
                          whiteSpace: 'nowrap' }}>
                {stage.key}
              </p>
            </div>
            {/* Connector */}
            {i < PIPELINE_STAGES.length - 1 && (
              <div style={
                isMobile
                  ? { width: '2px', height: '32px', marginLeft: '19px',
                      backgroundColor: statuses[i] === 'completed' ? '#34D399' : '#1E262E',
                      transition: 'background-color 300ms ease' }
                  : { flex: 1, height: '2px', marginTop: '-28px',
                      backgroundColor: statuses[i] === 'completed' ? '#34D399' : '#1E262E',
                      transition: 'background-color 300ms ease', minWidth: '16px' }
              }/>
            )}
          </ScanRevealBlock>
        ))}
      </div>

      {/* Parameter chips */}
      <ScanRevealBlock>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
          {CHIP_PARAMS.map((p) => (
            <span key={p} style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '4px 12px', backgroundColor: '#151C22',
              border: '1px solid #1E262E', borderRadius: '2px',
              fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem', color: '#8A939D',
            }}>
              <span style={{ color: '#22D3EE' }}>·</span>{p}
            </span>
          ))}
        </div>
      </ScanRevealBlock>

      {/* MVP scope note */}
      <ScanRevealBlock>
        <div style={{
          backgroundColor: '#151C22', border: '1px solid #1E262E',
          borderLeft: '4px solid #FBBF24', borderRadius: '4px', padding: '1rem 1.25rem',
        }}>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.65rem',
                      color: '#FBBF24', textTransform: 'uppercase', letterSpacing: '0.15em',
                      marginBottom: '6px' }}>
            MVP SCOPE
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem', color: '#E6EDF3', lineHeight: 1.6 }}>
            Initial validation targets BPSK and QPSK modulation with convolutional FEC codes.
            Each hypothesis undergoes real Viterbi decoding with syndrome checking before acceptance.
          </p>
        </div>
      </ScanRevealBlock>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11b — Upload / Get Started section
// ─────────────────────────────────────────────────────────────────────────────

function UploadSection() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (['iq', 'wav', 'raw'].includes(ext)) {
      navigate('/workstation', { state: { file } });
    }
  }, [navigate]);

  return (
    <section id="upload"
      style={{ padding: '6rem 1.5rem', maxWidth: '48rem', margin: '0 auto' }}>

      {/* Section header */}
      <ScanRevealBlock>
        <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                    color: '#8A939D', textTransform: 'uppercase', letterSpacing: '0.2em' }}>
          // GET STARTED
        </p>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginTop: '0.5rem',
                     marginBottom: '2rem' }}>
          Load a signal, start analysing
        </h2>
      </ScanRevealBlock>

      {/* Drop-zone + divider + demo link */}
      <ScanRevealBlock duration={900}>
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".iq,.wav,.raw"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />

        {/* Drop-zone */}
        <div
          role="button"
          aria-label="Drop signal file or click to browse"
          tabIndex={0}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragEnter={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
          style={{
            border: `1px dashed ${isDragOver ? '#22D3EE' : '#1E262E'}`,
            borderRadius: '4px',
            padding: '4rem 2rem',
            textAlign: 'center',
            backgroundColor: isDragOver ? 'rgba(34,211,238,0.04)' : '#11171D',
            transition: 'border-color 100ms ease, background-color 100ms ease',
            cursor: 'default',
          }}
        >
          {/* Upload icon */}
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none"
               aria-hidden="true" style={{ margin: '0 auto 1rem' }}>
            <circle cx="20" cy="20" r="18" stroke="#22D3EE" strokeWidth="1.5"/>
            <line x1="20" y1="28" x2="20" y2="12" stroke="#22D3EE" strokeWidth="1.5"/>
            <polyline points="14,18 20,12 26,18" fill="none" stroke="#22D3EE" strokeWidth="1.5"/>
          </svg>

          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.875rem',
                      color: '#E6EDF3', textTransform: 'uppercase', letterSpacing: '0.1em',
                      marginBottom: '0.5rem' }}>
            DROP SIGNAL FILE
          </p>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                      color: '#8A939D', marginBottom: '1.5rem' }}>
            Accepts .IQ · .WAV · .RAW
          </p>

          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
              textTransform: 'uppercase', letterSpacing: '0.1em',
              padding: '8px 20px', borderRadius: '2px',
              border: '1px solid #1E262E', color: '#8A939D',
              backgroundColor: 'transparent', cursor: 'pointer',
              transition: 'border-color 150ms ease, color 150ms ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#22D3EE';
              e.currentTarget.style.color = '#E6EDF3';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#1E262E';
              e.currentTarget.style.color = '#8A939D';
            }}
          >
            Browse Files
          </button>
        </div>

        {/* OR divider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '1.5rem 0' }}>
          <hr style={{ flex: 1, border: 'none', borderTop: '1px solid #1E262E' }}/>
          <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                         color: '#8A939D' }}>OR</span>
          <hr style={{ flex: 1, border: 'none', borderTop: '1px solid #1E262E' }}/>
        </div>

        {/* Demo signal — no mock data, navigates to Workstation directly */}
        <div style={{ textAlign: 'center' }}>
          <button
            onClick={() => navigate('/workstation')}
            style={{
              fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
              textTransform: 'uppercase', letterSpacing: '0.1em',
              color: '#8A939D', background: 'transparent',
              border: 'none', cursor: 'pointer',
              transition: 'color 150ms ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#E6EDF3'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = '#8A939D'; }}
          >
            Load demonstration signal
          </button>
        </div>
      </ScanRevealBlock>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11c — Impact section: useCountUp hook + stat cards + USP + benefits
// ─────────────────────────────────────────────────────────────────────────────

function useCountUp(target: number, isActive: boolean, duration = 1200): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!isActive) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return;
    }
    const start = performance.now();
    let rafId: number;
    const raf = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(Math.round(eased * target));
      if (p < 1) rafId = requestAnimationFrame(raf);
    };
    rafId = requestAnimationFrame(raf);
    return () => cancelAnimationFrame(rafId);
  }, [isActive, target, duration]);
  return value;
}

interface StatCardProps {
  target: number;
  label: string;
  delay: number;
}

function StatCard({ target, label, delay }: StatCardProps) {
  const { ref, isRevealed } = useScanReveal({ delay, threshold: 0.3 });
  const count = useCountUp(target, isRevealed);
  const [hovered, setHovered] = useState(false);

  return (
    <div ref={ref as React.RefObject<HTMLDivElement>}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: '#11171D',
        border: `1px solid ${hovered ? '#22D3EE' : '#1E262E'}`,
        borderRadius: '4px',
        padding: '2rem',
        transition: 'border-color 150ms ease',
      }}>
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontWeight: 700,
        fontSize: 'clamp(2rem, 5vw, 3.5rem)',
        color: '#E6EDF3',
        display: 'block',
      }}>
        {count.toLocaleString()}
      </span>
      <p style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: '0.65rem',
        color: '#8A939D',
        textTransform: 'uppercase',
        letterSpacing: '0.15em',
        marginTop: '4px',
      }}>
        {label}
      </p>
    </div>
  );
}

const STATS = [
  { target: 6,    label: 'PIPELINE STAGES' },
  { target: 256,  label: 'PSD FREQUENCY BINS' },
  { target: 512,  label: 'SPECTRUM DATA POINTS' },
  { target: 1000, label: 'MAX CONSTELLATION POINTS' },
];

const BENEFITS = [
  {
    title: 'Proof Over Confidence',
    body: 'Systems output "82% QPSK" without verification. SIGMA attempts actual demodulation and Viterbi FEC decoding. If syndrome checks pass, the hypothesis is proven.',
  },
  {
    title: 'Explainable Evidence',
    body: 'Every validated signal carries a complete evidence trail: which sync method succeeded, what demodulator configuration worked, which FEC parameters decoded cleanly.',
  },
  {
    title: 'Automation at Scale',
    body: 'Process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically.',
  },
  {
    title: 'Reduced False Positives',
    body: 'By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. Ambiguity is eliminated.',
  },
];

function ImpactSection() {
  const navigate = useNavigate();
  const [benefitHover, setBenefitHover] = useState<number | null>(null);

  return (
    <section id="impact"
      style={{ padding: '6rem 1.5rem', maxWidth: '80rem', margin: '0 auto' }}>

      {/* Section header */}
      <ScanRevealBlock>
        <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                    color: '#8A939D', textTransform: 'uppercase', letterSpacing: '0.2em' }}>
          // IMPACT
        </p>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginTop: '0.5rem',
                     marginBottom: '3rem' }}>
          Built for real signal work, not slideshows
        </h2>
      </ScanRevealBlock>

      {/* Stat counters */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '1.5rem', marginBottom: '2rem' }}
           className="lg:!grid-cols-4">
        {STATS.map((s, i) => (
          <StatCard key={s.label} target={s.target} label={s.label} delay={i * 100} />
        ))}
      </div>

      {/* Core USP panel */}
      <ScanRevealBlock>
        <div style={{
          backgroundColor: '#11171D',
          border: '1px solid #1E262E',
          borderTop: '2px solid #34D399',
          borderRadius: '4px',
          padding: '2rem',
          marginBottom: '1.5rem',
        }}>
          <h3 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                       fontSize: '1.25rem', color: '#E6EDF3', marginBottom: '0.75rem' }}>
            Closed-Loop Signal Hypothesis Validation
          </h3>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem',
                      color: '#E6EDF3', lineHeight: 1.7 }}>
            SIGMA doesn't predict modulation schemes — it proves them. Every hypothesis is
            physically tested through the complete signal chain: synchronization, demodulation,
            and forward error correction. A hypothesis either fully decodes or it does not.
          </p>
        </div>
      </ScanRevealBlock>

      {/* Benefits grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)',
                    gap: '1.5rem', marginBottom: '3rem' }}
           className="md:!grid-cols-2">
        {BENEFITS.map((b, i) => (
          <ScanRevealBlock key={b.title} delay={i * 100} duration={700}>
            <div
              onMouseEnter={() => setBenefitHover(i)}
              onMouseLeave={() => setBenefitHover(null)}
              style={{
                backgroundColor: '#11171D',
                border: `1px solid ${benefitHover === i ? '#34D399' : '#1E262E'}`,
                borderRadius: '4px',
                padding: '1.5rem',
                transition: 'border-color 150ms ease',
              }}
            >
              <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                          color: '#34D399', fontSize: '1rem', marginBottom: '0.5rem' }}>
                {b.title}
              </p>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem',
                          color: '#8A939D', lineHeight: 1.6 }}>
                {b.body}
              </p>
            </div>
          </ScanRevealBlock>
        ))}
      </div>

      {/* Final CTA */}
      <div style={{ textAlign: 'center' }}>
        <MagneticButton
          variant="primary"
          onClick={() => smoothScrollTo('#upload')}
        >
          Launch Workstation&nbsp;
          <span style={{
            display: 'inline-block',
            transition: 'transform 150ms ease',
          }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLSpanElement).style.transform = 'translateX(4px)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLSpanElement).style.transform = 'translateX(0)'; }}
          >→</span>
        </MagneticButton>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11d — Footer
// ─────────────────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid #1E262E',
      padding: '1.5rem 1.5rem',
      backgroundColor: '#0A0E12',
    }}>
      <div style={{
        maxWidth: '80rem',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span style={{ fontFamily: '"JetBrains Mono", monospace',
                       fontSize: '0.75rem', color: '#8A939D' }}>
          SIGMA
        </span>
        <span style={{ fontFamily: '"JetBrains Mono", monospace',
                       fontSize: '0.75rem', color: '#8A939D' }}>
          © 2026 SIGMA Project
        </span>
      </div>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T9 — Hero section
// ─────────────────────────────────────────────────────────────────────────────

function HeroSection() {
  const heroRef = useRef<HTMLElement>(null);
  const primaryBtnRef = useRef<HTMLButtonElement>(null);
  const secondaryBtnRef = useRef<HTMLButtonElement>(null);
  const isNarrow     = useMediaQuery('(max-width: 767px)');
  const prefersRed   = useMediaQuery('(prefers-reduced-motion: reduce)');

  // Arrow span hover handled inline — translateX(4px) on hover only
  const [arrowHovered, setArrowHovered] = useState(false);

  return (
    <section
      ref={heroRef}
      style={{
        position: 'relative',
        minHeight: '100dvh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* WebThreads background layer */}
      <div
        aria-hidden="true"
        role="none"
        style={{ position: 'absolute', inset: 0, zIndex: 0, pointerEvents: 'none' }}
      >
        {!isNarrow ? (
          <WebThreads
            color1="#22D3EE"
            color2="#34D399"
            color3="#E6EDF3"
            speed={prefersRed ? 0.01 : 0.15}
            threadCount={7}
            frequency={4.0}
            spread={0.22}
            taper={0.9}
            position={0.5}
            fanMode="center"
            glow={0.018}
            falloff={0.65}
            thickness={1.2}
            brightness={0.55}
            opacity={0.9}
            mirror={true}
            shimmer={false}
            grain={true}
            grainIntensity={0.04}
            mouseInteraction={!prefersRed}
            mouseStrength={0.25}
          />
        ) : (
          <div
            className="web-threads-fallback"
            style={{ position: 'absolute', inset: 0 }}
          />
        )}
      </div>

      {/* Reticle cursor */}
      <ReticleCursor heroRef={heroRef as RefObject<HTMLElement>} />

      {/* Content layer */}
      <div
        style={{
          position: 'relative',
          zIndex: 10,
          textAlign: 'center',
          padding: '0 1.5rem',
          maxWidth: '64rem',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem',
        }}
      >
        {/* Eyebrow */}
        <p style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '0.75rem',
          textTransform: 'uppercase',
          letterSpacing: '0.2em',
          color: '#8A939D',
        }}>
          Signal Intelligence Workstation
        </p>

        {/* Headline — gradient text fill */}
        <h1 style={{
          fontFamily: 'Inter, sans-serif',
          fontWeight: 900,
          fontSize: 'clamp(2.5rem, 8vw, 6rem)',
          lineHeight: 0.95,
          letterSpacing: '-0.02em',
          background: 'linear-gradient(135deg, #22D3EE 0%, #34D399 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          margin: 0,
        }}>
          Automated<br/>
          signal<br/>
          intelligence<br/>
          &amp; analysis
        </h1>

        {/* Supporting copy */}
        <p style={{
          fontFamily: 'Inter, sans-serif',
          fontSize: '1rem',
          color: '#8A939D',
          maxWidth: '30rem',
          lineHeight: 1.6,
        }}>
          Closed-loop hypothesis validation for unknown RF signals —
          from raw IQ to demodulated bitstream.
        </p>

        {/* CTA row */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>
          <MagneticButton
            ref={primaryBtnRef as React.RefObject<HTMLButtonElement>}
            variant="primary"
            onClick={() => smoothScrollTo('#upload')}
          >
            Launch Workstation&nbsp;
            <span
              style={{
                display: 'inline-block',
                transition: 'transform 150ms ease',
                transform: arrowHovered ? 'translateX(4px)' : 'translateX(0)',
              }}
              onMouseEnter={() => setArrowHovered(true)}
              onMouseLeave={() => setArrowHovered(false)}
            >
              →
            </span>
          </MagneticButton>

          <MagneticButton
            ref={secondaryBtnRef as React.RefObject<HTMLButtonElement>}
            variant="secondary"
            aria-disabled="true"
            title="Documentation coming soon"
            onClick={(e) => e.preventDefault()}
          >
            View Documentation
          </MagneticButton>
        </div>

        {/* Scroll indicator — opacity pulse only, no translateY */}
        <div
          aria-hidden="true"
          className="scroll-indicator"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '4px',
            marginTop: '1rem',
          }}
        >
          <span style={{ fontFamily: '"JetBrains Mono", monospace',
                         fontSize: '0.65rem', color: '#8A939D', letterSpacing: '0.15em' }}>
            SCROLL
          </span>
          <div style={{ width: '2px', height: '40px', backgroundColor: '#1E262E' }}/>
          <svg width="8" height="6" viewBox="0 0 8 6" fill="none">
            <polyline points="0,0 4,6 8,0" stroke="#8A939D" strokeWidth="1" fill="none"/>
          </svg>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Default export — LandingView
// ─────────────────────────────────────────────────────────────────────────────

export default function LandingView({ onLaunch: _onLaunch }: LandingViewProps) {
  return (
    <div style={{ backgroundColor: '#0A0E12', minHeight: '100vh', overflowX: 'hidden' }}>
      <Navbar />
      <main>
        {/* Navbar is 64px fixed — pad top of first section */}
        <div style={{ paddingTop: '64px' }}>
          <HeroSection />
        </div>
        <FeaturesSection />
        <PipelineSection />
        <UploadSection />
        <ImpactSection />
      </main>
      <Footer />
    </div>
  );
}
