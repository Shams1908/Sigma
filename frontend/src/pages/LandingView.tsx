// LandingView.tsx — full replacement per landing-page-redesign spec
// Tasks T5 (shell), T6 (MagneticButton), T7 (ReticleCursor),
//       T8 (Navbar), T9 (Hero), T10 (Features),
//       T11a (Pipeline), T11b (Upload), T11c (Impact), T11d (Footer)

import React, {
  useState,
  useEffect,
  useRef,
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
            background: '#6d28d9',
            boxShadow: isRevealed ? 'none' : '0 0 8px #6d28d9',
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
      ? { backgroundColor: '#6d28d9', color: '#E6EDF3', border: 'none' }
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
          btn.style.backgroundColor = '#14b8a6';
        } else {
          btn.style.borderColor = '#6d28d9';
          btn.style.color = '#E6EDF3';
        }
      }}
      onMouseLeave={(e) => {
        const btn = e.currentTarget;
        if (variant === 'primary') {
          btn.style.backgroundColor = '#6d28d9';
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
  const isInHeroRef = useRef(false); // ref mirror — readable inside rAF closure

  useEffect(() => {
    if (isTouch || prefersReduced) return;

    let posX = 0, posY = 0;
    let mouseX = 0, mouseY = 0;
    let rafId = 0;

    const loop = () => {
      // Only interpolate and write DOM when cursor is inside the hero.
      // Avoids burning a full rAF tick on every scroll frame outside the hero.
      if (isInHeroRef.current) {
        posX += (mouseX - posX) * 0.12;
        posY += (mouseY - posY) * 0.12;
        if (containerRef.current) {
          containerRef.current.style.transform =
            `translate(${(posX - 16).toFixed(1)}px, ${(posY - 16).toFixed(1)}px)`;
        }
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
    const onEnter = () => { setIsInHero(true);  isInHeroRef.current = true;  };
    const onLeave = () => { setIsInHero(false); isInHeroRef.current = false; };
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
        <line x1="16" y1="4"  x2="16" y2="12" stroke="#6d28d9" strokeWidth="1"/>
        <line x1="16" y1="20" x2="16" y2="28" stroke="#6d28d9" strokeWidth="1"/>
        <line x1="4"  y1="16" x2="12" y2="16" stroke="#6d28d9" strokeWidth="1"/>
        <line x1="20" y1="16" x2="28" y2="16" stroke="#6d28d9" strokeWidth="1"/>
        {/* Outer ring */}
        <circle cx="16" cy="16" r="8" stroke="#6d28d9" strokeWidth="1" opacity="0.4"/>
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8 — Navbar
// ─────────────────────────────────────────────────────────────────────────────

function Navbar() {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState('');

  useEffect(() => {
    const sections = ['features', 'pipeline', 'impact'];
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
        backgroundColor: 'rgba(17, 23, 29, 0.97)',
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
        onClick={() => navigate('/workstation')}
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '0.75rem',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          padding: '6px 16px',
          borderRadius: '2px',
          border: '1px solid #6d28d9',
          color: '#6d28d9',
          backgroundColor: 'transparent',
          cursor: 'pointer',
          transition: 'background-color 150ms ease, color 150ms ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#6d28d9';
          e.currentTarget.style.color = '#E6EDF3';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent';
          e.currentTarget.style.color = '#6d28d9';
        }}
      >
        Launch Workstation
      </button>
    </motion.nav>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Problem & Solution section
// ─────────────────────────────────────────────────────────────────────────────

const TRADITIONAL_ITEMS = [
  'Trial-and-error manual parameter adjustment',
  'Uncertainty chain: one wrong guess breaks everything downstream',
  'AI confidence scores without physical validation',
  'Hours spent on false positives and dead ends',
];

const SIGMA_ITEMS = [
  'Automated Signal Hypothesis Engine',
  'Closed-loop validation: ML proposes, demodulator proves',
  'Forward Error Correction verification against real Viterbi decoding',
  'Explainable evidence trail for every validated hypothesis',
];

function ProblemSolutionSection() {
  const [leftHover, setLeftHover] = useState(false);
  const [rightHover, setRightHover] = useState(false);

  const cardBase: React.CSSProperties = {
    backgroundColor: '#11171D',
    border: '1px solid #1E262E',
    borderRadius: '4px',
    padding: '2rem',
    transition: 'border-color 150ms ease',
  };

  return (
    <section
      id="features"
      style={{ padding: '6rem 1.5rem', maxWidth: '80rem', margin: '0 auto' }}
    >
      <ScanRevealBlock>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginBottom: '0.5rem' }}>
          The Problem &amp; Solution
        </h2>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '1rem',
                    color: '#8A939D', marginBottom: '3rem' }}>
          Traditional RF analysis is broken. SIGMA fixes it.
        </p>
      </ScanRevealBlock>

      {/* Two-column comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr',
                    gap: '1.5rem', marginBottom: '1.5rem' }}
           className="max-md:!grid-cols-1">

        {/* Left — Traditional Analysis */}
        <ScanRevealBlock delay={0} duration={700}>
          <div
            onMouseEnter={() => setLeftHover(true)}
            onMouseLeave={() => setLeftHover(false)}
            style={{
              ...cardBase,
              border: `1px solid ${leftHover ? '#F87171' : '#1E262E'}`,
              borderTop: '2px solid #F87171',
            }}
          >
            <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.7rem',
                        color: '#F87171', textTransform: 'uppercase',
                        letterSpacing: '0.15em', marginBottom: '1rem' }}>
              STATUS: LEGACY
            </p>
            <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                        fontSize: '1.1rem', color: '#F87171', marginBottom: '1.25rem' }}>
              Traditional Analysis
            </p>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0,
                         display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {TRADITIONAL_ITEMS.map((item) => (
                <li key={item} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <span style={{ color: '#F87171', fontWeight: 700,
                                 fontSize: '1rem', flexShrink: 0, marginTop: '1px' }}>×</span>
                  <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem',
                                 color: '#E6EDF3', lineHeight: 1.6 }}>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </ScanRevealBlock>

        {/* Right — SIGMA Approach */}
        <ScanRevealBlock delay={100} duration={700}>
          <div
            onMouseEnter={() => setRightHover(true)}
            onMouseLeave={() => setRightHover(false)}
            style={{
              ...cardBase,
              border: `1px solid ${rightHover ? '#14b8a6' : '#1E262E'}`,
              borderTop: '2px solid #14b8a6',
            }}
          >
            <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.7rem',
                        color: '#14b8a6', textTransform: 'uppercase',
                        letterSpacing: '0.15em', marginBottom: '1rem' }}>
              STATUS: ACTIVE
            </p>
            <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                        fontSize: '1.1rem', color: '#14b8a6', marginBottom: '1.25rem' }}>
              SIGMA Approach
            </p>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0,
                         display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {SIGMA_ITEMS.map((item) => (
                <li key={item} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <span style={{ color: '#14b8a6', fontWeight: 700,
                                 fontSize: '1rem', flexShrink: 0, marginTop: '1px' }}>✓</span>
                  <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem',
                                 color: '#E6EDF3', lineHeight: 1.6 }}>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </ScanRevealBlock>
      </div>

      {/* Uncertainty Chain callout */}
      <ScanRevealBlock delay={200} duration={900}>
        <div style={{
          backgroundColor: '#151C22',
          border: '1px solid #1E262E',
          borderLeft: '4px solid #6d28d9',
          borderRadius: '4px',
          padding: '1.5rem 1.75rem',
        }}>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem',
                      color: '#6d28d9', textTransform: 'uppercase',
                      letterSpacing: '0.15em', marginBottom: '0.75rem' }}>
            The Uncertainty Chain
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.9375rem',
                      color: '#E6EDF3', lineHeight: 1.7 }}>
            In RF signal analysis, parameters are interdependent. If you guess the wrong sample
            rate, your carrier frequency estimate will be off. If the carrier frequency is wrong,
            synchronization fails. If synchronization fails, demodulation produces garbage. SIGMA
            breaks this chain by systematically testing hypotheses and validating each stage with
            physical signal processing, not just statistical confidence.
          </p>
        </div>
      </ScanRevealBlock>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11a — Pipeline section
// ─────────────────────────────────────────────────────────────────────────────

type StageStatus = 'pending' | 'running' | 'completed';

const PIPELINE_STAGES = [
  { key: 'Ingestion',          label: 'STAGE 01', desc: 'Parse .IQ and .wav files, extract metadata, normalize sample format' },
  { key: 'DSP Analysis',       label: 'STAGE 02', desc: 'FFT, PSD, spectrogram, bandwidth, SNR, carrier offset estimation' },
  { key: 'Feature Extraction', label: 'STAGE 03', desc: 'Statistical, spectral, and cyclostationary features for ML classifier' },
  { key: 'ML Classification',  label: 'STAGE 04', desc: 'Modulation recognition with confidence scoring' },
  { key: 'Synchronization',    label: 'STAGE 05', desc: 'Carrier recovery, timing recovery, matched filtering' },
  { key: 'Validation',         label: 'STAGE 06', desc: 'Demodulation attempt, FEC verification, hypothesis ranking' },
];

function stageCircleColor(status: StageStatus): string {
  if (status === 'completed') return '#14b8a6';
  if (status === 'running')   return '#6d28d9';
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

  const CHIP_PARAMS = ['Sampling Frequency', 'Carrier Frequency', 'Modulation Type', 'Symbol Rate', 'FEC Scheme', 'Signal Bandwidth'];

  return (
    <section id="pipeline"
      style={{ padding: '6rem 1.5rem', maxWidth: '80rem', margin: '0 auto' }}>

      {/* Section header */}
      <ScanRevealBlock>
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginBottom: '3rem' }}>
          Signal Processing Pipeline
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
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.65rem',
                          color: '#8A939D', textAlign: 'center', marginTop: '4px',
                          maxWidth: '120px', lineHeight: 1.4 }}>
                {stage.desc}
              </p>
            </div>
            {/* Connector */}
            {i < PIPELINE_STAGES.length - 1 && (
              <div style={
                isMobile
                  ? { width: '2px', height: '32px', marginLeft: '19px',
                      backgroundColor: statuses[i] === 'completed' ? '#14b8a6' : '#1E262E',
                      transition: 'background-color 300ms ease' }
                  : { flex: 1, height: '2px', marginTop: '-28px',
                      backgroundColor: statuses[i] === 'completed' ? '#14b8a6' : '#1E262E',
                      transition: 'background-color 300ms ease', minWidth: '16px' }
              }/>
            )}
          </ScanRevealBlock>
        ))}
      </div>

      {/* Parameter chips */}
      <ScanRevealBlock>
        <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                    fontSize: '1.1rem', color: '#E6EDF3', marginBottom: '1rem' }}>
          Parameter Extraction
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
          {CHIP_PARAMS.map((p) => (
            <span key={p} style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '4px 12px', backgroundColor: '#151C22',
              border: '1px solid #1E262E', borderRadius: '2px',
              fontFamily: '"JetBrains Mono", monospace', fontSize: '0.75rem', color: '#8A939D',
            }}>
              <span style={{ color: '#6d28d9' }}>·</span>{p}
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
            MVP Validation Scope
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.875rem', color: '#E6EDF3', lineHeight: 1.6 }}>
            The minimum viable product focuses on BPSK and QPSK modulation schemes with
            convolutional FEC codes. Each hypothesis undergoes real Viterbi decoding with
            syndrome checking to confirm validity. This provides a concrete foundation for
            expanding to higher-order modulations.
          </p>
        </div>
      </ScanRevealBlock>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T11c — Impact section: USP callout + benefits grid
// ─────────────────────────────────────────────────────────────────────────────

const BENEFITS = [
  {
    title: 'Proof Over Confidence',
    body: 'Traditional systems output percentages like \u201882% QPSK\u201d without verification. SIGMA attempts actual demodulation and FEC decoding. If Viterbi syndrome checks pass, the hypothesis is proven\u2014not guessed.',
  },
  {
    title: 'Explainable Evidence',
    body: 'Every validated signal comes with a complete evidence trail: which synchronization method succeeded, what demodulator configuration worked, and which FEC parameters decoded cleanly.',
  },
  {
    title: 'Automation at Scale',
    body: 'Analysts can process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically, testing combinations that humans might never consider.',
  },
  {
    title: 'Reduced False Positives',
    body: 'By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. A hypothesis either fully decodes or it does not - there is no ambiguity.',
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
        <h2 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                     fontSize: '2rem', color: '#E6EDF3', marginBottom: '3rem' }}>
          Impact &amp; Benefits
        </h2>
      </ScanRevealBlock>

      {/* Core USP panel */}
      <ScanRevealBlock>
        <div style={{
          backgroundColor: '#11171D',
          border: '1px solid #1E262E',
          borderTop: '2px solid #14b8a6',
          borderRadius: '4px',
          padding: '2rem',
          marginBottom: '1.5rem',
        }}>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.7rem',
                      color: '#14b8a6', textTransform: 'uppercase',
                      letterSpacing: '0.15em', marginBottom: '0.75rem' }}>
            CORE USP
          </p>
          <h3 style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700,
                       fontSize: '1.25rem', color: '#E6EDF3', marginBottom: '0.75rem' }}>
            Closed-Loop Signal Hypothesis Validation
          </h3>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.9375rem',
                      color: '#E6EDF3', lineHeight: 1.7 }}>
            SIGMA doesn&apos;t just predict modulation schemes—it proves them. Every hypothesis is
            physically tested through the complete signal chain: synchronization, demodulation,
            and forward error correction.
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
                border: `1px solid ${benefitHover === i ? '#14b8a6' : '#1E262E'}`,
                borderRadius: '4px',
                padding: '1.5rem',
                transition: 'border-color 150ms ease',
              }}
            >
              <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600,
                          color: '#14b8a6', fontSize: '1rem', marginBottom: '0.5rem' }}>
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
          onClick={() => navigate('/workstation')}
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
  const navigate = useNavigate();
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
            color1="#6d28d9"
            color2="#14b8a6"
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
          background: 'linear-gradient(135deg, #6d28d9 0%, #14b8a6 100%)',
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
            onClick={() => navigate('/workstation')}
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
        <ProblemSolutionSection />
        <PipelineSection />
        <ImpactSection />
      </main>
      <Footer />
    </div>
  );
}
