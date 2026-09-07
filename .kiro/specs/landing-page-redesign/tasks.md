# Tasks — Landing Page Redesign

Derived from the reconciled `requirements.md` (R1–R14) and `design.md`.
Every file touched is inside `frontend/`. No task requires changes outside
`frontend/` — each task is pre-flagged with **⛔ OUTSIDE SCOPE** if it would;
none do.

**Execution dependency graph**
```
T1 (tokens + base CSS)  ─┐
T2 (WebThreads files)   ─┤
T3 (useScanReveal hook) ─┤──▶ T5 (LandingView full build) ──▶ T12 (verification)
                          │
T4 (Workstation patch)  ─┘──▶ T12

T6–T11 are sub-tasks of T5 and can be written in parallel within the same
file, but T5 must not be considered done until all sub-tasks pass lint.
```

Tasks T1–T4 have no inter-dependencies and can be executed in parallel.
T5 (and its sub-tasks T6–T11) depends on T1, T2, T3.
T4 is independent of T5 but must be complete before T12.
T12 runs last.

---

## T1 — Design tokens and base styles

**Files:** `frontend/tailwind.config.js`, `frontend/src/index.css`
**Requirement refs:** Design System Constraints, R2, R11, R12

### T1.1 — Tailwind colour tokens
- [ ] Open `frontend/tailwind.config.js`. Inside `theme.extend.colors`, add the
  following entries. Do **not** remove any existing entry (`sigma-purple`,
  `sigma-teal`, `sigma-surface`, `sigma-border` etc. — all used by
  `Workstation.tsx`):
  ```js
  background:     '#0A0E12',
  panel:          '#11171D',
  'panel-raised': '#151C22',
  border:         '#1E262E',
  'signal-cyan':  '#22D3EE',
  'signal-green': '#34D399',
  'signal-amber': '#FBBF24',
  'signal-red':   '#F87171',
  'text-primary': '#E6EDF3',
  'text-muted':   '#8A939D',
  ```
  Note: `text-muted` is `#8A939D` (not `#6B7785`) — the corrected value that
  passes WCAG AA (~5.1:1 on `panel`). See R11.

### T1.2 — Tailwind font families
- [ ] Inside `theme.extend`, add or update `fontFamily`:
  ```js
  fontFamily: {
    sans: ['Inter', 'system-ui', 'sans-serif'],
    mono: ['"JetBrains Mono"', 'monospace'],
  },
  ```
  Verify Inter and JetBrains Mono are available — if not loaded via CDN/import,
  add `@import` rules at the top of `index.css` (Google Fonts or Bunny Fonts).
  Do not install a new npm package for fonts.

### T1.3 — index.css global updates
- [ ] In `frontend/src/index.css`, update `body { background-color: ... }` from
  `#000000` to `#0A0E12`. Leave all other existing rules unchanged.
- [ ] Add `html { scroll-behavior: smooth; }` before the `body` rule (if not
  already present).
- [ ] Update the custom scrollbar rules:
  - Track: `#0A0E12`
  - Thumb: `#1E262E`
  - Thumb hover: `rgba(34, 211, 238, 0.27)` (`#22D3EE` at ~27%)

### T1.4 — Smoke check
- [ ] Run `npm run build` (or `npx vite build`) from `frontend/`. Confirm zero
  Tailwind config errors and zero TypeScript errors attributable to this task.

**AC covered:** AC16 (all files inside `frontend/`), AC17 (tsc passes)

---

## T2 — WebThreads component

**Files:** `frontend/src/components/WebThreads/WebThreads.jsx`,
`frontend/src/components/WebThreads/WebThreads.css`,
`frontend/src/components/WebThreads/WebThreads.d.ts`
**Requirement refs:** R1.1–R1.7

### T2.1 — Directory
- [ ] Create `frontend/src/components/WebThreads/` directory.

### T2.2 — WebThreads.jsx (verbatim source)
- [ ] Write `WebThreads.jsx` containing the **exact** source from the spec:
  - `hexToRgb` helper function — verbatim
  - `FAN_MODE` const — verbatim
  - `vertex` shader string — verbatim, no whitespace changes
  - `fragment` shader string — verbatim, no whitespace changes
  - `ctxMap` WeakMap — verbatim
  - `WebThreads` component with two `useEffect` blocks (setup + props-sync) —
    verbatim
  - Default export — verbatim
- [ ] Do **not** convert to `.tsx`. Do **not** touch GLSL to fix linting.

### T2.3 — WebThreads.css
- [ ] Write `WebThreads.css` containing exactly:
  ```css
  .web-threads-container {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }
  ```

### T2.4 — WebThreads.d.ts (TypeScript shim)
- [ ] Write `WebThreads.d.ts` with the ambient module declaration from R1.4.
  Use `declare module '*/WebThreads'` (wildcard path) so the declaration
  resolves regardless of relative import path:
  ```ts
  declare module '*/WebThreads' {
    import { FC } from 'react';
    export interface WebThreadsProps {
      color1?: string; color2?: string; color3?: string;
      speed?: number; threadCount?: number; frequency?: number;
      spread?: number; taper?: number; position?: number;
      fanMode?: 'center' | 'left' | 'right';
      glow?: number; falloff?: number; thickness?: number;
      brightness?: number; opacity?: number; mirror?: boolean;
      shimmer?: boolean; grain?: boolean; grainIntensity?: number;
      mouseInteraction?: boolean; mouseStrength?: number;
      backgroundColor?: string; lightMode?: boolean; className?: string;
    }
    const WebThreads: FC<WebThreadsProps>;
    export default WebThreads;
  }
  ```

### T2.5 — Verify import resolution
- [ ] After T5 (Hero section) is written, confirm `npx tsc --noEmit` reports
  no errors for the `WebThreads` import. If the wildcard path doesn't resolve,
  try `declare module './WebThreads'` or add the `WebThreads/` directory to
  `tsconfig.json` `paths` — do not modify the shader source.

**AC covered:** AC1 (no layout shift from canvas), AC17 (tsc passes)

---

## T3 — useScanReveal hook

**Files:** `frontend/src/hooks/useScanReveal.ts`
**Requirement refs:** R4B.1–R4B.4

### T3.1 — Directory
- [ ] Create `frontend/src/hooks/` directory.

### T3.2 — Hook implementation
- [ ] Write `frontend/src/hooks/useScanReveal.ts`:
  ```ts
  import { useRef, useState, useEffect } from 'react';

  interface UseScanRevealOptions {
    threshold?: number;
    delay?: number;       // stagger delay in ms
    duration?: number;    // override default 800ms
  }

  export function useScanReveal(options?: UseScanRevealOptions) {
    const ref = useRef<HTMLDivElement>(null);
    const [isRevealed, setIsRevealed] = useState(false);

    useEffect(() => {
      const el = ref.current;
      if (!el) return;

      // Immediately revealed under reduced motion
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        setIsRevealed(true);
        return;
      }

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setTimeout(() => setIsRevealed(true), options?.delay ?? 0);
            observer.unobserve(el);
          }
        },
        { threshold: options?.threshold ?? 0.2 }
      );
      observer.observe(el);
      return () => observer.disconnect();
    }, []); // intentionally empty — options are read once on mount

    const duration = options?.duration ?? 800;
    const style: React.CSSProperties = isRevealed
      ? { clipPath: 'inset(0 0% 0 0)' }
      : {
          clipPath: 'inset(0 100% 0 0)',
          transition: `clip-path ${duration}ms cubic-bezier(0.16, 1, 0.3, 1)`,
        };

    return { ref, isRevealed, style };
  }
  ```

### T3.3 — ScanRevealBlock wrapper component
- [ ] Define `ScanRevealBlock` inline near the top of `LandingView.tsx`
  (not in its own file — keeps the diff self-contained):
  ```tsx
  interface ScanRevealBlockProps {
    children: React.ReactNode;
    delay?: number;
    duration?: number;
    className?: string;
  }

  function ScanRevealBlock({ children, delay, duration, className }: ScanRevealBlockProps) {
    const { ref, isRevealed, style } = useScanReveal({ delay, duration });
    return (
      <div ref={ref} style={{ position: 'relative', ...style }} className={className}>
        {children}
        {/* Radar sweep line — visible during reveal only */}
        <div
          aria-hidden="true"
          style={{
            position: 'absolute', top: 0, bottom: 0,
            right: isRevealed ? '0%' : '100%',
            width: '2px',
            background: '#22D3EE',
            boxShadow: '0 0 8px #22D3EE',
            opacity: isRevealed ? 0 : 1,
            transition: isRevealed
              ? `right ${duration ?? 800}ms cubic-bezier(0.16,1,0.3,1), opacity 50ms ease ${duration ?? 800}ms`
              : 'none',
            pointerEvents: 'none',
          }}
        />
      </div>
    );
  }
  ```
  The sweep line `box-shadow: 0 0 8px #22D3EE` is the **only permitted glow**
  outside the hero (R4B.3) and vanishes after ≤ 900 ms. Under
  `prefers-reduced-motion`, `useScanReveal` sets `isRevealed = true`
  immediately, so the sweep line never becomes visible.

**AC covered:** AC4 (clip-reveal respects reduced-motion), AC7 (glow audit)

---

## T4 — Workstation.tsx handoff patch

**File:** `frontend/src/pages/Workstation.tsx`
**Requirement refs:** R7.6

> ⚠️ This is the only task that touches a file used by the existing dashboard.
> The change is **purely additive** — two import additions and one `useEffect`
> at the top of the component body. No existing logic is modified or removed.
> All existing upload, analysis, and error-handling flows are preserved exactly.
> **No files outside `frontend/` are touched.**

- [ ] Add `useLocation` to the `react-router-dom` import line (already imports
  `useNavigate`):
  ```ts
  import { useNavigate, useLocation } from 'react-router-dom';
  ```
- [ ] After all existing `useState` declarations (before the first handler
  function), add:
  ```ts
  const location = useLocation();
  ```
- [ ] Immediately after the `location` line, add a single `useEffect`:
  ```ts
  useEffect(() => {
    const preloadFile = (location.state as { file?: File } | null)?.file;
    if (preloadFile) handleFileUpload(preloadFile);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  ```
  The empty dependency array is intentional — this runs once on mount to
  consume navigation state. The `eslint-disable` comment prevents a false
  warning about `handleFileUpload` being a dependency.
- [ ] Confirm: navigate to `/workstation` directly (no state) still shows the
  upload drop zone without triggering any upload.

**AC covered:** AC8 (nav CTA → upload section → workstation transition), AC10,
AC11, AC12

---

## T5 — LandingView.tsx full rewrite

**File:** `frontend/src/pages/LandingView.tsx`
**Requirement refs:** R2–R9, R10, R11, R12

Replace the existing 636-line file entirely. Keep the same module signature:
```ts
interface LandingViewProps { onLaunch: () => void; }
export default function LandingView({ onLaunch }: LandingViewProps) { ... }
```
`App.tsx` must not be changed.

Structure the file top-to-bottom:
1. Imports
2. Shared utility: `smoothScrollTo`
3. Shared utility: `useMediaQuery`
4. Sub-task T6: `MagneticButton` component
5. Sub-task T7: `ReticleCursor` component
6. `ScanRevealBlock` (from T3.3)
7. `LandingView` default export containing sub-tasks T8–T11 as inline sections

---

## T6 — Hero: MagneticButton component

**Within:** `LandingView.tsx` (above the main export)
**Requirement refs:** R4.6, R4.8, R10

- [ ] Implement `smoothScrollTo(id: string)`:
  ```ts
  function smoothScrollTo(id: string) {
    document.querySelector(id)?.scrollIntoView({ behavior: 'smooth' });
  }
  ```

- [ ] Implement `useMediaQuery(query: string): boolean`:
  ```ts
  function useMediaQuery(query: string): boolean {
    const [matches, setMatches] = useState(
      () => window.matchMedia(query).matches
    );
    useEffect(() => {
      const mq = window.matchMedia(query);
      const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }, [query]);
    return matches;
  }
  ```

- [ ] Implement `useMagneticHover(ref: RefObject<HTMLElement>): void`:
  - Early-return when `prefers-reduced-motion: reduce` matches (check once on
    hook call, not inside `useEffect`)
  - Track `mousemove` on `ref.current`
  - Compute `dx = e.clientX - centreX`, `dy = e.clientY - centreY`
  - Target offset: `txOx = clamp(dx * 0.15, -7, 7)`, `txOy = clamp(dy * 0.15, -7, 7)`
  - `rAF` loop: `ox += (txOx - ox) * 0.18`, `oy += (toyOy - oy) * 0.18`
  - Apply via `el.style.transform = \`translate(${ox.toFixed(2)}px, ${oy.toFixed(2)}px)\``
  - On `mouseleave`: `txOx = txOy = 0`; cancel `rAF` once `|ox| < 0.1 && |oy| < 0.1`
  - Cleanup: `cancelAnimationFrame` and remove event listeners on unmount

- [ ] Implement `MagneticButton`:
  ```tsx
  interface MagneticButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary';
    children: React.ReactNode;
  }
  function MagneticButton({ variant = 'secondary', children, ...props }: MagneticButtonProps) {
    const ref = useRef<HTMLButtonElement>(null);
    useMagneticHover(ref);
    const base = 'font-mono text-xs uppercase tracking-wider border-radius-[2px] px-6 py-2.5 cursor-pointer transition-colors duration-150';
    const styles = variant === 'primary'
      ? 'bg-[#22D3EE] text-[#0A0E12] border-none hover:bg-[#34D399]'
      : 'bg-transparent border border-[#1E262E] text-[#8A939D] hover:border-[#22D3EE] hover:text-[#E6EDF3]';
    return <button ref={ref} className={`${base} ${styles}`} {...props}>{children}</button>;
  }
  ```
  Note: the primary button's `→` arrow must be wrapped in a `<span>` with
  `style={{ display: 'inline-block', transition: 'transform 150ms ease' }}`
  and `onMouseEnter`/`onMouseLeave` on the button to apply
  `translateX(4px)` to that span only — this is the sole permitted
  `translateX` on the page (R10, hover state not entrance).

**AC covered:** AC2 (magnetic hover reduced-motion), AC3 (cursor reduced-motion)

---

## T7 — Hero: ReticleCursor component

**Within:** `LandingView.tsx` (above the main export)
**Requirement refs:** R4.7, R10, R11

- [ ] Implement `ReticleCursor({ heroRef }: { heroRef: RefObject<HTMLElement> })`:

  **Skip rendering entirely when:**
  - `window.matchMedia('(hover: none)').matches` — touch device
  - `window.matchMedia('(prefers-reduced-motion: reduce)').matches` — reduced motion

  Both checks run once in a `useState` initialiser (not reactive — these
  preferences don't change mid-session in a way that requires reactivity).

  **Position tracking:**
  - `useRef` for `posX`, `posY` (current displayed position)
  - `useRef` for `mouseX`, `mouseY` (raw mouse position)
  - `useRef` for `rafId`
  - `rAF` loop: `posX += (mouseX - posX) * 0.12`, same for Y; apply via
    `containerRef.current.style.transform = \`translate(${posX-16}px, ${posY-16}px)\``
  - `window.addEventListener('mousemove', ...)` in `useEffect`

  **Hero bounds:**
  - `isInHero` state, toggled by `mouseenter`/`mouseleave` on `heroRef.current`
  - `opacity: isInHero ? 1 : 0`, `transition: 'opacity 100ms'`
  - `cursor: none` applied to `heroRef.current.style.cursor` on enter;
    restored on leave

  **SVG crosshair** (32×32 px viewBox, `aria-hidden="true"`):
  ```svg
  <!-- four tick marks, 4px gap at centre -->
  <line x1="16" y1="4"  x2="16" y2="12" stroke="#22D3EE" strokeWidth="1"/>
  <line x1="16" y1="20" x2="16" y2="28" stroke="#22D3EE" strokeWidth="1"/>
  <line x1="4"  y1="16" x2="12" y2="16" stroke="#22D3EE" strokeWidth="1"/>
  <line x1="20" y1="16" x2="28" y2="16" stroke="#22D3EE" strokeWidth="1"/>
  <!-- outer ring -->
  <circle cx="16" cy="16" r="8" fill="none" stroke="#22D3EE"
          strokeWidth="1" opacity="0.4"/>
  ```

  **Container element:**
  - `position: fixed`, `z-index: 9999`, `pointer-events: none`,
    `width: 32px`, `height: 32px`, `top: 0`, `left: 0`
  - `aria-hidden="true"`

  **Cleanup:** cancel `rAF`, remove `mousemove` listener, restore cursor on unmount.

**AC covered:** AC3 (reduced-motion: reticle not rendered), AC11 (aria-hidden)

---

## T8 — Navbar section

**Within:** `LandingView.tsx` (inside main export, above `<main>`)
**Requirement refs:** R3.1–R3.6, R10, R11

- [ ] Render `<nav aria-label="Main navigation">` as a `framer-motion motion.nav`
  with a single `animate={{ opacity: 1 }}` / `initial={{ opacity: 0 }}`
  transition of `400ms` — this is the **only** permitted framer-motion wrapper
  on the page (R10). No children of the nav use `motion.*`.

- [ ] Styles: `position: fixed; top: 0; left: 0; right: 0; z-index: 50;
  height: 64px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 2rem;` with `bg-[#11171D]/85 backdrop-blur-sm border-b border-[#1E262E]`.
  No `border-radius`, no `box-shadow`.

- [ ] **Wordmark**: `<span className="font-mono font-bold tracking-widest text-[#E6EDF3]">SIGMA</span>`

- [ ] **Nav links** (hidden below 768 px via `hidden md:flex`):
  - Each link wraps its text between two `<span>` halves for the centre-outward
    underline. Use CSS classes (defined in a `<style>` block or `index.css`):
    ```css
    .nav-link { position: relative; font-size: 0.875rem; color: #8A939D;
                text-decoration: none; transition: color 150ms ease; }
    .nav-link:hover { color: #E6EDF3; }
    .nav-link-ul-left, .nav-link-ul-right {
      position: absolute; bottom: -2px; height: 2px; background: #8A939D;
      width: 0; transition: width 200ms ease; }
    .nav-link-ul-left  { right: 50%; }
    .nav-link-ul-right { left: 50%; }
    .nav-link:hover .nav-link-ul-left,
    .nav-link:hover .nav-link-ul-right { width: 50%; }
    .nav-link.active { color: #E6EDF3; }
    .nav-link.active .nav-link-ul-left,
    .nav-link.active .nav-link-ul-right { width: 50%; background: #22D3EE; }
    ```
  - Links: `<a href="#features">Features</a>`, `#pipeline`, `#impact`
    (note: nav links do NOT link to `#upload` — that's reserved for the
    Launch button)
  - Scroll-spy: one `IntersectionObserver` per section (`features`, `pipeline`,
    `upload`, `impact`), `threshold: 0.4`, persists (does not disconnect). Sets
    `activeSection` state. Applies `.active` class when section matches.
    Observer set up in a `useEffect`, cleaned up on unmount.

- [ ] **Launch Workstation button** (visible at all widths):
  ```tsx
  <button
    onClick={() => smoothScrollTo('#upload')}
    className="font-mono text-xs uppercase tracking-wider px-4 py-1.5
               border border-[#22D3EE] text-[#22D3EE] rounded-[2px]
               bg-transparent hover:bg-[#22D3EE] hover:text-[#0A0E12]
               transition-colors duration-150"
  >
    Launch Workstation
  </button>
  ```
  No drop shadow, no scale, no pill shape.

**AC covered:** AC8 (nav CTA scrolls to #upload), AC11 (nav landmark), AC13–15

---

## T9 — Hero section

**Within:** `LandingView.tsx`
**Requirement refs:** R4.1–R4.9, R1.5–R1.7

- [ ] Section wrapper:
  ```tsx
  <section
    ref={heroSectionRef}
    style={{ position: 'relative', minHeight: '100dvh', overflow: 'hidden',
             display: 'flex', flexDirection: 'column',
             alignItems: 'center', justifyContent: 'center' }}
  >
  ```
  `overflow: hidden` is non-negotiable — prevents WebThreads canvas from
  causing horizontal scroll (AC1).

- [ ] **WebThreads background layer** (`position: absolute; inset: 0; z-index: 0;
  pointer-events: none; aria-hidden="true" role="none"`):
  - `isNarrow = useMediaQuery('(max-width: 767px)')`
  - `prefersRed = useMediaQuery('(prefers-reduced-motion: reduce)')`
  - When `!isNarrow`: mount `<WebThreads>` with all R1.5 props, but override
    `speed={prefersRed ? 0.01 : 0.15}` and `mouseInteraction={!prefersRed}`
  - When `isNarrow`: render fallback `<div>` with:
    ```css
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 40%,
      rgba(34,211,238,0.07) 0%, rgba(52,211,153,0.04) 50%, transparent 100%);
    ```
    This is the only permitted `radial-gradient` on the page (R1.7).

- [ ] Mount `<ReticleCursor heroRef={heroSectionRef} />` as a sibling to the
  background layer (inside the section, rendered unconditionally — it self-
  suppresses on touch/reduced-motion internally).

- [ ] **Content layer** (`relative z-10, text-center, px-6, max-w-4xl mx-auto,
  flex flex-col items-center gap-6`):

  1. **Eyebrow**: plain `<p className="font-mono text-xs text-[#8A939D] uppercase tracking-[0.2em]">SIGNAL INTELLIGENCE WORKSTATION</p>`

  2. **Headline**:
     ```tsx
     <h1 style={{
       fontFamily: 'Inter, sans-serif', fontWeight: 900,
       fontSize: 'clamp(2.5rem, 8vw, 6rem)', lineHeight: '0.95',
       letterSpacing: '-0.02em',
       background: 'linear-gradient(135deg, #22D3EE 0%, #34D399 100%)',
       WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
       backgroundClip: 'text',
     }}>
       Automated<br/>signal<br/>intelligence<br/>&amp; analysis
     </h1>
     ```
     No animation, no textShadow.

  3. **Supporting copy**: `<p className="font-sans text-base text-[#8A939D] max-w-md">Closed-loop hypothesis validation for unknown RF signals — from raw IQ to demodulated bitstream.</p>`

  4. **CTA row** (`flex gap-4 justify-center flex-wrap sm:flex-row flex-col`):
     - Primary `<MagneticButton variant="primary" onClick={() => smoothScrollTo('#upload')}>Launch Workstation <span className="arrow">→</span></MagneticButton>`
     - Secondary `<MagneticButton aria-disabled="true" title="Documentation coming soon" onClick={(e) => e.preventDefault()}>View Documentation</MagneticButton>`

  5. **Scroll indicator** (`aria-hidden="true"`, below CTA row):
     ```tsx
     <div aria-hidden="true" className="scroll-indicator flex flex-col items-center gap-1 mt-4">
       <span className="font-mono text-xs text-[#8A939D]">SCROLL</span>
       <div style={{ width: '2px', height: '40px', background: '#1E262E' }} />
       <svg width="8" height="6" viewBox="0 0 8 6">
         <polyline points="0,0 4,6 8,0" fill="none" stroke="#8A939D" strokeWidth="1"/>
       </svg>
     </div>
     ```
     CSS `@keyframes pulse-opacity { 0%,100% { opacity:0.4 } 50% { opacity:1 } }`,
     applied via `animation: pulse-opacity 1.8s ease-in-out infinite`.
     **No translateY.**

**AC covered:** AC1 (no layout shift), AC2 (magnetic), AC3 (reticle),
AC9 (hero CTA scrolls to #upload), AC13–15 (responsive)

---

## T10 — Features section

**Within:** `LandingView.tsx` (or extracted to
`frontend/src/components/landing/FeaturesSection.tsx` if file > 500 lines)
**Requirement refs:** R5.1–R5.8, R4B.5

- [ ] `<section id="features" className="py-24 md:py-16 px-6 max-w-7xl mx-auto">`

- [ ] **Section header** wrapped in `<ScanRevealBlock>` (no stagger):
  ```tsx
  <ScanRevealBlock>
    <p className="font-mono text-xs text-[#8A939D] uppercase tracking-widest">// FEATURES</p>
    <h2 className="font-sans font-bold text-[2rem] text-[#E6EDF3] mt-2">
      From raw capture to validated signal
    </h2>
  </ScanRevealBlock>
  ```

- [ ] **Asymmetric grid** — implement as two rows, each a CSS grid:
  ```tsx
  {/* Row 1: card 1 wide (7fr), card 2 narrow (5fr) */}
  <div style={{ display: 'grid', gridTemplateColumns: '7fr 5fr', gap: '1.5rem' }}
       className="max-md:grid-cols-1">
    <FeatureCard ... staggerDelay={0} />
    <FeatureCard ... staggerDelay={100} />
  </div>
  {/* Row 2: card 3 narrow (5fr), card 4 wide (7fr) */}
  <div style={{ display: 'grid', gridTemplateColumns: '5fr 7fr', gap: '1.5rem' }}
       className="max-md:grid-cols-1">
    <FeatureCard ... staggerDelay={200} />
    <FeatureCard ... staggerDelay={300} />
  </div>
  ```
  On `< 768px` both rows collapse to single column.
  On `768px–1023px`: override both grids to `1fr 1fr`.
  **Do not** use `repeat(4, 1fr)` anywhere.

- [ ] **FeatureCard** inner component (inline):
  - Wraps content in `<ScanRevealBlock delay={staggerDelay}>`
  - `bg-[#11171D] border border-[#1E262E] rounded-[4px] p-6`
  - Top border override: `border-top: 2px solid <accent>` via inline style
  - Hover: `border-color: <accent>` at 150ms — CSS only, no JS
  - Content layout: `flex flex-row gap-6` (wide card) / `flex flex-col gap-4`
    (narrow card, mini-visual below text)
  - Category label: `font-mono text-xs text-[#8A939D] tracking-widest`
  - Feature name: `font-sans font-semibold text-[#E6EDF3] text-base mt-1`
  - Body: `font-sans text-sm text-[#8A939D] leading-relaxed mt-2`

- [ ] **Mini-visual SVGs** — four separate inline SVGs, all `aria-hidden="true"`,
  hardcoded static data:

  **Feature 1 — Spectrum Analysis** (accent `#22D3EE`):
  - 120×80 px SVG
  - Two dashed horizontal grid lines at y=20, y=60: `stroke="#1E262E" strokeDasharray="3,3" strokeWidth="0.5"`
  - `<polyline>` of 22 points: noise floor at y≈65 with Gaussian peak centred
    at x=60 (peak y≈15). Points: for x in [5,10,15…120],
    `y = 65 - 50 * Math.exp(-Math.pow(x-60,2)/200)`. Hardcode computed values.
    Stroke `#22D3EE`, `strokeWidth=1.5`, `fill=none`.

  **Feature 2 — Waterfall View** (accent `#34D399`):
  - 120×80 px `<canvas>` drawn once in `useEffect`
  - Same LUT as `WaterfallViewer.tsx`: `t<0.2` → `rgb(15,23,42)`;
    `0.2≤t<0.5` → interpolate to `rgb(20,184,166)`;
    `t≥0.5` → interpolate to `rgb(255,255,166)`
  - Pixel value: `t = (Math.sin(col/8)*0.5+0.5) * (1 - Math.abs(row/40 - 1))`
  - One-time draw on mount. `imageRendering: 'pixelated'`.

  **Feature 3 — Modulation Classification** (accent `#FBBF24`):
  - 120×80 px SVG
  - 4 rows of confidence bars at y = 8, 26, 44, 62; height 12px each
  - Each row: background `<rect fill="#1E262E" x=0 width=120 height=12/>`
  - Foreground `<rect fill="#FBBF24" opacity={[0.9,0.5,0.25,0.1][i]}
    width={[104,13,2,1][i]} height=12/>`
    (proportional to 87%, 11%, 2%, 1%)
  - No text labels

  **Feature 4 — Constellation Mapping** (accent `#22D3EE`):
  - 120×80 px SVG, viewBox `0 0 120 80`
  - Crosshair axes: horizontal line y=40, vertical line x=60,
    `stroke="#1E262E" strokeWidth="0.5"`
  - 4 QPSK clusters at (18,22), (102,22), (18,58), (102,58).
    Each cluster: 8 dots with hardcoded ±4px offsets (no `Math.random()`).
    `r=2 fill="#22D3EE" fillOpacity=0.6`

**AC covered:** AC7 (no glow in features), AC13–15 (responsive grid)

---

## T11 — Pipeline, Upload, Impact sections and Footer

**Within:** `LandingView.tsx` (or extracted components under
`frontend/src/components/landing/`)
**Requirement refs:** R6, R7, R8, R9, R4B.5

### T11a — Pipeline section

- [ ] `<section id="pipeline" className="py-24 md:py-16 px-6 max-w-7xl mx-auto">`

- [ ] **Section header** in `<ScanRevealBlock>` (no stagger):
  ```tsx
  <p className="font-mono text-xs text-[#8A939D] uppercase tracking-widest">// PIPELINE</p>
  <h2 className="font-sans font-bold text-[2rem] text-[#E6EDF3] mt-2">Six stages, fully traced</h2>
  ```

- [ ] **Stage auto-play state machine**:
  ```ts
  const STAGES = ['INGESTION','DSP ANALYSIS','MODULATION','DEMODULATION','FEC','VALIDATION'];
  type StageStatus = 'pending' | 'running' | 'completed';
  const [stageStatuses, setStageStatuses] = useState<StageStatus[]>(Array(6).fill('pending'));
  const [hasPlayed, setHasPlayed] = useState(false);
  const { ref: pipelineRef, isRevealed } = useScanReveal({ threshold: 0.2 });
  ```
  On `isRevealed && !hasPlayed`:
  ```ts
  useEffect(() => {
    if (!isRevealed || hasPlayed) return;
    setHasPlayed(true);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setStageStatuses(Array(6).fill('completed')); return;
    }
    let i = 0;
    const id = setInterval(() => {
      setStageStatuses(prev => {
        const next = [...prev] as StageStatus[];
        if (i > 0) next[i - 1] = 'completed';
        if (i < 6) next[i] = 'running';
        return next;
      });
      i++;
      if (i > 6) { clearInterval(id); setStageStatuses(Array(6).fill('completed')); }
    }, 600);
    return () => clearInterval(id);
  }, [isRevealed, hasPlayed]);
  ```

- [ ] **Timeline layout** — do **not** use the existing `ProcessingChain`
  component (it uses framer-motion `initial={{ scale: 0 }}` — prohibited):
  - Desktop (`≥ 768px`): `flex items-center justify-between` with connector
    `<div>` lines between nodes (`flex-1 h-[2px]`, colour matches upstream
    status: `#34D399` if completed, else `#1E262E`)
  - Mobile (`< 768px`): `flex flex-col items-start gap-0` with vertical
    `2px wide` connector strips
  - Each node: 40×40 px circle; status colours: `pending #1E262E`,
    `running #22D3EE` + CSS `@keyframes pulse-opacity` ring, `completed #34D399`
  - Below each circle: `STAGE 0N` label (`font-mono text-xs text-[#8A939D]`)
    and stage name (`font-sans text-xs text-[#E6EDF3]`)
  - Clip-reveal stagger: each node in its own `<ScanRevealBlock
    delay={i * 150}>` (0, 150, 300, 450, 600, 750 ms)

- [ ] **Parameter chips** in a single `<ScanRevealBlock>`:
  ```tsx
  {['fs','fc','modulation','symbol_rate','fec_scheme','bandwidth'].map(p => (
    <span key={p} className="inline-flex items-center gap-1 px-3 py-1
                              bg-[#151C22] border border-[#1E262E] rounded-[2px]
                              font-mono text-xs text-[#8A939D]">
      <span style={{ color: '#22D3EE' }}>·</span>{p}
    </span>
  ))}
  ```

- [ ] **MVP scope note** in a `<ScanRevealBlock>`:
  ```tsx
  <div className="border border-[#1E262E] rounded-[4px] p-4 bg-[#151C22]"
       style={{ borderLeft: '4px solid #FBBF24' }}>
    <p className="font-mono text-xs tracking-widest" style={{ color: '#FBBF24' }}>MVP SCOPE</p>
    <p className="font-sans text-sm text-[#E6EDF3] mt-1">
      Initial validation targets BPSK and QPSK modulation with convolutional
      FEC codes. Each hypothesis undergoes real Viterbi decoding with syndrome
      checking before acceptance.
    </p>
  </div>
  ```

### T11b — Upload / Get Started section

- [ ] `<section id="upload" className="py-24 md:py-16 px-6 max-w-4xl mx-auto">`
  This `id="upload"` is the scroll target for ALL three CTAs (navbar, hero
  primary, impact final CTA). Verify the id is exactly `upload`.

- [ ] **Section header** in `<ScanRevealBlock>`:
  ```tsx
  <p className="font-mono text-xs text-[#8A939D] uppercase tracking-widest">// GET STARTED</p>
  <h2 className="font-sans font-bold text-[2rem] text-[#E6EDF3] mt-2">
    Load a signal, start analysing
  </h2>
  ```

- [ ] **Drop-zone** — hidden file input + drag-drop area:
  ```tsx
  <input ref={fileInputRef} type="file" accept=".iq,.wav,.raw" className="hidden"
         onChange={handleFileSelect} />
  ```
  Drop-zone div:
  - Idle: `border border-dashed border-[#1E262E] rounded-[4px] py-16 px-8 text-center`
  - Drag-over state: add `border-[#22D3EE] bg-[rgba(34,211,238,0.04)]`
  - No scale animation on drag-over

  Content (idle):
  1. Upward-arrow SVG icon (40×40, `#22D3EE` stroke, 1.5px, `aria-hidden`):
     ```svg
     <circle cx="20" cy="20" r="18" fill="none" stroke="#22D3EE" strokeWidth="1.5"/>
     <line x1="20" y1="28" x2="20" y2="12" stroke="#22D3EE" strokeWidth="1.5"/>
     <polyline points="14,18 20,12 26,18" fill="none" stroke="#22D3EE" strokeWidth="1.5"/>
     ```
  2. `"DROP SIGNAL FILE"` — `font-mono text-sm text-[#E6EDF3] uppercase tracking-wider`
  3. `"Accepts .IQ · .WAV · .RAW"` — `font-mono text-xs text-[#8A939D]`
  4. Browse Files button: transparent, `border border-[#1E262E]`,
     `font-mono text-xs text-[#8A939D] uppercase tracking-wider`,
     `rounded-[2px] px-5 py-2`, hover `border-[#22D3EE] text-[#E6EDF3]`,
     no scale/shadow

  Drag event handlers: `onDragEnter`, `onDragOver` (prevent default),
  `onDragLeave`, `onDrop` — on drop, validate extension (`.iq`, `.wav`, `.raw`)
  then call `navigate('/workstation', { state: { file } })`.

  File select handler (`handleFileSelect`): same navigate call.

- [ ] **OR divider**:
  ```tsx
  <div className="flex items-center gap-4 my-6">
    <hr className="flex-1 border-t border-[#1E262E]" />
    <span className="font-mono text-xs text-[#8A939D]">OR</span>
    <hr className="flex-1 border-t border-[#1E262E]" />
  </div>
  ```

- [ ] **Demo signal button** — navigates to `/workstation` with no state.
  `generateMockSignal()` does **not exist**; do not import or create it:
  ```tsx
  <button
    onClick={() => navigate('/workstation')}
    className="font-mono text-xs text-[#8A939D] uppercase tracking-wider
               hover:text-[#E6EDF3] transition-colors duration-150 bg-transparent
               border-none cursor-pointer"
  >
    Load demonstration signal
  </button>
  ```

- [ ] Wrap the drop-zone + divider + demo button all inside a single
  `<ScanRevealBlock>` so they reveal as a unit.

### T11c — Impact section

- [ ] `<section id="impact" className="py-24 md:py-16 px-6 max-w-7xl mx-auto">`

- [ ] **Section header** in `<ScanRevealBlock>`:
  ```tsx
  <p className="font-mono text-xs text-[#8A939D] uppercase tracking-widest">// IMPACT</p>
  <h2 className="font-sans font-bold text-[2rem] text-[#E6EDF3] mt-2">
    Built for real signal work, not slideshows
  </h2>
  ```

- [ ] **`useCountUp` hook** (inline in `LandingView.tsx`):
  ```ts
  function useCountUp(target: number, isActive: boolean, duration = 1200) {
    const [value, setValue] = useState(0);
    useEffect(() => {
      if (!isActive) return;
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        setValue(target); return;
      }
      const start = performance.now();
      const raf = (now: number) => {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3); // ease-out cubic
        setValue(Math.round(eased * target));
        if (p < 1) requestAnimationFrame(raf);
      };
      requestAnimationFrame(raf);
    }, [isActive, target, duration]);
    return value;
  }
  ```

- [ ] **Stat counter cards** — four cards in `grid-cols-2 lg:grid-cols-4`, each
  in its own `<ScanRevealBlock delay={i*100}>`:

  | i | target | label |
  |---|---|---|
  | 0 | 6 | PIPELINE STAGES |
  | 1 | 256 | PSD FREQUENCY BINS |
  | 2 | 512 | SPECTRUM DATA POINTS |
  | 3 | 1000 | MAX CONSTELLATION POINTS |

  Values are verified against the codebase (see requirements.md R8.3).
  Each card triggers its own `useScanReveal` for `isActive` → passed to
  `useCountUp`.

  Card display:
  ```tsx
  <div className="bg-[#11171D] border border-[#1E262E] rounded-[4px] p-8">
    <span className="font-mono font-bold text-[#E6EDF3]"
          style={{ fontSize: 'clamp(2rem, 5vw, 3.5rem)' }}>
      {countValue}
    </span>
    <p className="font-mono text-xs text-[#8A939D] tracking-widest mt-1">{label}</p>
  </div>
  ```

- [ ] **Core USP panel** in `<ScanRevealBlock>`:
  ```tsx
  <div className="bg-[#11171D] border border-[#1E262E] rounded-[4px] p-8 mt-8"
       style={{ borderTop: '2px solid #34D399' }}>
    <h3 className="font-sans font-bold text-[#E6EDF3] text-xl">
      Closed-Loop Signal Hypothesis Validation
    </h3>
    <p className="font-sans text-sm text-[#E6EDF3] leading-relaxed mt-3">
      SIGMA doesn't predict modulation schemes — it proves them. Every
      hypothesis is physically tested through the complete signal chain:
      synchronization, demodulation, and forward error correction. A
      hypothesis either fully decodes or it does not.
    </p>
  </div>
  ```

- [ ] **Benefits grid** — `grid grid-cols-1 md:grid-cols-2 gap-6 mt-8`:

  | Title | Body |
  |---|---|
  | Proof Over Confidence | Systems output "82% QPSK" without verification. SIGMA attempts actual demodulation and Viterbi FEC decoding. If syndrome checks pass, the hypothesis is proven. |
  | Explainable Evidence | Every validated signal carries a complete evidence trail: which sync method succeeded, what demodulator configuration worked, which FEC parameters decoded cleanly. |
  | Automation at Scale | Process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically. |
  | Reduced False Positives | By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. Ambiguity is eliminated. |

  Each card: `bg-[#11171D] border border-[#1E262E] rounded-[4px] p-6` in
  `<ScanRevealBlock delay={i*100}>`.
  Title: `font-sans font-semibold text-[#34D399]`.
  Body: `font-sans text-sm text-[#8A939D] leading-relaxed mt-2`.
  Hover: `border-color: #34D399` via CSS only, no translateY.

- [ ] **Final CTA** (centred, `mt-12`):
  ```tsx
  <MagneticButton variant="primary" onClick={() => smoothScrollTo('#upload')}>
    Launch Workstation <span className="arrow">→</span>
  </MagneticButton>
  ```

### T11d — Footer

- [ ] Plain `<footer>` — no `<ScanRevealBlock>`, no framer-motion:
  ```tsx
  <footer className="border-t border-[#1E262E] py-6 px-6">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <span className="font-mono text-xs text-[#8A939D]">SIGMA</span>
      <span className="font-mono text-xs text-[#8A939D]">© 2026 SIGMA Project</span>
    </div>
  </footer>
  ```

**AC covered:** AC5 (pipeline reduced-motion), AC6 (counters reduced-motion),
AC8 (upload CTA chain), AC10 (file drop → workstation), AC11 (demo link),
AC13–15 (responsive)

---

## T12 — Final verification pass

**No new files.** Run after all other tasks are complete.

Work through the AC table from requirements.md R13 in order.

### Build check
- [ ] Run `npm run build` from `frontend/`. Zero TypeScript errors. Zero Vite
  bundler errors. Note: GLSL syntax inside template literal strings will not
  be type-checked — that is expected.
- [ ] Run `git diff --name-only`. Confirm every changed file path starts with
  `frontend/`. **AC16.**

### Reduced-motion audit
- [ ] In Chrome DevTools → Rendering → "Emulate CSS media: prefers-reduced-motion:
  reduce". Verify:
  - [ ] WebThreads is near-static (speed ~0.01), no mouse interaction. **AC2** (partial)
  - [ ] Hero reticle does not render. **AC3.**
  - [ ] Magnetic button hover: buttons do not move. **AC2.**
  - [ ] Clip-reveal: sections appear instantly (200ms, no sweep line). **AC4.**
  - [ ] Pipeline auto-play: all 6 stages show `completed` immediately. **AC5.**
  - [ ] Stat counters: final values shown immediately, no animation. **AC6.**

### Glow / gradient audit **AC7**
- [ ] Search new/modified files for `box-shadow`, `blur`, `radial-gradient`,
  `linear-gradient`, `drop-shadow`, `text-shadow`:
  - `box-shadow: 0 0 8px #22D3EE` — permitted only in `ScanRevealBlock`
    sweep line, visible ≤ 900 ms
  - `linear-gradient(135deg, #22D3EE, #34D399)` — permitted only in hero `<h1>`
  - `radial-gradient(...)` — permitted only in mobile hero fallback `<div>`
  - All other instances are prohibited — fix any found

### Scroll target chain **AC8, AC9**
- [ ] Click navbar "Launch Workstation" — confirm smooth scroll to `#upload`
  section.
- [ ] Click hero "Launch Workstation →" — confirm smooth scroll to `#upload`.
- [ ] Click impact final CTA — confirm smooth scroll to `#upload`.

### Upload flow **AC10, AC11, AC12**
- [ ] Drop a `.iq` file onto the landing drop-zone. Confirm `navigate` fires
  to `/workstation` with `{ state: { file } }`. Confirm Workstation auto-
  starts `handleFileUpload`. **AC10.**
- [ ] Click "Load demonstration signal". Confirm navigation to `/workstation`
  with no state. Confirm Workstation shows standard empty state. **AC11.**
- [ ] Navigate directly to `/workstation` (no state). Confirm upload drop
  zone works as before. **AC12.**

### Responsive checks **AC13, AC14, AC15**
- [ ] DevTools device emulation at 375px:
  - [ ] `<h1>` does not overflow or cause horizontal scroll. **AC13.**
  - [ ] Feature grid: single column.
  - [ ] Pipeline: vertical stack.
  - [ ] Nav: wordmark + Launch button only (no links).
- [ ] At 768px:
  - [ ] Feature grid: two equal columns. **AC14.**
  - [ ] Pipeline: horizontal rail.
  - [ ] Nav links visible.
- [ ] At 1440px:
  - [ ] Feature grid: asymmetric 7fr/5fr rows. **AC15.**
  - [ ] Nav + hero + all sections render without overflow.

### Layout shift check **AC1**
- [ ] Confirm `overflow: hidden` is present on the hero `<section>`.
- [ ] Resize from 320px to 1920px — no horizontal scrollbar appears at any
  width.
- [ ] Optional: run Lighthouse on localhost; CLS score should be 0.

### TypeScript / console **AC17, AC18**
- [ ] `npx tsc --noEmit` from `frontend/`. Zero errors. **AC17.**
- [ ] Open browser console on fresh page load. Zero `console.error` calls. **AC18.**

---

## Files changed — complete list

| File | Task | Action |
|---|---|---|
| `frontend/tailwind.config.js` | T1 | Modify — add tokens, font families |
| `frontend/src/index.css` | T1 | Modify — bg colour, scroll-behaviour, scrollbar |
| `frontend/src/hooks/useScanReveal.ts` | T3 | Create |
| `frontend/src/components/WebThreads/WebThreads.jsx` | T2 | Create — verbatim source |
| `frontend/src/components/WebThreads/WebThreads.css` | T2 | Create |
| `frontend/src/components/WebThreads/WebThreads.d.ts` | T2 | Create — TS shim |
| `frontend/src/pages/LandingView.tsx` | T5–T11 | Replace entirely |
| `frontend/src/pages/Workstation.tsx` | T4 | Modify — add `useLocation` + file preload effect |

**Total: 8 files. All inside `frontend/`.
No file outside `frontend/` is created, modified, or deleted.**
