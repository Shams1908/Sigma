# Design — Landing Page Redesign

## 1. Component tree

```
frontend/src/
├── hooks/
│   └── useScanReveal.ts             NEW — shared clip-path reveal
├── components/
│   └── WebThreads/
│       ├── WebThreads.jsx           NEW — verbatim shader component
│       ├── WebThreads.css           NEW — container rule
│       └── WebThreads.d.ts         NEW — ambient TS declaration
├── pages/
│   └── LandingView.tsx              REPLACE — full rewrite of existing 636-line file
└── index.css                        MODIFY — body bg token
tailwind.config.js                   MODIFY — add design tokens + fonts
```

The existing `App.tsx` routing is **not changed**. The route `/ → LandingView`
stays; `LandingView.tsx` is the only page file touched.

No new npm packages. `ogl` and `framer-motion` are already installed.

---

## 2. Design token additions (`tailwind.config.js`)

Extend the existing `theme.extend.colors` object — do not remove `sigma-purple`
or `sigma-teal` (used by `Workstation.tsx`).

```js
// Add inside theme.extend.colors:
background:     '#0A0E12',
panel:          '#11171D',
'panel-raised': '#151C22',
border:         '#1E262E',
'signal-cyan':  '#22D3EE',
'signal-green': '#34D399',
'signal-amber': '#FBBF24',
'signal-red':   '#F87171',
'text-primary': '#E6EDF3',
'text-muted':   '#8A939D',  // #6B7785 fails WCAG AA at small text; corrected to #8A939D (~5.1:1 on panel)
```

Extend `theme.extend.fontFamily`:

```js
sans: ['Inter', 'system-ui', 'sans-serif'],
mono: ['"JetBrains Mono"', 'monospace'],
```

Update `index.css` body: `background-color: #0A0E12`.

---

## 3. `frontend/src/services/mockData.ts` — NOT CREATED

> **Codebase verdict**: `generateMockSignal()` does **not exist** in this
> project. The directories `frontend/src/services/` and `frontend/src/hooks/`
> do not exist. A full grep of the repository confirms zero matches for
> `mockData`, `generateMockSignal`, or `MockSignalData`. This section has
> been removed from the implementation plan.
>
> The "Load demonstration signal" button navigates directly to `/workstation`
> via `navigate('/workstation')` — no mock data is passed. The Workstation
> presents its standard empty state; the user can upload a real file from
> there. Do **not** create `mockData.ts` or `MockSignalData` types.
>
> The `Workstation.tsx` guard for `location.state?.mockData` described in
> section 7 is also removed — only the `location.state?.file` guard (for
> real file handoff) is implemented.

---

## 4. `frontend/src/hooks/useScanReveal.ts`

```ts
function useScanReveal(options?: { threshold?: number }): {
  ref: React.RefObject<HTMLDivElement>;
  isRevealed: boolean;
  style: React.CSSProperties;
}
```

Internally:
- Creates an `IntersectionObserver` with `threshold: options?.threshold ?? 0.2`.
- On intersection, sets `isRevealed = true` and disconnects (fires once).
- Returned `style` object:
  - Before reveal: `{ clipPath: 'inset(0 100% 0 0)', transition: 'clip-path 800ms ease-out' }`
  - After reveal: `{ clipPath: 'inset(0 0% 0 0)' }`
- Checks `window.matchMedia('(prefers-reduced-motion: reduce)')` — if true,
  returns `isRevealed: true` immediately and `style: {}` (no animation).

The scanning cyan line is a sibling `<div>` rendered by the consumer, not
inside the hook — the hook only manages `isRevealed`. A shared wrapper
component `ScanRevealBlock` (defined inline in `LandingView.tsx`) combines
the hook, the content slot, and the trailing cyan-line `<div>` for convenience.

---

## 5. `frontend/src/components/WebThreads/`

### `WebThreads.jsx`
Verbatim copy of the specified component. No changes to shader GLSL or
React logic. Props interface is as given in the spec.

### `WebThreads.css`
```css
.web-threads-container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
```

### `WebThreads.d.ts`
```ts
declare module './WebThreads' {
  import { FC } from 'react';
  interface WebThreadsProps {
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

### Mounting logic (inside `Hero` section of `LandingView`)
```tsx
const isNarrow   = useMediaQuery('(max-width: 767px)');
const prefersRed = useMediaQuery('(prefers-reduced-motion: reduce)');

{!isNarrow && (
  <WebThreads
    color1="#22D3EE" color2="#34D399" color3="#E6EDF3"
    speed={prefersRed ? 0.02 : 0.15}
    mouseInteraction={!prefersRed}
    /* ...rest of props */
  />
)}
{isNarrow && <div className="web-threads-fallback" />}
```

`useMediaQuery` is a tiny inline hook (5 lines, `window.matchMedia` +
`addEventListener('change', …)`) — no library needed.

---

## 6. `frontend/src/pages/LandingView.tsx` — full structure

The file is a **full replacement** of the existing `LandingView.tsx`. It keeps
the same default export signature `({ onLaunch }: LandingViewProps)` so
`App.tsx` needs no change.

### Section layout (top to bottom)

```
<Nav />                  sticky, z-50
<main>
  <HeroSection />        id omitted (full viewport, WebThreads bg)
  <FeaturesSection />    id="features"
  <PipelineSection />    id="pipeline"
  <UploadSection />      id="get-started"
  <ImpactSection />      id="impact"
</main>
<Footer />
```

Each section is a named inner component defined in the same file (not
exported separately) to keep the diff self-contained. If the file grows
past ~500 lines, extract `FeaturesSection`, `PipelineSection`, and
`ImpactSection` into separate files under `frontend/src/components/landing/`.

### Nav component (inline)

```
[SIGMA wordmark]  [Features] [Pipeline] [Impact]  [Launch Workstation ↗]
```

- Wordmark: `<span className="font-mono font-bold tracking-widest text-text-primary">SIGMA</span>`
- Nav links: `<a href="#features">` etc., `text-muted hover:text-primary`; hover underline via CSS `::after` scale from centre: `transform: scaleX(0) → scaleX(1); transform-origin: center`.
- CTA: `border border-signal-cyan text-signal-cyan hover:bg-signal-cyan hover:text-background` + `onClick → smoothScrollTo('#get-started')`.
- `smoothScrollTo(id)` utility: `document.querySelector(id)?.scrollIntoView({ behavior: 'smooth' })`.

### Hero section

```
<section className="relative h-screen overflow-hidden flex flex-col items-center justify-center">
  {/* Absolute background */}
  <div className="absolute inset-0 z-0">
    {!isNarrow && <WebThreads ...themeProps />}
    {isNarrow  && <div className="web-threads-fallback absolute inset-0" />}
  </div>

  {/* Reticle cursor — rendered as fixed SVG, hidden on touch */}
  <ReticleCursor heroRef={heroSectionRef} />

  {/* Content */}
  <div className="relative z-10 text-center px-6 max-w-5xl mx-auto space-y-6">
    <p className="font-mono text-xs text-muted tracking-[0.2em] uppercase">
      Signal Intelligence Workstation
    </p>
    <h1 className="font-sans font-black leading-[0.9]
                   text-[clamp(2.5rem,8vw,6rem)]
                   bg-gradient-to-br from-signal-cyan to-signal-green
                   bg-clip-text text-transparent">
      Automated<br/>signal<br/>intelligence<br/>& analysis
    </h1>
    <p className="text-muted text-base max-w-md mx-auto">
      Upload a raw capture. SIGMA runs automated hypothesis testing,
      closed-loop demodulation, and FEC validation — no manual tuning.
    </p>
    <div className="flex gap-4 justify-center flex-wrap">
      <MagneticButton primary onClick={() => smoothScrollTo('#get-started')}>
        Launch Workstation <ArrowRight />
      </MagneticButton>
      {/* No documentation page exists — render as disabled, no link, no window.open */}
      <MagneticButton
        aria-disabled="true"
        title="Documentation coming soon"
        onClick={(e) => e.preventDefault()}
      >
        View Documentation
      </MagneticButton>
    </div>
  </div>
</section>
```

### MagneticButton component (inline)

Implemented as a plain `<button>` wrapper with `onMouseMove` handler
computing cursor offset relative to button centre, lerped via `rAF`:

```ts
// State: offsetX, offsetY (both start at 0)
// On mousemove: compute dx/dy from button centre, clamp to ±7px
// On mouseleave: lerp back to 0
// Apply as: style={{ transform: `translate(${offsetX}px, ${offsetY}px)` }}
// rAF loop: currentX += 0.12 * (targetX - currentX) each frame
```

No framer-motion `useSpring` for this — `requestAnimationFrame` only, as
required. Arrow nudge: secondary `<span>` with `translate-x-0 group-hover:translate-x-1`
Tailwind class.

### ReticleCursor component (inline)

```tsx
// Only rendered when !prefersTouch && !prefersRed
// position: fixed, z-[9999], pointer-events-none
// SVG: 24×24 crosshair, stroke signal-cyan, opacity 0.8
// useEffect: window mousemove → lerp currentX/Y toward mouse via rAF
// Visible only when mouse is inside heroRef bounding rect
```

### FeaturesSection (inline or extracted)

Asymmetric CSS Grid:
```css
/* Desktop (≥ 768px) */
grid-template-columns: 2fr 1fr;
/* Rows: feature 1 (wide left), feature 2 (narrow right), feature 3 (narrow left), feature 4 (wide right) */
```

Each feature card: `bg-panel border border-[#1E262E] p-6 flex gap-6`.

Mini SVG visuals (all `aria-hidden="true"`, `width="120" height="80"`):

| Feature | SVG content |
|---|---|
| Spectrum Analysis | Polyline simulating FFT: flat noise floor with one sharp peak near centre, x-axis, stroke `signal-cyan` |
| Waterfall View | 5×8 rect grid, fill mapped from `signal-green` (low) → `signal-amber` (mid) → `signal-red` (high), simulating a heatmap |
| Modulation Classification | 4 horizontal bars (BPSK 88%, QPSK 7%, QAM 3%, FSK 1%), labelled left, fill `signal-cyan` proportional |
| Constellation Mapping | 4 clusters of 5–6 dots at ±1,±1 quadrant positions (QPSK pattern), stroke/fill `signal-green` |

### PipelineSection (inline or extracted)

Stage data (matches `ProcessingChain` exactly):
```ts
const STAGES = [
  'INGESTED', 'DSP ANALYSIS', 'MODULATION',
  'DEMODULATION', 'FEC', 'VALIDATION'
];
```

Desktop layout: `flex items-center justify-between` with connecting lines.
Mobile: `flex flex-col` with vertical connectors.

Auto-play on reveal: `useEffect` fires when `isRevealed` from `useScanReveal`
becomes `true`. Increments `activeStage` counter every 600 ms via
`setInterval`; clears when `activeStage >= STAGES.length`. Does **not** use
framer-motion variants for the step sequence — plain `useState` + `setInterval`.

Colour logic per stage index `i`:
- `i < activeStage`: completed — `text-signal-green`, filled circle
- `i === activeStage`: running — `text-signal-cyan`, pulsing ring
- `i > activeStage`: pending — `text-muted`, hollow circle

### UploadSection (inline or extracted)

Migrates the drop zone logic from `Workstation.tsx` approach but wired
differently: instead of starting analysis inline, file selection calls
`navigate('/workstation', { state: { file } })`. `Workstation.tsx` guards
`location.state?.file` on mount and triggers `handleFileUpload` automatically.

Demo signal: no `generateMockSignal()` exists in this codebase — do not
create one. The "Load demonstration signal" button calls
`navigate('/workstation')` with no state, taking the user to Workstation's
standard empty upload state.

Drop zone styling:
```tsx
<div className="border border-dashed border-[#1E262E] rounded-[4px] p-16 text-center
                hover:border-[#22D3EE] transition-colors">
```

Drag-over state adds `border-[#22D3EE] bg-[rgba(34,211,238,0.04)]`.
`Browse Files` button: `border border-[#1E262E] text-muted hover:border-signal-cyan hover:text-primary`.

### ImpactSection (inline or extracted)

Stats array (all values verified against codebase — see requirements.md R8.3):
```ts
const STATS = [
  { value: 6,    label: 'PIPELINE STAGES' },        // ProcessingChain, Workstation
  { value: 256,  label: 'PSD FREQUENCY BINS' },      // backend/dsp/psd.py nperseg=256 default
  { value: 512,  label: 'SPECTRUM DATA POINTS' },    // backend/api/analysis.py range(512)
  { value: 1000, label: 'MAX CONSTELLATION POINTS' },// backend/api/visualizations.py [:1000]
];
```

Counter hook (inline `useCountUp`): `requestAnimationFrame` loop,
`easeOutQuad` timing, 1200 ms duration, fires once on `isRevealed`.

Grid: `grid grid-cols-2 lg:grid-cols-4 gap-6`.

---

## 7. Workstation.tsx modifications (minimal)

One small addition — no structural change:

**File pre-load guard** (after state declarations):
```tsx
const location = useLocation();
useEffect(() => {
  const preloadFile = location.state?.file as File | undefined;
  if (preloadFile) handleFileUpload(preloadFile);
}, []);
```

This is additive. The existing upload-in-workstation flow is entirely
preserved — `Workstation.tsx` still works when navigated to directly without
state.

> **Removed from this section**: the `location.state?.mockData` hydration
> guard. `mockData.ts` is not created; no mock data is ever passed via
> navigation state.

---

## 8. Styling constraints checklist

| Rule | Enforcement |
|---|---|
| No `translateY` + opacity fade-in entrance | `useScanReveal` clip-path only; framer-motion `animate` used only for micro-interactions (button tap scale) |
| No drop shadows / blur blobs / gradient blobs | No `boxShadow`, `filter: blur`, or `radial-gradient` background outside the hero fallback |
| Gradient text on headline only | `bg-gradient-to-br from-signal-cyan to-signal-green bg-clip-text text-transparent` applied to `<h1>` in Hero; nowhere else |
| Glow only in WebThreads canvas | No `text-shadow`, no `box-shadow glow-*`, no CSS `drop-shadow` filter outside hero |
| Pill buttons banned | CTAs use `rounded-md` (or square), not `rounded-full` with shadow |
| No auto-playing carousel | Not used anywhere |
| Icon-pack glyphs banned | Feature visuals are hand-authored SVGs |

---

## 9. File change summary

| File | Action | Notes |
|---|---|---|
| `frontend/tailwind.config.js` | Modify | Add tokens (text-muted = #8A939D), preserve existing |
| `frontend/src/index.css` | Modify | Update body bg colour |
| `frontend/src/hooks/useScanReveal.ts` | Create | clip-path reveal hook |
| `frontend/src/components/WebThreads/WebThreads.jsx` | Create | Verbatim shader |
| `frontend/src/components/WebThreads/WebThreads.css` | Create | Container rule |
| `frontend/src/components/WebThreads/WebThreads.d.ts` | Create | TS ambient decl |
| `frontend/src/pages/LandingView.tsx` | Replace | Full rewrite |
| `frontend/src/pages/Workstation.tsx` | Modify | Add `location.state?.file` guard only |

**Total: 8 files. All inside `frontend/`. `mockData.ts` is not created.**

No files outside `frontend/` are touched.
