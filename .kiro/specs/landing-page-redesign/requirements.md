# Landing Page Redesign — Requirements

> Status: **FINAL DRAFT — ready for approval**
>
> Covers all four prompt messages. Items that could not be verified against
> the codebase are flagged with ⚠️ inline.

---

## Background & Context

`frontend/src/pages/LandingView.tsx` is currently a full framer-motion
marketing page built on a purple/teal Topography WebGL background with blob
glows, rounded-pill elements, and `translateY` fade-up animations on every
section — all patterns this redesign explicitly prohibits.

The redesign **replaces `LandingView.tsx` entirely**. The route structure in
`App.tsx` (`/` → `<LandingView>`, `/workstation` → `<Workstation>`) is
unchanged. The `onLaunch` prop signature on `LandingView` is preserved.

`ogl@^1.0.6` is already in `package.json` — no new dependency install needed.

---

## Design System Constraints

All colours, typography, and spacing must use the following tokens exclusively.
No new palette values may be introduced anywhere in this feature.

| Token | Value | Usage |
|---|---|---|
| `background` | `#0A0E12` | Page background, footer bg |
| `panel` | `#11171D` | Card/panel fill |
| `panel-raised` | `#151C22` | Elevated panel, chip bg |
| `border` | `#1E262E` | All borders, dividers, grid lines |
| `signal-cyan` | `#22D3EE` | Active/live signal, primary accent |
| `signal-green` | `#34D399` | Success / validated / detected |
| `signal-amber` | `#FBBF24` | Uncertain / in progress |
| `signal-red` | `#F87171` | Failed / invalid |
| `text-primary` | `#E6EDF3` | Body text, headings |
| `text-muted` | `#8A939D` | Secondary labels, captions (corrected from `#6B7785` — see R11) |

**Typography**
- `font-mono` (JetBrains Mono) — all numbers, technical labels, data readouts,
  `STAGE N` prefixes, counter values, chip text, button text for CTAs.
- `font-sans` (Inter) — body copy, section headings, feature card prose.

Both fonts must be loaded via the project's existing Tailwind config. If they
are not currently configured there, add them via `@import` in `index.css` and
register them in `tailwind.config.js` — do not add a new npm package.

**The one glow rule**: `box-shadow: 0 0 8px #22D3EE` is permitted **only** on
the clip-reveal radar sweep line (R4B.3) and exists for ≤ 900 ms before
disappearing. No other glow, blur blob, or radial gradient background appears
anywhere on the page outside the hero section (R4.1–R4.2).

**The one gradient rule**: The hero `<h1>` uses a `linear-gradient`
`#22D3EE → #34D399` as a text fill (R4.4). No other gradient fill or
background gradient appears anywhere on the page.

**Prohibited patterns — must not appear anywhere on this page:**
- `translateY` / `translateX` entrance animations (the arrow nudge in R4.6 is
  a hover state, not an entrance)
- `scale` on buttons or cards at any interaction state
- `whileInView` / `useInView` or any framer-motion scroll-triggered animation
- `useSpring` / `useScroll` parallax
- Framer Motion `motion.*` wrappers on any element except the navbar bar (R3.2)
- Auto-playing carousels or looping content
- Rounded-pill buttons with drop shadows
- `blur-[…]` colour halos behind cards
- Generic icon-pack glyphs in feature cards
- Any purple/teal tones from the old design (`#6d28d9`, `#14b8a6`, etc.)

---

## Requirement 1 — WebThreads Animated Background Component

### R1.1 File locations
Create exactly three files:
- `frontend/src/components/WebThreads/WebThreads.jsx`
- `frontend/src/components/WebThreads/WebThreads.css`
- `frontend/src/components/WebThreads/WebThreads.d.ts`

### R1.2 Source fidelity
`WebThreads.jsx` must contain the **verbatim** source provided across the
specification messages. Specifically:
- The `vertex` and `fragment` GLSL shader strings must not be modified in any
  way — not reformatted, not "improved", not ported to a template literal with
  different whitespace.
- The `ctxMap` WeakMap pattern, the `FAN_MODE` const, the `hexToRgb` helper,
  the two `useEffect` blocks (setup + props sync), and the resize/intersection/
  visibility observer teardown sequence must be preserved exactly.
- The component is a plain `.jsx` file — do not convert to `.tsx`. Vite's React
  plugin handles `.jsx` without configuration.

### R1.3 CSS file
`WebThreads.css` must contain exactly:
```css
.web-threads-container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
```
No additional rules.

### R1.4 TypeScript ambient declaration
`WebThreads.d.ts` provides a type shim so `.tsx` files can import the component
without `tsc -b` errors. It must declare the full prop interface matching the
component's defaultProps. It must not contain any shader or runtime logic.

```ts
declare module '*/WebThreads' {
  import { FC } from 'react';
  export interface WebThreadsProps {
    color1?: string;
    color2?: string;
    color3?: string;
    speed?: number;
    threadCount?: number;
    frequency?: number;
    spread?: number;
    taper?: number;
    position?: number;
    fanMode?: 'center' | 'left' | 'right';
    glow?: number;
    falloff?: number;
    thickness?: number;
    brightness?: number;
    opacity?: number;
    mirror?: boolean;
    shimmer?: boolean;
    grain?: boolean;
    grainIntensity?: number;
    mouseInteraction?: boolean;
    mouseStrength?: number;
    backgroundColor?: string;
    lightMode?: boolean;
    className?: string;
  }
  const WebThreads: FC<WebThreadsProps>;
  export default WebThreads;
}
```

### R1.5 Theme props
When `<WebThreads />` is mounted in the hero section it must receive exactly
these props (all others remain at component defaults):

```jsx
color1="#22D3EE"
color2="#34D399"
color3="#E6EDF3"
speed={0.15}
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
mouseInteraction={true}
mouseStrength={0.25}
```

### R1.6 Reduced-motion handling
Inside `WebThreads.jsx`, in the setup `useEffect`, add:
```js
const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
if (mq.matches) {
  // override props for this render cycle
  speed = 0.01;
  mouseInteraction = false;
}
```
The canvas still renders — only interactivity and perceptible motion are
suppressed. The `mq` listener does not need to be reactive; it is read once
on mount.

### R1.7 Mobile fallback
In `LandingView.tsx`, before rendering the hero section:
- Create a `useWindowWidth` hook (or equivalent inline `useState` + `resize`
  listener in a `useEffect`) that returns the current `window.innerWidth`.
- If `windowWidth < 768`: do not mount `<WebThreads />`. Instead, render a
  `<div>` with:
  ```css
  background: radial-gradient(
    ellipse at 50% 40%,
    rgba(34, 211, 238, 0.07) 0%,
    rgba(52, 211, 153, 0.04) 50%,
    transparent 100%
  );
  ```
  over the `#0A0E12` section background. This is the only permitted
  `radial-gradient` and it is for the mobile hero background only.
- The `WebThreads` canvas is never instantiated on small screens.

---

## Requirement 2 — Page Structure

The page is a single vertically-scrollable document with no horizontal scroll.
Section order (top to bottom):

1. **Navbar** — fixed, `z-50` (R3)
2. **Hero** — full viewport height (R4)
3. **Features** — `id="features"` (R5)
4. **Pipeline** — `id="pipeline"` (R6)
5. **Upload / Get Started** — `id="upload"` (R7)
6. **Impact** — `id="impact"` (R8)
7. **Footer** (R9)

Global page settings in `index.css`:
- `html { scroll-behavior: smooth; }`
- `body { background-color: #0A0E12; color: #E6EDF3; }`
  (replaces existing `background-color: #000000`)
- Custom scrollbar: track `#0A0E12`, thumb `#1E262E`, thumb-hover `#22D3EE44`

---

## Requirement 3 — Navbar

### R3.1 Structure
`<nav aria-label="Main navigation">`, `position: fixed`, `top: 0`, `left: 0`,
`right: 0`, `z-index: 50`, height `64px` (`h-16`).

Three zones in a single flex row:
- **Left** (flex-shrink-0): `SIGMA` wordmark
- **Centre-right** (flex-1, justify-end, gap): nav links — hidden below 768 px
- **Right** (flex-shrink-0): `Launch Workstation` button

### R3.2 Styling
- Background: `rgba(17, 23, 29, 0.85)` (`#11171D` at 85% opacity)
- `backdrop-filter: blur(4px)` (`backdrop-blur-sm`)
- Bottom border: `1px solid #1E262E` only — no other border, no border-radius,
  no box-shadow
- Full viewport width — no max-width constraint on the bar itself

### R3.3 Wordmark
`SIGMA` in `font-mono font-bold`, `color: #E6EDF3`, `letter-spacing: 0.2em`
(`tracking-widest`). No gradient, no glow, no icon.

### R3.4 Nav links
Each link wraps its label in two `<span>` halves for the centre-outward
underline effect:

```jsx
<a href="#features" className="nav-link">
  <span className="nav-link-left-half" />  {/* left underline half */}
  Features
  <span className="nav-link-right-half" /> {/* right underline half */}
</a>
```

CSS:
```css
.nav-link { position: relative; font-size: 0.875rem; color: #6B7785;
            transition: color 150ms ease; text-decoration: none; }
.nav-link:hover { color: #E6EDF3; }

/* underline halves — sit below the text */
.nav-link-left-half,
.nav-link-right-half {
  position: absolute; bottom: -2px; height: 2px;
  background: #6B7785; width: 0;
  transition: width 200ms ease;
}
.nav-link-left-half  { right: 50%; }
.nav-link-right-half { left: 50%; }

.nav-link:hover .nav-link-left-half,
.nav-link:hover .nav-link-right-half { width: 50%; }

/* Active state — full cyan underline, no animation */
.nav-link.active { color: #E6EDF3; }
.nav-link.active .nav-link-left-half,
.nav-link.active .nav-link-right-half { width: 50%; background: #22D3EE; }
```

**Scroll-spy**: one `IntersectionObserver` per section (`#features`,
`#pipeline`, `#upload`, `#impact`), `threshold: 0.4`. When a section is
≥ 40% in the viewport, set it as active. Observer persists — does not
disconnect after first fire (unlike the clip-reveal observer). Uses a single
`activeSection` state string.

**Smooth scroll**: `href="#sectionId"` anchors work naturally with
`scroll-behavior: smooth` on `<html>`.

### R3.5 Launch Workstation button
```css
/* rest */
background: transparent;
border: 1px solid #22D3EE;
color: #22D3EE;
font-family: JetBrains Mono, monospace;
font-size: 0.75rem;
text-transform: uppercase;
letter-spacing: 0.1em;
border-radius: 2px;
padding: 6px 16px;
transition: background 150ms ease, color 150ms ease;

/* hover */
background: #22D3EE;
color: #0A0E12;
```
No drop shadow, no scale, no pill shape.

Behaviour: smooth-scrolls to `#upload` (same target as hero primary CTA).

### R3.6 Mobile nav
Below 768 px (`md:` breakpoint): `display: none` on the nav links wrapper.
Wordmark and Launch button remain visible. No hamburger required for MVP.

---

## Requirement 4 — Hero Section

### R4.1 Layout
```
<section style="position: relative; min-height: 100dvh; overflow: hidden">
  <!-- WebThreads fills section absolutely -->
  <div style="position: absolute; inset: 0; z-index: 0"> ... </div>
  <!-- Content centred -->
  <div style="position: relative; z-index: 10; display: flex;
              flex-direction: column; align-items: center;
              justify-content: center; min-height: 100dvh; padding: 0 1.5rem">
    ...
  </div>
</section>
```
`overflow: hidden` on the section prevents WebThreads from causing horizontal
scroll (acceptance criterion AC1).

### R4.2 WebThreads background
Mount `<WebThreads />` with R1.5 props. Apply R1.6 (reduced-motion) and R1.7
(mobile fallback). The canvas container is `position: absolute; inset: 0;
z-index: 0; pointer-events: none` — it must not intercept clicks on hero
content. `aria-hidden="true"` and `role="none"` on the container div.

### R4.3 Eyebrow label
```
SIGNAL INTELLIGENCE WORKSTATION
```
`font-mono`, `font-size: 0.75rem`, `text-transform: uppercase`,
`letter-spacing: 0.2em`, `color: #6B7785`. Plain text element, no border, no
background, no badge chrome, no entrance animation.

### R4.4 Headline
```html
<h1>
  Automated<br/>
  signal<br/>
  intelligence<br/>
  &amp; analysis
</h1>
```
```css
h1 {
  font-family: Inter, sans-serif;
  font-weight: 900;
  font-size: clamp(2.5rem, 8vw, 6rem);
  line-height: 0.95;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #22D3EE 0%, #34D399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```
No per-word animation. No `textShadow`. Visible immediately on page load.
This is the **one intentional gradient** on the page — it echoes the
WebThreads thread colours.

### R4.5 Supporting copy
```
Closed-loop hypothesis validation for unknown RF signals —
from raw IQ to demodulated bitstream.
```
`font-sans`, `font-size: 1rem`, `color: #6B7785`, `max-width: 480px`,
centred. No animation.

### R4.6 CTA buttons
Two buttons, `display: flex; gap: 1rem` on `≥ 640px`, `flex-direction: column`
on mobile.

**Primary — "Launch Workstation →"**
```css
background: #22D3EE;
color: #0A0E12;
font-family: JetBrains Mono, monospace;
font-size: 0.75rem;
text-transform: uppercase;
letter-spacing: 0.1em;
border: none;
border-radius: 2px;
padding: 10px 24px;
cursor: pointer;
transition: background 150ms ease;
```
Hover: `background: #34D399`.

The `→` character is wrapped in a `<span>` with:
```css
display: inline-block;
transition: transform 150ms ease;
```
On button `:hover`, apply `transform: translateX(4px)` to that span only.
**This is the only `translateX` permitted on the page.**

Behaviour: `onClick={() => document.querySelector('#upload')?.scrollIntoView({ behavior: 'smooth' })}`.

**Magnetic hover** — see R4.8.

**Secondary — "View Documentation"**
```css
background: transparent;
border: 1px solid #1E262E;
color: #6B7785;
font-family: JetBrains Mono, monospace;
font-size: 0.75rem;
text-transform: uppercase;
letter-spacing: 0.1em;
border-radius: 2px;
padding: 10px 24px;
cursor: pointer;
transition: border-color 150ms ease, color 150ms ease;
```
Hover: `border-color: #22D3EE; color: #E6EDF3`.

Link target: render as `<button aria-disabled="true" title="Documentation coming soon">`.
No href, no fake route, no link to `frontend/README.md`.

**Magnetic hover** — see R4.8.

### R4.7 Hero cursor reticle
A custom crosshair that tracks the cursor inside the hero section only.

**SVG design** (32 × 32 px viewBox):
```svg
<!-- four tick marks around a 4px centre void -->
<line x1="16" y1="4"  x2="16" y2="12" stroke="#22D3EE" stroke-width="1"/>  <!-- top -->
<line x1="16" y1="20" x2="16" y2="28" stroke="#22D3EE" stroke-width="1"/>  <!-- bottom -->
<line x1="4"  y1="16" x2="12" y2="16" stroke="#22D3EE" stroke-width="1"/>  <!-- left -->
<line x1="20" y1="16" x2="28" y2="16" stroke="#22D3EE" stroke-width="1"/>  <!-- right -->
<!-- outer ring at 40% opacity -->
<circle cx="16" cy="16" r="8" fill="none" stroke="#22D3EE"
        stroke-width="1" opacity="0.4"/>
```

**Component implementation**:
- `position: fixed; z-index: 50; pointer-events: none; aria-hidden="true"`
- Contains the inline SVG above, `width: 32px; height: 32px`
- Centred on cursor via `transform: translate(posX - 16px, posY - 16px)`
- `rAF` lerp loop: on each frame, `displayX += (mouseX - displayX) * 0.12`,
  `displayY += (mouseY - displayY) * 0.12`. Apply via
  `element.style.transform`.
- `isInHero` state toggled by `mouseenter`/`mouseleave` on the hero
  `<section>`. Show/hide via `opacity: 0/1; transition: opacity 100ms`.
- While active: `cursor: none` on the hero `<section>`. On leave: `cursor: auto`.
- **Do not render** when `'ontouchstart' in window` (touch device).
- **Do not render** when `prefers-reduced-motion: reduce`.
- Cancel rAF on component unmount.

### R4.8 Magnetic hover on CTA buttons

**`useMagneticHover(ref: RefObject<HTMLElement>): void`** — side-effect hook only.

```ts
function useMagneticHover(ref) {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const el = ref.current;
    if (!el) return;

    let rafId = 0;
    let ox = 0, oy = 0;       // current displayed offset
    let txOx = 0, txOy = 0;   // target offset

    const onMove = (e) => {
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top  + rect.height / 2;
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;
      txOx = Math.max(-7, Math.min(7, dx * 0.15));
      txOy = Math.max(-7, Math.min(7, dy * 0.15));
    };

    const onLeave = () => { txOx = 0; txOy = 0; };

    const loop = () => {
      ox += (txOx - ox) * 0.18;
      oy += (txOy - oy) * 0.18;
      el.style.transform = `translate(${ox.toFixed(2)}px, ${oy.toFixed(2)}px)`;
      if (Math.abs(ox) > 0.1 || Math.abs(oy) > 0.1 || Math.abs(txOx) > 0.1 || Math.abs(txOy) > 0.1) {
        rafId = requestAnimationFrame(loop);
      }
    };

    const onEnter = () => { rafId = requestAnimationFrame(loop); };

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
```

Call `useMagneticHover(primaryBtnRef)` and `useMagneticHover(secondaryBtnRef)`
in the hero component. Disable entirely when `prefers-reduced-motion: reduce`.

**Design rationale**: framer-motion's `useMotionValue`/`useSpring` is avoided
here to keep the hero interaction layer consistent — both the reticle (R4.7)
and the magnetic buttons use the same plain `rAF` lerp pattern, so they share
one mental model for the implementer and have zero library overhead.

### R4.9 Scroll indicator
Centred below the CTA buttons:
```
SCROLL        ← font-mono text-xs text-muted
  |           ← 2px wide × 40px tall, background #1E262E
  ∨           ← 8px chevron-down SVG, stroke #6B7785
```
CSS `@keyframes`:
```css
@keyframes pulse-opacity {
  0%, 100% { opacity: 0.4; }
  50%       { opacity: 1.0; }
}
.scroll-indicator { animation: pulse-opacity 1.8s ease-in-out infinite; }
```
**No `translateY`**. `aria-hidden="true"`.

---

## Requirement 4B — Shared Scroll-Reveal System

One mechanism, reused by every section below the hero. Referenced as
"clip-reveal" throughout later requirements.

### R4B.1 Mechanism — clip-path unmask
```css
/* hidden state */
.reveal-target {
  clip-path: inset(0 100% 0 0);
  transition: clip-path 700ms cubic-bezier(0.16, 1, 0.3, 1);
}
/* revealed state (class added by JS) */
.reveal-target.is-revealed {
  clip-path: inset(0 0% 0 0);
}
```
Duration: `700ms` for cards, `900ms` for full-width panels. Easing:
`cubic-bezier(0.16, 1, 0.3, 1)` (strong ease-out). No opacity change, no
translation — the clip is the only animated property.

Under `prefers-reduced-motion: reduce`:
- Duration → `200ms`
- Sweep line (R4B.3) skipped entirely

### R4B.2 Trigger — IntersectionObserver
```ts
const observer = new IntersectionObserver(
  ([entry]) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.classList.add('is-revealed');
        entry.target.dataset.revealed = 'true';
      }, delay);
      observer.unobserve(entry.target); // fire once only
    }
  },
  { threshold: 0.2 }
);
observer.observe(element);
```
The observer disconnects immediately after firing via `unobserve`. A
`data-revealed` attribute guards against double-fires.

### R4B.3 Radar sweep line
An `::after` pseudo-element on `.reveal-target`:
```css
.reveal-target::after {
  content: '';
  position: absolute;
  top: 0; right: 100%; /* starts off right edge of clip */
  width: 2px; height: 100%;
  background: #22D3EE;
  box-shadow: 0 0 8px #22D3EE; /* ONLY permitted glow outside hero */
  transition: right 700ms cubic-bezier(0.16, 1, 0.3, 1),
              opacity 50ms ease 700ms; /* fade out after sweep */
  pointer-events: none;
}
.reveal-target.is-revealed::after {
  right: 0%;
  opacity: 0; /* gone once clip completes */
}
```
This is the sole permitted glow outside the hero, and it exists only during
the ≤ 900 ms reveal transition. Omit entirely under `prefers-reduced-motion`.

### R4B.4 `useReveal` hook
```ts
function useReveal(ref: RefObject<HTMLElement>, options?: { delay?: number }) {
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setRevealed(true), options?.delay ?? 0);
          observer.unobserve(el);
        }
      },
      { threshold: 0.2 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref, options?.delay]);
  return { revealed };
}
```
Usage: `const { revealed } = useReveal(cardRef, { delay: 100 })` → apply
`className={revealed ? 'reveal-target is-revealed' : 'reveal-target'}`.

### R4B.5 Elements that receive clip-reveal
| Element | Stagger |
|---|---|
| Section header (label + heading) — all sections | No stagger, fires as unit |
| Feature cards ×4 (R5) | 0 ms, 100 ms, 200 ms, 300 ms |
| Pipeline stage nodes ×6 (R6) | 0 ms, 150 ms, 300 ms, 450 ms, 600 ms, 750 ms |
| Upload drop-zone panel (R7) | No stagger, fires as unit |
| Impact USP block (R8) | No stagger |
| Impact benefit cards ×4 (R8) | 0 ms, 100 ms, 200 ms, 300 ms |

**Not** clip-revealed: navbar, hero section, footer, mini-visuals inside
feature cards (they appear already-visible when their parent card reveals).

---

## Requirement 5 — Features Section

### R5.1 Section id and scroll target
`<section id="features">`. Padding: `py-24` desktop, `py-16` mobile.

### R5.2 Section header
Clip-reveal as a unit (no stagger):
- Label: `font-mono text-xs text-muted uppercase tracking-widest` — `// FEATURES`
- Heading: `font-sans font-bold text-[2rem] text-primary` —
  `"From raw capture to validated signal"`
- `margin-bottom: 3rem`
- No gradient text, no framer-motion

### R5.3 Asymmetric grid layout
On desktop (`≥ 1024px`), a CSS grid with **non-uniform columns**:
```css
.features-grid {
  display: grid;
  grid-template-columns: 7fr 5fr;
  gap: 1.5rem;
}
/* Row 2 reverses the spans */
.features-grid .card:nth-child(3) { grid-column: 1; }
.features-grid .card:nth-child(4) { grid-column: 2; }
/* But swap widths in row 2 via order or explicit spans */
```

Exact pattern:
```
Row 1:  [card-1: col-span 7]  [card-2: col-span 5]
Row 2:  [card-3: col-span 5]  [card-4: col-span 7]
```

Use `grid-template-columns: 7fr 5fr` for row 1 and `5fr 7fr` for row 2.
Achievable with two separate sub-grids or by wrapping each row in a div with
its own grid. The implementation must produce the visual asymmetry — do not
use `repeat(4, 1fr)` or any equal-column variant.

Tablet (`768px–1023px`): `grid-template-columns: 1fr 1fr`.
Mobile (`< 768px`): single column.

### R5.4 Feature card structure
Each card:
```
┌─────────────────────────────────────────────────┐  ← 2px top border (accent colour)
│                                                 │  ← 1px border #1E262E, border-radius 4px
│  [text content: left]    [mini-visual: right]   │
│                                                 │
└─────────────────────────────────────────────────┘
```
- Background: `#11171D`
- Border: `1px solid #1E262E`
- Top border override: `2px solid <accent>`
- `border-radius: 4px`
- Padding: `p-6`
- No box-shadow, no blur
- Clip-reveal with stagger (R4B.5)
- Hover: `border-color: <accent>`, `transition: border-color 150ms ease`
  **No translateY lift**

On narrow cards: mini-visual below text content (flex-direction column).
On wide cards: mini-visual to the right of text content (flex-direction row).

Inside text content:
- Category label: `font-mono text-xs text-muted tracking-widest`
- Feature name: `font-sans font-semibold text-primary text-base mt-1`
- Body: `font-sans text-sm text-muted leading-relaxed mt-2`

### R5.5 Feature 1 — Spectrum Analysis
- Accent: `#22D3EE`
- Label: `SPECTRUM ANALYSIS`
- Name: `"Frequency-domain decomposition"`
- Body: `"FFT-derived power spectral density with peak detection and bandwidth
  estimation across the full capture."`
- **Mini-visual** (`120×80 px` inline SVG):
  - Background rect: `#0A0E12`, `rx=2`
  - Two horizontal dashed grid lines at y=20 and y=60: `stroke="#1E262E"
    stroke-dasharray="3,3" stroke-width="0.5"`
  - `<polyline>` of ~22 points forming a noise floor at y≈65 with a Gaussian
    peak centred at x=60, peak at y≈15. Points derived from:
    `y = 65 - 50 * exp(-((x-60)^2) / 200)` sampled at x = 5,10,15…120.
  - Stroke: `#22D3EE`, `stroke-width=1.5`, `fill=none`
  - Source: simplified from the polyline approach in `SpectrumViewer.tsx`;
    entirely hardcoded — no data prop.

### R5.6 Feature 2 — Waterfall View
- Accent: `#34D399`
- Label: `WATERFALL VIEW`
- Name: `"Time-frequency intensity map"`
- Body: `"Scrolling spectrogram rendered row-by-row, revealing signal
  persistence, drift, and spectral occupancy over time."`
- **Mini-visual** (`120×80 px` `<canvas>` drawn once in `useEffect`):
  - Use the same colour LUT as `WaterfallViewer.tsx`:
    - `t < 0.2`: `rgb(15, 23, 42)` (dark navy)
    - `0.2 ≤ t < 0.5`: interpolate to `rgb(20, 184, 166)` (teal)
    - `t ≥ 0.5`: interpolate to `rgb(255, 255, 166)` (bright)
  - Pixel value: `t = (sin(col/8) * 0.5 + 0.5) * (1 - |row/40 - 1|)`
    produces a diagonal bright band brighter in the middle rows.
  - One-time draw, no animation loop.
  - `width=120 height=80 style="image-rendering: pixelated"`

### R5.7 Feature 3 — Modulation Classification
- Accent: `#FBBF24`
- Label: `MODULATION CLASSIFICATION`
- Name: `"ML-ranked hypothesis candidates"`
- Body: `"Convolutional classifier ranks modulation schemes by confidence.
  Each candidate undergoes physical demodulation and FEC verification before
  acceptance."`
- **Mini-visual** (`120×80 px` inline SVG):
  - 4 rows of confidence bars. Each row: background `<rect fill="#1E262E">`
    full-width, foreground `<rect fill="#FBBF24">` at relative widths
    87%, 11%, 2%, 1%. Heights: `12px`, vertical gap `6px`.
  - Opacity per bar: `0.9`, `0.5`, `0.25`, `0.1` (visually encodes rank).
  - No text labels — bars only.
  - Source: simplified from the confidence bar in `HypothesisExplorer.tsx`.

### R5.8 Feature 4 — Constellation Mapping
- Accent: `#22D3EE`
- Label: `CONSTELLATION MAPPING`
- Name: `"IQ scatter diagram"`
- Body: `"Phase-space plot of demodulated symbols. Tight clusters indicate
  successful synchronization and low EVM; scatter indicates sync failure."`
- **Mini-visual** (`120×80 px` inline SVG):
  - Faint crosshair axes: two lines through centre, `stroke="#1E262E"
    stroke-width="0.5"`
  - Four QPSK symbol clusters at normalised positions `(±42, ±28)` in SVG
    coordinates (centre 60,40). Each cluster: 8 dots with ±4 px random-looking
    scatter (hardcode offsets — no `Math.random()`). `r=2 fill="#22D3EE"
    fill-opacity="0.6"`
  - Source: simplified from `ConstellationViewer.tsx` circle approach.

---

## Requirement 6 — Pipeline Section

### R6.1 Section id
`<section id="pipeline">`. Padding: `py-24` desktop, `py-16` mobile.

### R6.2 Section header
Clip-reveal as a unit:
- Label: `font-mono text-xs text-muted uppercase tracking-widest` — `// PIPELINE`
- Heading: `font-sans font-bold text-[2rem] text-primary` —
  `"Six stages, fully traced"`

### R6.3 Stage timeline — visual spec
**Desktop** (`≥ 768px`): horizontal rail — six nodes connected by lines.
**Mobile** (`< 768px`): vertical stack — nodes stacked with vertical connectors.

Node:
```
  ┌──────┐
  │  ✓   │  ← 40×40px circle, fill = status colour
  └──────┘
  STAGE 01  ← font-mono text-xs text-muted
  INGESTION ← font-sans text-xs text-primary
```

Status colours — match `ProcessingChain.tsx` exactly:
- `pending`: fill `#1E262E`, icon `○`
- `running`: fill `#22D3EE`, icon `⟳`, CSS `opacity: 0.6→1.0` pulse 1s loop
- `completed`: fill `#34D399`, icon `✓`

Connector line (horizontal desktop / vertical mobile):
- `2px` thick
- Colour: `#34D399` if upstream node is `completed`, else `#1E262E`
- Transitions to `#34D399` as the auto-play advances

**Do not use the existing `ProcessingChain` component** — it wraps nodes in
`framer-motion motion.div` with `initial={{ scale: 0 }} animate={{ scale: 1 }}`
which violates R10. Implement this timeline in plain React + CSS. The existing
`ProcessingChain.tsx` file is left unchanged.

Stage data:
| # | Label | Name |
|---|---|---|
| 01 | STAGE 01 | INGESTION |
| 02 | STAGE 02 | DSP ANALYSIS |
| 03 | STAGE 03 | MODULATION |
| 04 | STAGE 04 | DEMODULATION |
| 05 | STAGE 05 | FEC |
| 06 | STAGE 06 | VALIDATION |

### R6.4 Auto-play on scroll-into-view
Triggered once when the timeline section crosses 20% into viewport.

```ts
const [stages, setStages] = useState<Status[]>(Array(6).fill('pending'));

useEffect(() => {
  if (!hasPlayed && isInView) {
    setHasPlayed(true);
    let i = 0;
    const interval = setInterval(() => {
      setStages(prev => {
        const next = [...prev];
        if (i > 0) next[i - 1] = 'completed';
        if (i < 6) next[i] = 'running';
        return next;
      });
      i++;
      if (i > 6) {
        clearInterval(interval);
        setStages(Array(6).fill('completed'));
      }
    }, 600);
    return () => clearInterval(interval);
  }
}, [isInView, hasPlayed]);
```

`isInView`: derived from an `IntersectionObserver` (threshold 0.2), same
pattern as R4B.2 but without disconnecting (or disconnect after first fire and
use `hasPlayed` guard).

`prefers-reduced-motion: reduce`: skip the interval; immediately set all stages
to `completed` when `isInView` becomes true.

### R6.5 Parameter chip strip
Below the timeline. Clip-reveal as a unit (no stagger).

Six chips in a flex-wrap row:
```
· fs    · fc    · modulation    · symbol_rate    · fec_scheme    · bandwidth
```
Each chip:
- Background: `#151C22`, border `1px solid #1E262E`, `border-radius: 2px`
- Padding: `px-3 py-1`, `font-mono text-xs text-muted`
- The `·` prefix is `color: #22D3EE`

### R6.6 MVP scope note
Full-width panel below chips. Clip-reveal as a unit.
- Background: `#151C22`
- Border: `1px solid #1E262E`
- Left border: `4px solid #FBBF24`
- `border-radius: 4px`
- Label: `font-mono text-xs` `color: #FBBF24` `tracking-widest` — `MVP SCOPE`
- Body: `font-sans text-sm text-primary` —
  `"Initial validation targets BPSK and QPSK modulation with convolutional FEC
  codes. Each hypothesis undergoes real Viterbi decoding with syndrome
  checking before acceptance."`

---

## Requirement 7 — Upload / Get Started Section

### R7.1 Section id
`<section id="upload">`. This is the scroll target for:
- Navbar `Launch Workstation` button (R3.5)
- Hero primary CTA `Launch Workstation →` (R4.6)
- Impact final CTA (R8.5)

### R7.2 Section header
Clip-reveal as a unit:
- Label: `font-mono text-xs text-muted uppercase tracking-widest` —
  `// GET STARTED`
- Heading: `font-sans font-bold text-[2rem] text-primary` —
  `"Load a signal, start analysing"`

### R7.3 Drop-zone panel
Clip-reveal as a unit (R4B.5).

```css
/* idle */
border: 1px dashed #1E262E;
background: #11171D;
border-radius: 4px;
padding: 4rem 2rem;
text-align: center;
transition: border-color 100ms ease, background 100ms ease;

/* drag-over */
border-color: #22D3EE;
background: rgba(34, 211, 238, 0.04);
```

Idle content (top to bottom, all centred):

1. **Upload SVG icon** (40×40 px inline SVG, no icon library):
   ```svg
   <circle cx="20" cy="20" r="18" fill="none" stroke="#22D3EE" stroke-width="1.5"/>
   <line x1="20" y1="28" x2="20" y2="12" stroke="#22D3EE" stroke-width="1.5"/>
   <polyline points="14,18 20,12 26,18" fill="none" stroke="#22D3EE" stroke-width="1.5"/>
   ```
2. **Heading**: `font-mono text-sm text-primary uppercase tracking-wider` —
   `"DROP SIGNAL FILE"`
3. **Format label**: `font-mono text-xs text-muted` — `"Accepts .IQ · .WAV · .RAW"`
   (⚠️ Note: `.RAW` is included in the UI label and `<input accept>` but the
   backend currently only handles `.iq` and `.wav`. A `.raw` file will be
   accepted by the UI but will fail at the API upload step. This is
   documented, acceptable for MVP, and must not silently discard the file —
   the existing error handling in Workstation will display the API error.)
4. **Browse Files button**: `font-mono text-xs text-muted uppercase tracking-wider`,
   `border: 1px solid #1E262E`, `border-radius: 2px`, `px-5 py-2`,
   transparent background. Hover: `border-color: #22D3EE; color: #E6EDF3`.
   `onClick`: `fileInputRef.current?.click()`. No scale, no shadow, no pill.

Hidden `<input type="file" accept=".iq,.wav,.raw" ref={fileInputRef}>`.

Drag events: `onDragEnter`, `onDragOver`, `onDragLeave`, `onDrop` — same
pattern as `Workstation.tsx` but simplified (no `isDragging` scale).

### R7.4 Divider
```html
<div class="divider">
  <hr/> <span>OR</span> <hr/>
</div>
```
```css
.divider { display: flex; align-items: center; gap: 1rem; margin: 1.5rem 0; }
.divider hr { flex: 1; border: none; border-top: 1px solid #1E262E; }
.divider span { font-family: monospace; font-size: 0.75rem; color: #6B7785; }
```

### R7.5 "Load demonstration signal" link
```
LOAD DEMONSTRATION SIGNAL
```
`font-mono text-xs text-muted uppercase tracking-wider`. Hover: `color: #E6EDF3`.
No border, no background — plain styled button.

**Behaviour**: `generateMockSignal()` does not exist in this codebase — there
is no `frontend/src/services/mockData.ts` and the global codebase grep
confirmed no mock signal generation function. The demo link calls
`navigate('/workstation')`, taking the user to the Workstation's own upload
flow. Do **not** create a `mockData.ts` stub or fake data generator.

### R7.6 File handling — landing → workstation handoff
On file drop/select in the landing upload area:
```ts
navigate('/workstation', { state: { file } });
```

`Workstation.tsx` must be updated to read this on mount:
```ts
const location = useLocation();
useEffect(() => {
  const file = location.state?.file as File | undefined;
  if (file) handleFileUpload(file);
}, []);
```
This preserves all existing upload, analysis, and error-handling logic in
Workstation without duplication.

### R7.7 Drop-zone states
- **Idle**: content from R7.3
- **Drag-over**: border/background from R7.3 drag-over CSS
- No "Analyzing" spinner in the landing drop-zone — analysis begins after
  navigation to Workstation

---

## Requirement 8 — Impact Section

### R8.1 Section id
`<section id="impact">`. Padding: `py-24` desktop, `py-16` mobile.

### R8.2 Section header
Clip-reveal as a unit:
- Label: `font-mono text-xs text-muted uppercase tracking-widest` — `// IMPACT`
- Heading: `font-sans font-bold text-[2rem] text-primary` —
  `"Built for real signal work, not slideshows"`

### R8.3 Animated stat counters
Four stats in a `2×2` grid (desktop) / `1×4` stack (mobile). Each stat
consists of a large number that counts up from 0 when scrolled into view,
plus a text label beneath.

**Counter animation**: plain `requestAnimationFrame`, not framer-motion.
```ts
function useCountUp(target: number, isInView: boolean, duration = 1200) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!isInView) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target); return;
    }
    const start = performance.now();
    const raf = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(eased * target));
      if (progress < 1) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  }, [isInView, target, duration]);
  return value;
}
```

**Stat values — verified against codebase**:

| Stat | Value | Label | Source |
|---|---|---|---|
| Pipeline stages | `6` | `PIPELINE STAGES` | `ProcessingChain.tsx`, `Workstation.tsx` processingStages array |
| PSD frequency bins | `256` | `PSD FREQUENCY BINS` | `backend/dsp/psd.py` `nperseg=256` default; `backend/dsp/bandwidth.py`, `carrier.py`, `snr.py` all use the same default |
| Spectrum API points | `512` | `SPECTRUM DATA POINTS` | `backend/api/analysis.py` line 346: `for i in range(512)` |
| Constellation points | `1000` | `MAX CONSTELLATION POINTS` | `backend/api/visualizations.py`: `i_vals = iq_data[0, ::100][:1000]` |

⚠️ **Flagged — not used**: the prompt suggested `64×256 waterfall resolution`.
The actual backend uses `nperseg=128` in `visualizations.py`
(`compute_spectrogram(..., nperseg=128)`), which yields 128 frequency bins.
The number of time slices is variable (depends on file length). There is no
fixed `64×256` resolution constant in the codebase. This stat has been
replaced with `256 PSD bins` (verified) to avoid misrepresenting the project.

⚠️ **Flagged — classification time**: the prompt suggested including a
realistic "classification time" figure. No timing benchmark exists in the
frontend or backend code. ML inference timing would depend on hardware and
is not a constant. This stat has been replaced with `512 spectrum points`
(verified from the API mock) rather than invent a figure.

**Counter display**:
```
┌──────────────────────┐
│                      │
│    6                 │  ← font-mono font-bold text-[3.5rem] text-primary
│    PIPELINE STAGES   │  ← font-mono text-xs text-muted tracking-widest mt-1
│                      │
└──────────────────────┘
```
- Background: `#11171D`, border `1px solid #1E262E`, `border-radius: 4px`
- `padding: 2rem`
- The number uses `font-mono` (JetBrains Mono) — this is data, not a heading
- `text-[3.5rem]` or `clamp(2rem, 5vw, 3.5rem)` to fit responsively
- No suffix/prefix unless meaningful (no generic `+` or `%`)
- Clip-reveal with stagger: 0, 100, 200, 300 ms

### R8.4 Core USP panel
Full-width. Clip-reveal as a unit.
- Background: `#11171D`
- Border: `1px solid #1E262E`, top border `2px solid #34D399`
- `border-radius: 4px`, `padding: 2rem`
- No glow, no gradient background
- Heading: `font-sans font-bold text-primary text-xl` —
  `"Closed-Loop Signal Hypothesis Validation"`
- Body: `font-sans text-sm text-primary leading-relaxed` —
  `"SIGMA doesn't predict modulation schemes — it proves them. Every
  hypothesis is physically tested through the complete signal chain:
  synchronization, demodulation, and forward error correction. A hypothesis
  either fully decodes or it does not."`

### R8.5 Benefits grid
Four cards in `2×2` (desktop) / `1×4` (mobile). Clip-reveal with stagger.

| Title | Body |
|---|---|
| Proof Over Confidence | `"Systems output '82% QPSK' without verification. SIGMA attempts actual demodulation and Viterbi FEC decoding. If syndrome checks pass, the hypothesis is proven."` |
| Explainable Evidence | `"Every validated signal carries a complete evidence trail: which sync method succeeded, what demodulator configuration worked, which FEC parameters decoded cleanly."` |
| Automation at Scale | `"Process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically."` |
| Reduced False Positives | `"By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. Ambiguity is eliminated."` |

Card spec:
- Background: `#11171D`, border `1px solid #1E262E`, `border-radius: 4px`
- Title: `font-sans font-semibold color: #34D399`
- Body: `font-sans text-sm text-muted leading-relaxed`
- Hover: `border-color: #34D399`, `150ms ease`. No translateY.

### R8.6 Final CTA
Centred, `margin-top: 3rem`. Same button spec as R4.6 primary.
Label: `Launch Workstation →`. Scrolls to `#upload`.

---

## Requirement 9 — Footer

Minimal single-row footer. No clip-reveal.

```
┌──────────────────────────────────────────────────────────────────────────┐  ← 1px #1E262E top border
│  SIGMA                                          © 2026 SIGMA Project    │
└──────────────────────────────────────────────────────────────────────────┘
```
- Background: `#0A0E12` (page bg — no panel fill)
- Content: `max-width: 1280px`, centred, `px-6 py-6`
- Both text nodes: `font-mono text-xs text-muted`
- No social icons, no link columns, no newsletter field

---

## Requirement 10 — Animation Constraints

The complete permitted animation inventory. **Nothing outside this table is
allowed.**

| Element | Animation | Mechanism |
|---|---|---|
| WebThreads canvas | Continuous GLSL shader | OGL rAF loop (inside component) |
| Clip-reveal unmask | `clip-path` inset sweep, 700–900 ms | CSS `transition` |
| Clip-reveal sweep line | `::after` right 100%→0%, same timing | CSS `transition` + `box-shadow` glow ≤ 900 ms |
| Pipeline auto-play | `pending→running→completed` per stage | `setInterval` in `useEffect`, cleared after 6 fires |
| Stat counters | Count 0→target, 1200 ms, ease-out cubic | `requestAnimationFrame` |
| Hero cursor reticle | Lerp-follow mouse, factor 0.12 | `requestAnimationFrame` |
| Hero CTA magnetic hover | Lerp-offset ±7 px, factor 0.18 | `requestAnimationFrame` via `useMagneticHover` |
| Hero CTA arrow `→` | `translateX(4px)` on `:hover` only | CSS `transition: transform 150ms ease` |
| Scroll indicator | `opacity` 0.4→1.0→0.4, 1.8 s loop | CSS `@keyframes`, no framer-motion |
| Navbar bar entrance | `opacity` 0→1, 400 ms, mount only | Single framer-motion `motion.nav animate` |
| Nav underline grow | Two halves `width 0→50%`, 200 ms | CSS `transition: width 200ms ease` |
| CTA / button hover fill | Background/border colour swap | CSS `transition 150ms ease` |
| Card border hover | Border colour change | CSS `transition 150ms ease` |
| Navbar link active | Instant colour + underline | No transition |
| Pipeline `running` node | `opacity` pulse 0.6→1.0, 1 s loop | CSS `@keyframes` |

**Explicitly prohibited:**
- `translateY` / `translateX` entrance animations on any element
- `scale` on buttons, cards, or any element at any interaction state
- `whileInView`, `useInView`, `AnimatePresence` on any content element
- `useSpring`, `useScroll`, `useTransform` parallax
- Any framer-motion `initial → animate` stagger on list items
- Continuous bounce/float/levitate on any non-canvas element
- framer-motion `motion.*` wrappers on anything except the navbar `<nav>` bar

---

## Requirement 11 — Accessibility

- All interactive elements: `aria-label` or visible text label
- WebThreads canvas container: `aria-hidden="true" role="none"`
- Hero reticle div: `aria-hidden="true"`
- Scroll indicator: `aria-hidden="true"`
- Navbar: `<nav aria-label="Main navigation">`
- Disabled documentation button: `aria-disabled="true"` and `title="Documentation coming soon"`
- Section headings: proper heading hierarchy (`h1` hero, `h2` each section)
- Upload drop-zone: `role="button" aria-label="Drop signal file or click to browse"`
- Colour contrast: `text-muted` is set to `#8A939D` (corrected from the
  original `#6B7785` which yielded ~3.9:1 — failing WCAG AA for small text).
  `#8A939D` on `#11171D` yields ~5.1:1, passing AA. This value is already
  reflected in the design token table above and in `tailwind.config.js`.
- `prefers-reduced-motion: reduce` disables: WebThreads motion (near-static),
  hero reticle (not rendered), magnetic hover (no-op), clip-reveal sweep
  line (skipped), pipeline auto-play (instant-complete), counters
  (instant-final-value). Clip-reveal itself reduces to 200 ms with no line.
- `(hover: none)` media query or `'ontouchstart' in window`: hero reticle not
  rendered.

---

## Requirement 12 — Responsive Breakpoints

| Viewport | Navbar | Hero headline | Feature grid | Pipeline | Upload |
|---|---|---|---|---|---|
| 375 px | Wordmark + button only | `clamp` min (~2.5rem), no overflow | 1 column | Vertical stack | Full width |
| 768 px | All links visible | ~4.5rem | 2 equal columns | Horizontal rail | Full width |
| 1440 px | All links + button | ~6rem | Asymmetric 7/5 | Horizontal rail | Max-width centred |

Explicit requirements:
- At 375 px, the `<h1>` must not cause horizontal scroll. Test with
  `overflow-x: hidden` on `<html>`.
- Pipeline timeline switches horizontal → vertical below `768px`.
- Feature layout: asymmetric desktop, equal-column tablet, single-column mobile.
- No fixed `px` font sizes on headings — use `clamp()` or responsive `text-[…]`
  Tailwind utilities.

---

## Requirement 13 — Verification / Acceptance Criteria

All criteria must be confirmed before the spec is considered complete.

| ID | Criterion | How to verify |
|---|---|---|
| AC1 | WebThreads causes no horizontal scroll or layout shift | Load at 375px, 768px, 1440px; check `overflow-x` with DevTools; confirm no CLS in Lighthouse |
| AC2 | Magnetic buttons respect `prefers-reduced-motion` | Emulate via DevTools → Rendering → "Prefer reduced motion"; buttons must not move |
| AC3 | Custom cursor respects `prefers-reduced-motion` | Same emulation; reticle must not render |
| AC4 | Clip-reveal respects `prefers-reduced-motion` | Sweep line absent; duration 200ms; no glow |
| AC5 | Pipeline auto-play respects `prefers-reduced-motion` | All stages appear as `completed` instantly |
| AC6 | Stat counters respect `prefers-reduced-motion` | Final values shown immediately, no counting animation |
| AC7 | No glow/shadow/gradient outside permitted locations | Manual audit: grep `box-shadow`, `blur`, `radial-gradient`, `linear-gradient` in new files; only permitted instances should match R4B.3 (sweep glow) and R4.4 (headline gradient) and R1.7 (mobile hero bg) |
| AC8 | Navbar `Launch Workstation` → scrolls to `#upload` | Click at any scroll position; section comes into view |
| AC9 | Hero `Launch Workstation →` → scrolls to `#upload` | Same as AC8 from top of page |
| AC10 | File drop in upload section → navigates to Workstation and triggers analysis | Drop a `.iq` or `.wav` file; confirm `navigate('/workstation', { state: { file } })` fires; confirm Workstation auto-starts upload |
| AC11 | "Load demonstration signal" → navigates to Workstation | Click; confirm `/workstation` loads with empty state |
| AC12 | Workstation dashboard (State B) still functions | Upload a file from Workstation directly; confirm full analysis flow unchanged |
| AC13 | Responsive at 375px: no overflow, headline fits | DevTools device emulation |
| AC14 | Responsive at 768px: feature grid 2-col, pipeline horizontal | DevTools device emulation |
| AC15 | Responsive at 1440px: feature grid asymmetric, full layout | Wide browser window |
| AC16 | All created/modified files are inside `frontend/` | `git diff --name-only` — no file outside `frontend/` |
| AC17 | `tsc -b` passes with no new type errors | Run `npm run build` (Vite invokes tsc) |
| AC18 | No `console.error` in browser on page load | Open DevTools console on fresh load |

---

## Requirement 14 — File Modifications Summary

| File | Action | Related requirement |
|---|---|---|
| `frontend/src/components/WebThreads/WebThreads.jsx` | **Create** — verbatim source | R1.2 |
| `frontend/src/components/WebThreads/WebThreads.css` | **Create** | R1.3 |
| `frontend/src/components/WebThreads/WebThreads.d.ts` | **Create** — type shim | R1.4 |
| `frontend/src/pages/LandingView.tsx` | **Replace entirely** | R2–R9 |
| `frontend/src/pages/Workstation.tsx` | **Update** — add `useLocation` + `useEffect` to auto-trigger upload from `location.state?.file` | R7.6 |
| `frontend/src/index.css` | **Update** — `background: #0A0E12`, `scroll-behavior: smooth`, scrollbar colours | R2 |
| `frontend/tailwind.config.js` | **Update** — add new design tokens; update `text-muted` to `#8A939D`; keep all existing entries | R11 |
| `frontend/package.json` | **No change** — `ogl@^1.0.6` already present | R1 |

No files outside `frontend/` are modified.

---

## Flagged Items (not included / changed from prompts)

| Item from prompt | Status | Reason |
|---|---|---|
| `64×256 waterfall resolution` as a stat | **Not used** | Backend uses `nperseg=128` (128 freq bins); time slices are variable. No `64×256` constant exists. Replaced with `256 PSD bins` (verified). |
| Classification time figure | **Not included** | No timing benchmark in codebase. Inventing a number would misrepresent the project. Replaced with `512 spectrum data points` (verified from API). |
| `generateMockSignal()` / `mockData.ts` | **Not created** | Function does not exist in codebase. Demo path navigates to `/workstation` directly. |
| Old Problem/Solution section | **Replaced by Features** | The prompt superseded the initial draft's Problem/Solution section with the Features section in prompt 3. |
| `.raw` file support | **UI only, not wired** | UI accepts `.raw` in label and `<input accept>` but backend does not handle it; error surfaced to user via existing API error handling. |

---

## Out of Scope

- Changes to any component used exclusively by `Workstation.tsx`
  (no `SpectrumViewer`, `WaterfallViewer`, `ConstellationViewer`, etc. are
  modified — they are only referenced for mini-visual derivation)
- Backend, ML, dataset, or root-level files
- Hamburger / drawer mobile nav
- Dark/light mode toggle
- i18n / localisation
- Analytics instrumentation
- The `Topography.jsx` component (used by Workstation; not touched)
