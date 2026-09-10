/**
 * DemoBanner — visible indicator that a UI section is populated with
 * SYNTHETIC DEMO DATA, not real backend analysis results.
 *
 * Drop this anywhere a section uses values from demoData.ts.
 * It renders nothing when the `active` prop is false, so the same
 * component can be placed unconditionally and simply toggled off
 * once real data is available.
 */

interface DemoBannerProps {
  /** When false the banner renders nothing (real data is present). */
  active?: boolean;
  /** Optional override for the message text. */
  message?: string;
  /** 'bar' = full-width bar across top of section (default)
   *  'badge' = small inline pill */
  variant?: 'bar' | 'badge';
}

export default function DemoBanner({
  active = true,
  message = 'SYNTHETIC DEMO DATA — backend decoder integration pending',
  variant = 'bar',
}: DemoBannerProps) {
  if (!active) return null;

  if (variant === 'badge') {
    return (
      <span
        className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono font-bold
                   bg-amber-500/15 border border-amber-500/40 text-amber-400 select-none"
        title={message}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
        SYNTHETIC DEMO DATA
      </span>
    );
  }

  // 'bar' variant — full-width strip
  return (
    <div
      className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg mb-3
                 bg-amber-500/10 border border-amber-500/30 select-none"
      role="status"
      aria-label="Synthetic demo data notice"
    >
      <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse flex-shrink-0" />
      <span className="text-[11px] font-mono text-amber-400 leading-none">
        {message}
      </span>
    </div>
  );
}
