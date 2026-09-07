import { useRef, useState, useEffect } from 'react';

interface UseScanRevealOptions {
  /** IntersectionObserver threshold — default 0.2 (20% visible) */
  threshold?: number;
  /** Stagger delay in milliseconds before the reveal fires */
  delay?: number;
  /** Clip-path transition duration in milliseconds — default 800 */
  duration?: number;
}

interface UseScanRevealReturn {
  ref: React.RefObject<HTMLDivElement>;
  isRevealed: boolean;
  style: React.CSSProperties;
}

/**
 * Triggers a clip-path left→right unmask when the element enters the
 * viewport. Fires once per page load; observer disconnects after first fire.
 *
 * Under prefers-reduced-motion the element is immediately revealed with no
 * animation (duration collapses to 0 and no sweep line is shown).
 */
export function useScanReveal(options?: UseScanRevealOptions): UseScanRevealReturn {
  const ref = useRef<HTMLDivElement>(null);
  const [isRevealed, setIsRevealed] = useState(false);

  const threshold = options?.threshold ?? 0.2;
  const delay     = options?.delay     ?? 0;
  const duration  = options?.duration  ?? 800;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Immediately reveal — no animation — under reduced-motion preference.
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setIsRevealed(true);
      return;
    }

    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          // Fire once only
          observer.unobserve(el);
          if (delay > 0) {
            timeoutId = setTimeout(() => setIsRevealed(true), delay);
          } else {
            setIsRevealed(true);
          }
        }
      },
      { threshold }
    );

    observer.observe(el);

    return () => {
      observer.disconnect();
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Options are read once on mount — intentional empty dep array.

  const style: React.CSSProperties = isRevealed
    ? { clipPath: 'inset(0 0% 0 0)' }
    : {
        clipPath: 'inset(0 100% 0 0)',
        transition: `clip-path ${duration}ms cubic-bezier(0.16, 1, 0.3, 1)`,
      };

  return { ref, isRevealed, style };
}
