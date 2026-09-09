export function findCanvasInComponent(containerRef: HTMLElement | null, searchDepth: number = 5): HTMLCanvasElement | null {
  if (!containerRef) return null;

  const canvases = containerRef.querySelectorAll('canvas');
  if (canvases.length > 0) {
    return canvases[0] as HTMLCanvasElement;
  }

  return null;
}

export async function captureComponentCanvas(element: HTMLElement | null): Promise<string | null> {
  const canvas = findCanvasInComponent(element);
  if (!canvas) {
    console.warn('No canvas found in component');
    return null;
  }

  try {
    return canvas.toDataURL('image/jpeg', 0.9);
  } catch (error) {
    console.error('Failed to capture canvas:', error);
    return null;
  }
}
