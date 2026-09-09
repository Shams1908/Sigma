import { useEffect, useRef } from 'react';

export default function Topography() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const gridSize = 60;
    const cols = Math.ceil(canvas.width / gridSize) + 1;
    const rows = Math.ceil(canvas.height / gridSize) + 1;

    let time = 0;
    let animationId: number;

    const animate = () => {
      ctx.fillStyle = 'rgb(15, 23, 42)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      time += 0.01;

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const x = j * gridSize;
          const y = i * gridSize;

          const wave1 = Math.sin(x * 0.01 + time) * 10;
          const wave2 = Math.cos(y * 0.01 + time * 0.8) * 10;
          const elevation = wave1 + wave2;

          const opacity = (elevation + 20) / 40;

          ctx.beginPath();
          ctx.arc(x, y + elevation, 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(20, 184, 166, ${Math.max(0.1, Math.min(0.4, opacity))})`;
          ctx.fill();

          if (j < cols - 1) {
            const nextX = (j + 1) * gridSize;
            const nextWave1 = Math.sin(nextX * 0.01 + time) * 10;
            const nextWave2 = Math.cos(y * 0.01 + time * 0.8) * 10;
            const nextElevation = nextWave1 + nextWave2;

            ctx.beginPath();
            ctx.moveTo(x, y + elevation);
            ctx.lineTo(nextX, y + nextElevation);
            ctx.strokeStyle = `rgba(20, 184, 166, 0.15)`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }

          if (i < rows - 1) {
            const nextY = (i + 1) * gridSize;
            const nextWave1 = Math.sin(x * 0.01 + time) * 10;
            const nextWave2 = Math.cos(nextY * 0.01 + time * 0.8) * 10;
            const nextElevation = nextWave1 + nextWave2;

            ctx.beginPath();
            ctx.moveTo(x, y + elevation);
            ctx.lineTo(x, nextY + nextElevation);
            ctx.strokeStyle = `rgba(20, 184, 166, 0.15)`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
