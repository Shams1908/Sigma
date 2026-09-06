import { useEffect, useRef } from 'react';

interface Shard {
  x: number;
  y: number;
  size: number;
  rotation: number;
  rotationSpeed: number;
  vx: number;
  vy: number;
  opacity: number;
}

export default function AeroShards() {
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

    const shards: Shard[] = [];
    const shardCount = 25;

    for (let i = 0; i < shardCount; i++) {
      shards.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 60 + 30,
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.02,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        opacity: Math.random() * 0.15 + 0.05
      });
    }

    let animationId: number;

    const animate = () => {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      shards.forEach(shard => {
        shard.x += shard.vx;
        shard.y += shard.vy;
        shard.rotation += shard.rotationSpeed;

        if (shard.x < -100) shard.x = canvas.width + 100;
        if (shard.x > canvas.width + 100) shard.x = -100;
        if (shard.y < -100) shard.y = canvas.height + 100;
        if (shard.y > canvas.height + 100) shard.y = -100;

        ctx.save();
        ctx.translate(shard.x, shard.y);
        ctx.rotate(shard.rotation);

        ctx.beginPath();
        ctx.moveTo(0, -shard.size / 2);
        ctx.lineTo(shard.size / 3, shard.size / 2);
        ctx.lineTo(-shard.size / 3, shard.size / 2);
        ctx.closePath();

        const gradient = ctx.createLinearGradient(0, -shard.size / 2, 0, shard.size / 2);
        gradient.addColorStop(0, `rgba(59, 130, 246, ${shard.opacity})`);
        gradient.addColorStop(1, `rgba(20, 184, 166, ${shard.opacity * 0.5})`);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.strokeStyle = `rgba(59, 130, 246, ${shard.opacity * 0.8})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.restore();
      });

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
