import { useEffect, useRef } from 'react';

interface Shard {
  x: number;
  y: number;
  z: number;
  size: number;
  rotation: number;
  rotationSpeed: number;
  vx: number;
  vy: number;
  vz: number;
  opacity: number;
  hue: number;
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
    const shardCount = 40;

    for (let i = 0; i < shardCount; i++) {
      shards.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        z: Math.random() * 1000,
        size: Math.random() * 80 + 40,
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.02,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        vz: Math.random() * 0.5 + 0.2,
        opacity: Math.random() * 0.3 + 0.1,
        hue: Math.random() * 60 + 170
      });
    }

    let animationId: number;
    let mouseX = canvas.width / 2;
    let mouseY = canvas.height / 2;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    canvas.addEventListener('mousemove', handleMouseMove);

    const animate = () => {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.15)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      shards.sort((a, b) => a.z - b.z);

      shards.forEach(shard => {
        const dx = mouseX - shard.x;
        const dy = mouseY - shard.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 200) {
          const force = (200 - distance) / 200;
          shard.vx -= (dx / distance) * force * 0.1;
          shard.vy -= (dy / distance) * force * 0.1;
        }

        shard.x += shard.vx;
        shard.y += shard.vy;
        shard.z -= shard.vz;
        shard.rotation += shard.rotationSpeed;

        if (shard.z < 0) {
          shard.z = 1000;
          shard.x = Math.random() * canvas.width;
          shard.y = Math.random() * canvas.height;
        }

        if (shard.x < -100) shard.x = canvas.width + 100;
        if (shard.x > canvas.width + 100) shard.x = -100;
        if (shard.y < -100) shard.y = canvas.height + 100;
        if (shard.y > canvas.height + 100) shard.y = -100;

        const scale = 1 - shard.z / 1000;
        const size = shard.size * scale;
        const alpha = shard.opacity * scale;

        ctx.save();
        ctx.translate(shard.x, shard.y);
        ctx.rotate(shard.rotation);
        ctx.globalAlpha = alpha;

        ctx.beginPath();
        ctx.moveTo(0, -size / 2);
        ctx.lineTo(size / 3, size / 2);
        ctx.lineTo(-size / 3, size / 2);
        ctx.closePath();

        const gradient = ctx.createLinearGradient(0, -size / 2, 0, size / 2);
        gradient.addColorStop(0, `hsla(${shard.hue}, 70%, 60%, ${alpha})`);
        gradient.addColorStop(1, `hsla(${shard.hue + 20}, 65%, 50%, ${alpha * 0.5})`);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.strokeStyle = `hsla(${shard.hue}, 75%, 65%, ${alpha * 0.8})`;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.restore();
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}
