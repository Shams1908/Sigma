import { useEffect, useRef, useMemo } from 'react';

interface ConstellationPoint {
  i: number;
  q: number;
}

interface ConstellationViewerProps {
  data: ConstellationPoint[];
}

export default function ConstellationViewer({ data }: ConstellationViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { width, height, scale, extent } = useMemo(() => {
    if (!data.length) return { width: 400, height: 400, scale: (v: number) => v, extent: 1 };

    const w = 400;
    const h = 400;
    const margin = 40;
    const innerSize = Math.min(w, h) - 2 * margin;

    const iValues = data.map(d => d.i);
    const qValues = data.map(d => d.q);
    const maxI = Math.max(...iValues.map(Math.abs));
    const maxQ = Math.max(...qValues.map(Math.abs));
    const maxExtent = Math.max(maxI, maxQ) * 1.25;

    const scale = (v: number) => margin + innerSize / 2 + (v / maxExtent) * (innerSize / 2);

    return { width: w, height: h, scale, extent: maxExtent };
  }, [data]);

  useEffect(() => {
    const circles = svgRef.current?.querySelectorAll('.constellation-point');
    circles?.forEach((circle, i) => {
      const delay = Math.floor(i / 25) * 12;
      (circle as SVGElement).style.animationDelay = `${delay}ms`;
    });
  }, [data]);

  if (!data.length) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-[#0A0A0A] rounded-2xl border border-[#222222]">
        <p className="text-slate-500 font-mono text-sm">NO CONSTELLATION DATA</p>
      </div>
    );
  }

  const centerX = width / 2;
  const centerY = height / 2;

  return (
    <div ref={containerRef} className="w-full h-full bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 relative hover:border-sigma-teal-900 transition-colors duration-300">
      <div className="absolute top-4 left-4 z-10">
        <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">CONSTELLATION</div>
      </div>
      <div className="absolute top-4 right-4 z-10 bg-[#0A0A0A] border border-sigma-teal/30 px-3 py-1.5 rounded">
        <div className="text-xs font-mono text-slate-400 uppercase">POINTS</div>
        <div className="text-sm font-mono text-sigma-teal">{data.length}</div>
      </div>
      <svg ref={svgRef} width={width} height={height} className="w-full h-auto">
        <defs>
          <style>
            {`
              @keyframes constellation-emerge {
                0% {
                  opacity: 0;
                  transform: scale(0);
                }
                100% {
                  opacity: 0.7;
                  transform: scale(1);
                }
              }
              .constellation-point {
                animation: constellation-emerge 0.4s ease-out forwards;
                transform-origin: center;
                opacity: 0;
              }
            `}
          </style>
        </defs>

        <line
          x1={40}
          y1={centerY}
          x2={width - 40}
          y2={centerY}
          stroke="rgb(71, 85, 105)"
          strokeWidth="1"
        />
        <line
          x1={centerX}
          y1={40}
          x2={centerX}
          y2={height - 40}
          stroke="rgb(71, 85, 105)"
          strokeWidth="1"
        />

        <text
          x={width - 30}
          y={centerY - 8}
          className="text-xs font-mono fill-slate-500"
          textAnchor="middle"
        >
          I
        </text>
        <text
          x={centerX + 8}
          y={50}
          className="text-xs font-mono fill-slate-500"
          textAnchor="middle"
        >
          Q
        </text>

        {[-0.5, 0.5].map(val => {
          const pos = scale(val * extent);
          return (
            <g key={val}>
              <line
                x1={pos}
                y1={centerY - 3}
                x2={pos}
                y2={centerY + 3}
                stroke="rgb(71, 85, 105)"
                strokeWidth="1"
              />
              <line
                x1={centerX - 3}
                y1={scale(-val * extent)}
                x2={centerX + 3}
                y2={scale(-val * extent)}
                stroke="rgb(71, 85, 105)"
                strokeWidth="1"
              />
            </g>
          );
        })}

        {data.map((point, i) => (
          <circle
            key={i}
            className="constellation-point"
            cx={scale(point.i)}
            cy={scale(-point.q)}
            r="2.5"
            fill="rgb(20, 184, 166)"
            fillOpacity="0.7"
          />
        ))}
      </svg>
    </div>
  );
}
