import { useState } from 'react';
import { motion } from 'framer-motion';

interface BitstreamViewerProps {
  bits: string;
  totalBits: number;
  entropy: number;
  onesRatio: number;
  frameBoundaries?: number[];
  headerEnd?: number;
}

type ViewMode = 'binary' | 'hex' | 'bytes';

export default function BitstreamViewer({
  bits,
  totalBits,
  entropy,
  onesRatio,
  frameBoundaries = [],
  headerEnd
}: BitstreamViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('binary');
  const [maxDisplay, setMaxDisplay] = useState(512);

  const binaryToHex = (binary: string): string => {
    const paddedBinary = binary.padEnd(Math.ceil(binary.length / 4) * 4, '0');
    let hex = '';
    for (let i = 0; i < paddedBinary.length; i += 4) {
      const chunk = paddedBinary.slice(i, i + 4);
      hex += parseInt(chunk, 2).toString(16).toUpperCase();
    }
    return hex;
  };

  const binaryToBytes = (binary: string): string => {
    const paddedBinary = binary.padEnd(Math.ceil(binary.length / 8) * 8, '0');
    const bytes: string[] = [];
    for (let i = 0; i < paddedBinary.length; i += 8) {
      const byte = paddedBinary.slice(i, i + 8);
      bytes.push(byte);
    }
    return bytes.join(' ');
  };

  const formatBinaryWithSpaces = (binary: string, groupSize: number = 8): string => {
    const groups: string[] = [];
    for (let i = 0; i < binary.length; i += groupSize) {
      groups.push(binary.slice(i, i + groupSize));
    }
    return groups.join(' ');
  };

  const getDisplayContent = (): string => {
    const displayBits = bits.slice(0, maxDisplay);
    switch (viewMode) {
      case 'hex':
        return binaryToHex(displayBits);
      case 'bytes':
        return binaryToBytes(displayBits);
      case 'binary':
      default:
        return formatBinaryWithSpaces(displayBits);
    }
  };

  const getGroupSize = (): number => {
    switch (viewMode) {
      case 'hex':
        return 4;
      case 'bytes':
        return 16;
      case 'binary':
      default:
        return 32;
    }
  };

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">Bitstream View</h3>
          <p className="text-xs text-gray-500 font-mono">Decoded Output</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('binary')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'binary'
                ? 'bg-sigma-teal text-black'
                : 'bg-[#111111] text-gray-400 hover:text-white border border-[#222222]'
            }`}
          >
            Binary
          </button>
          <button
            onClick={() => setViewMode('hex')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'hex'
                ? 'bg-sigma-teal text-black'
                : 'bg-[#111111] text-gray-400 hover:text-white border border-[#222222]'
            }`}
          >
            Hex
          </button>
          <button
            onClick={() => setViewMode('bytes')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              viewMode === 'bytes'
                ? 'bg-sigma-teal text-black'
                : 'bg-[#111111] text-gray-400 hover:text-white border border-[#222222]'
            }`}
          >
            Bytes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-[#111111] border border-[#222222] rounded-lg p-3">
          <div className="text-xs text-gray-500 mb-1">Total Bits</div>
          <div className="text-lg font-bold text-white font-mono">{totalBits.toLocaleString()}</div>
        </div>
        <div className="bg-[#111111] border border-[#222222] rounded-lg p-3">
          <div className="text-xs text-gray-500 mb-1">Entropy</div>
          <div className="text-lg font-bold text-cyan-400 font-mono">{entropy.toFixed(3)}</div>
        </div>
        <div className="bg-[#111111] border border-[#222222] rounded-lg p-3">
          <div className="text-xs text-gray-500 mb-1">Ones Ratio</div>
          <div className="text-lg font-bold text-sigma-teal font-mono">{(onesRatio * 100).toFixed(1)}%</div>
        </div>
      </div>

      {headerEnd && (
        <div className="mb-3 p-3 bg-[#111111] border border-sigma-teal/30 rounded-lg">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-sigma-teal rounded-full animate-pulse" />
            <span className="text-xs text-gray-400">
              Header detected: bits 0-{headerEnd}
            </span>
          </div>
        </div>
      )}

      <div className="flex-1 bg-[#111111] border border-[#222222] rounded-xl p-4 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <pre className="text-xs font-mono text-green-400 leading-relaxed whitespace-pre-wrap break-all">
            {getDisplayContent()}
          </pre>
        </div>

        {bits.length > maxDisplay && (
          <motion.button
            onClick={() => setMaxDisplay(prev => prev + 512)}
            className="mt-3 w-full px-4 py-2 bg-[#1a1a1a] hover:bg-[#222222] border border-[#333333] hover:border-sigma-teal/50 text-xs font-semibold text-white rounded-lg transition-all"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Load More ({bits.length - maxDisplay} bits remaining)
          </motion.button>
        )}
      </div>

      {frameBoundaries.length > 0 && (
        <div className="mt-3 p-3 bg-[#111111] border border-[#222222] rounded-lg">
          <div className="text-xs text-gray-500 mb-2">Frame Boundaries Detected</div>
          <div className="flex flex-wrap gap-2">
            {frameBoundaries.slice(0, 10).map((boundary, idx) => (
              <div
                key={idx}
                className="px-2 py-1 bg-[#1a1a1a] border border-[#333333] rounded text-xs font-mono text-sigma-teal"
              >
                @{boundary}
              </div>
            ))}
            {frameBoundaries.length > 10 && (
              <div className="px-2 py-1 text-xs text-gray-600">
                +{frameBoundaries.length - 10} more
              </div>
            )}
          </div>
        </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #0a0a0a;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #333333;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #0dd9c5;
        }
      `}</style>
    </div>
  );
}
