import { motion } from 'framer-motion';

interface AnalysisItem {
  id: string;
  fileName: string;
  timestamp: string;
  status: 'completed' | 'failed' | 'pending';
  snr?: number;
}

interface RecentAnalysisSidebarProps {
  analyses: AnalysisItem[];
  currentId: string | null;
  onSelect: (id: string) => void;
}

export default function RecentAnalysisSidebar({ analyses, currentId, onSelect }: RecentAnalysisSidebarProps) {
  
  if (analyses.length === 0) {
    return (
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 h-full">
        <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">RECENT</div>
        <div className="flex items-center justify-center h-32">
          <p className="text-gray-500 text-xs">No analyses yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 h-full overflow-hidden flex flex-col">
      <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">RECENT</div>
      
      <div className="flex-1 overflow-y-auto space-y-2 pr-2">
        {analyses.slice(0, 10).map((item, index) => (
          <motion.button
            key={item.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => onSelect(item.id)}
            className={`w-full text-left p-3 rounded-lg transition-all duration-200 ${
              currentId === item.id
                ? 'bg-cyan-950/40 border border-cyan-900'
                : 'bg-[#1a1a1a] border border-[#222222] hover:border-cyan-900'
            }`}
          >
            <div className="flex items-start justify-between mb-1">
              <div className="text-slate-300 text-xs font-mono truncate flex-1">
                {item.fileName}
              </div>
              <div className={`w-2 h-2 rounded-full ml-2 mt-1 ${
                item.status === 'completed' ? 'bg-emerald-500' :
                item.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
              }`} />
            </div>
            <div className="flex items-center justify-between">
              <div className="text-gray-500 text-[10px] font-mono">
                {new Date(item.timestamp).toLocaleTimeString()}
              </div>
              {item.snr !== undefined && (
                <div className="text-cyan-400 text-[10px] font-mono">
                  {item.snr.toFixed(1)} dB
                </div>
              )}
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
