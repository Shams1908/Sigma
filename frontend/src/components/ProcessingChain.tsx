import { motion } from 'framer-motion';

interface ProcessingChainProps {
  stages: {
    name: string;
    status: 'completed' | 'running' | 'pending' | 'failed';
    method?: string;
  }[];
}

export default function ProcessingChain({ stages }: ProcessingChainProps) {
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-emerald-500';
      case 'running': return 'bg-cyan-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'running': return '⟳';
      case 'failed': return '✗';
      default: return '○';
    }
  };

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 hover:border-cyan-900 transition-colors duration-300">
      <div className="text-cyan-500 font-mono text-xs tracking-wider mb-6 uppercase">PROCESSING CHAIN</div>
      
      <div className="flex items-center justify-between">
        {stages.map((stage, index) => (
          <div key={index} className="flex items-center">
            <div className="flex flex-col items-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`w-10 h-10 rounded-full ${getStatusColor(stage.status)} flex items-center justify-center text-white font-bold text-sm shadow-lg`}
              >
                {getStatusIcon(stage.status)}
              </motion.div>
              <div className="mt-2 text-center">
                <div className="text-slate-300 text-xs font-mono">{stage.name}</div>
                {stage.method && (
                  <div className="text-gray-500 text-[10px] font-mono mt-0.5">{stage.method}</div>
                )}
              </div>
            </div>
            
            {index < stages.length - 1 && (
              <div className="w-12 h-0.5 bg-gray-700 mx-2 relative">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ delay: index * 0.1 + 0.05, duration: 0.3 }}
                  className={`h-full ${
                    stages[index + 1].status !== 'pending' ? 'bg-cyan-500' : 'bg-gray-700'
                  }`}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
