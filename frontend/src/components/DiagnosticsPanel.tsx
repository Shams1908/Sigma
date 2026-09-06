import { motion } from 'framer-motion';

interface DiagnosticsPanelProps {
  evmRms: number | null;
  timingErrorRms: number | null;
  syncLocked: boolean;
  demodLocked: boolean;
  fecValid: boolean;
  snr: number;
  carrierOffset: number;
  failureReason?: string;
}

export default function DiagnosticsPanel({
  evmRms,
  timingErrorRms,
  syncLocked,
  demodLocked,
  fecValid,
  snr,
  carrierOffset,
  failureReason
}: DiagnosticsPanelProps) {
  
  const StatusIndicator = ({ locked, label }: { locked: boolean; label: string }) => (
    <div className="flex items-center justify-between py-2">
      <span className="text-gray-400 text-xs font-mono">{label}</span>
      <div className="flex items-center gap-2">
        <motion.div
          className={`w-2 h-2 rounded-full ${locked ? 'bg-emerald-500' : 'bg-red-500'}`}
          animate={{
            boxShadow: locked
              ? ['0 0 0 0 rgba(16, 185, 129, 0.7)', '0 0 8px 2px rgba(16, 185, 129, 0)', '0 0 0 0 rgba(16, 185, 129, 0.7)']
              : ['0 0 0 0 rgba(239, 68, 68, 0.7)', '0 0 6px 1px rgba(239, 68, 68, 0)', '0 0 0 0 rgba(239, 68, 68, 0.7)']
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <span className={`text-xs font-mono ${locked ? 'text-emerald-400' : 'text-red-400'}`}>
          {locked ? 'LOCKED' : 'UNLOCKED'}
        </span>
      </div>
    </div>
  );

  const MetricRow = ({ label, value, unit }: { label: string; value: number | null; unit?: string }) => (
    <div className="flex items-center justify-between py-2 border-b border-[#1a1a1a]">
      <span className="text-gray-400 text-xs font-mono">{label}</span>
      <span className="text-slate-200 text-sm font-mono">
        {value !== null ? value.toFixed(2) : '—'} {unit && <span className="text-gray-500">{unit}</span>}
      </span>
    </div>
  );

  const isAvailable = evmRms !== null;

  if (!isAvailable) {
    return (
      <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col">
        <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">DIAGNOSTICS</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3">
            <div className="text-4xl">🔧</div>
            <p className="text-gray-400 text-sm">Analysis Pending</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 h-full flex flex-col hover:border-cyan-900 transition-colors duration-300">
      <div className="text-cyan-500 font-mono text-xs tracking-wider mb-4 uppercase">DIAGNOSTICS</div>
      
      <div className="space-y-1 mb-4">
        <StatusIndicator locked={syncLocked} label="SYNCHRONIZATION" />
        <StatusIndicator locked={demodLocked} label="DEMODULATION" />
        <StatusIndicator locked={fecValid} label="FEC VALIDATION" />
      </div>

      <div className="space-y-0 border-t border-[#222222] pt-4">
        <MetricRow label="EVM RMS" value={evmRms} unit="%" />
        <MetricRow label="TIMING ERROR" value={timingErrorRms} unit="μs" />
        <MetricRow label="SNR" value={snr} unit="dB" />
        <MetricRow label="CARRIER OFFSET" value={carrierOffset} unit="Hz" />
      </div>

      {failureReason && (
        <div className="mt-4 pt-4 border-t border-[#222222]">
          <div className="text-red-400 text-[10px] font-mono uppercase tracking-wider mb-2">FAILURE REASON</div>
          <div className="text-gray-400 text-xs leading-relaxed bg-red-950/20 border border-red-900/30 rounded p-2">
            {failureReason}
          </div>
        </div>
      )}
    </div>
  );
}
