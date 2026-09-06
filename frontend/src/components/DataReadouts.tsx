interface DataReadoutsProps {
  carrierFrequency: number;
  sampleRate: number;
  bandwidth: number;
  snr: number;
  symbolRate: number;
  confidence?: {
    bandwidth?: number;
    snr?: number;
    symbolRate?: number;
  };
}

export default function DataReadouts({
  carrierFrequency,
  sampleRate,
  bandwidth,
  snr,
  symbolRate,
  confidence
}: DataReadoutsProps) {
  const formatFrequency = (hz: number): string => {
    if (hz >= 1e9) return `${(hz / 1e9).toFixed(3)} GHz`;
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
    if (hz >= 1e3) return `${(hz / 1e3).toFixed(3)} kHz`;
    return `${hz.toFixed(0)} Hz`;
  };

  const formatSampleRate = (rate: number): string => {
    if (rate >= 1e6) return `${(rate / 1e6).toFixed(2)} Msps`;
    if (rate >= 1e3) return `${(rate / 1e3).toFixed(2)} ksps`;
    return `${rate.toFixed(0)} sps`;
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-emerald-500';
    if (conf >= 0.6) return 'bg-cyan-500';
    if (conf >= 0.4) return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  const readouts = [
    { label: 'CARRIER FREQUENCY', value: formatFrequency(carrierFrequency), unit: '', conf: 1.0 },
    { label: 'SAMPLE RATE', value: formatSampleRate(sampleRate), unit: '', conf: 1.0 },
    { label: 'BANDWIDTH', value: formatFrequency(bandwidth), unit: '', conf: confidence?.bandwidth || 0.75 },
    { label: 'SNR', value: snr.toFixed(1), unit: 'dB', conf: confidence?.snr || 0.85 },
    { label: 'SYMBOL RATE', value: formatSampleRate(symbolRate), unit: '', conf: confidence?.symbolRate || 0.70 }
  ];

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-6 hover:border-sigma-teal-900 transition-colors duration-300">
      <div className="text-sigma-teal font-mono text-xs tracking-wider mb-4 uppercase">SIGNAL PARAMETERS</div>
      <div className="space-y-4">
        {readouts.map((item, i) => (
          <div key={i} className="space-y-1">
            <div className="flex justify-between items-baseline">
              <div className="text-slate-400 font-mono text-xs">{item.label}</div>
              <div className="text-slate-200 font-mono text-sm">
                {item.value} {item.unit && <span className="text-slate-500">{item.unit}</span>}
              </div>
            </div>
            <div className="w-full h-1 bg-[#1a1a1a] rounded-full overflow-hidden">
              <div 
                className={`h-full ${getConfidenceColor(item.conf)} transition-all duration-500`}
                style={{ width: `${item.conf * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
