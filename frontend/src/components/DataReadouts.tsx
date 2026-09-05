interface DataReadoutsProps {
  carrierFrequency: number;
  sampleRate: number;
  bandwidth: number;
  snr: number;
  symbolRate: number;
}

export default function DataReadouts({
  carrierFrequency,
  sampleRate,
  bandwidth,
  snr,
  symbolRate
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

  const readouts = [
    { label: 'CARRIER FREQUENCY', value: formatFrequency(carrierFrequency), unit: '' },
    { label: 'SAMPLE RATE', value: formatSampleRate(sampleRate), unit: '' },
    { label: 'BANDWIDTH', value: formatFrequency(bandwidth), unit: '' },
    { label: 'SNR', value: snr.toFixed(1), unit: 'dB' },
    { label: 'SYMBOL RATE', value: formatSampleRate(symbolRate), unit: '' }
  ];

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6">
      <div className="text-teal-400 font-mono text-xs tracking-wider mb-4">SIGNAL PARAMETERS</div>
      <div className="space-y-4">
        {readouts.map((item, i) => (
          <div key={i} className="flex justify-between items-baseline">
            <div className="text-slate-400 font-mono text-xs">{item.label}</div>
            <div className="text-slate-200 font-mono text-sm">
              {item.value} {item.unit && <span className="text-slate-500">{item.unit}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
