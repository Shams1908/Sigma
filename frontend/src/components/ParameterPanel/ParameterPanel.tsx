import type { CSSProperties } from 'react';
import type { EstimatedParameters } from '../../types';

type ParameterPanelProps = {
  parameters?: EstimatedParameters | null;
  className?: string;
  style?: CSSProperties;
};

interface FormattedParam {
  label: string;
  value: string;
  unit: string;
}

function formatSnr(snr: number): { value: string; unit: string } {
  return {
    value: `${snr >= 0 ? '+' : ''}${snr.toFixed(1)}`,
    unit: 'dB',
  };
}

function formatBandwidth(hz: number): { value: string; unit: string } {
  if (Math.abs(hz) >= 1e6) {
    return { value: (hz / 1e6).toFixed(2), unit: 'MHz' };
  }
  if (Math.abs(hz) >= 1e3) {
    return { value: (hz / 1e3).toFixed(1), unit: 'kHz' };
  }
  return { value: hz.toFixed(0), unit: 'Hz' };
}

function formatSymbolRate(sps: number): { value: string; unit: string } {
  if (Math.abs(sps) >= 1e6) {
    return { value: (sps / 1e6).toFixed(2), unit: 'MSym/s' };
  }
  if (Math.abs(sps) >= 1e3) {
    return { value: (sps / 1e3).toFixed(1), unit: 'kSym/s' };
  }
  return { value: sps.toFixed(0), unit: 'Sym/s' };
}

function formatCarrierOffset(hz: number): { value: string; unit: string } {
  const sign = hz > 0 ? '+' : hz < 0 ? '-' : '';
  const absHz = Math.abs(hz);
  if (absHz >= 1e6) {
    return { value: `${sign}${(absHz / 1e6).toFixed(2)}`, unit: 'MHz' };
  }
  if (absHz >= 1e3) {
    return { value: `${sign}${(absHz / 1e3).toFixed(2)}`, unit: 'kHz' };
  }
  return { value: `${sign}${absHz.toFixed(0)}`, unit: 'Hz' };
}

export function ParameterPanel({
  parameters,
  className = '',
  style,
}: ParameterPanelProps) {
  const items: FormattedParam[] = parameters
    ? [
        { label: 'SNR', ...formatSnr(parameters.snr) },
        { label: 'BANDWIDTH', ...formatBandwidth(parameters.bandwidth) },
        { label: 'SYMBOL RATE', ...formatSymbolRate(parameters.symbolRate) },
        { label: 'CARRIER OFFSET', ...formatCarrierOffset(parameters.carrierOffset) },
      ]
    : [
        { label: 'SNR', value: '--', unit: 'dB' },
        { label: 'BANDWIDTH', value: '--', unit: 'kHz' },
        { label: 'SYMBOL RATE', value: '--', unit: 'kSym/s' },
        { label: 'CARRIER OFFSET', value: '--', unit: 'kHz' },
      ];

  return (
    <section
      className={`relative flex min-h-0 flex-col border border-grid bg-panel ${className}`}
      style={style}
    >
      <header className="flex shrink-0 items-center justify-between border-b border-grid px-3 py-2">
        <h2 className="font-sans text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">
          PARAMETERS
        </h2>
        <span className="font-sans text-[9px] tracking-wider text-muted">ESTIMATED</span>
      </header>

      <div className="flex min-h-0 flex-1 flex-col justify-center p-3">
        <div className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-2.5">
          {items.map((item) => (
            <div key={item.label} className="contents">
              <span className="font-sans text-[10px] font-medium uppercase tracking-[0.16em] text-muted">
                {item.label}
              </span>
              <div className="flex items-baseline justify-end gap-1.5 font-mono tabular-nums">
                <span className="text-sm font-semibold tracking-wider text-primary">
                  {item.value}
                </span>
                <span className="w-12 text-left text-[10px] tracking-wider text-muted">
                  {item.unit}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
