import type {
  AnalyzedSignal,
  ConstellationPoint,
  EstimatedParameters,
  HypothesisCandidate,
  PipelineStage,
  SignalMetadata,
  SpectrumPoint,
} from '../types';

const MOCK_SEED = 0x51474d41; // "SGMA"

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function buildMetadata(): SignalMetadata {
  const sampleRate = 2.4e6;
  const duration = 0.512;
  const bytesPerSample = 4; // interleaved int16 I/Q

  return {
    fileName: 'capture_042.iq',
    sampleRate,
    centerFrequency: 915e6,
    duration,
    fileSize: Math.round(sampleRate * duration * bytesPerSample),
    ingestionTime: '2026-08-31T08:12:00.000Z',
  };
}

function buildParameters(): EstimatedParameters {
  return {
    snr: 18.6,
    bandwidth: 2.05e5,
    carrierOffset: 1.875e4,
    symbolRate: 2e5,
  };
}

function buildSpectrum(
  rng: () => number,
  sampleRate: number,
  parameters: EstimatedParameters,
): SpectrumPoint[] {
  const bins = 512;
  const noiseFloorDb = -91;
  const peakDb = noiseFloorDb + parameters.snr + 12;
  const sigmaHz = parameters.bandwidth / 2.35;

  const points: SpectrumPoint[] = [];
  for (let i = 0; i < bins; i += 1) {
    const frequency = -sampleRate / 2 + (i / bins) * sampleRate;
    const detune = frequency - parameters.carrierOffset;
    const peakLinear = 10 ** (peakDb / 10) * Math.exp(-(detune * detune) / (2 * sigmaHz * sigmaHz));
    const noiseLinear = 10 ** ((noiseFloorDb + gaussian(rng) * 1.35) / 10);
    const magnitudeDb = 10 * Math.log10(peakLinear + Math.max(noiseLinear, 1e-18));
    points.push({ frequency, magnitudeDb });
  }
  return points;
}

function buildWaterfall(
  rng: () => number,
  sampleRate: number,
  parameters: EstimatedParameters,
): number[][] {
  const timeSlices = 64;
  const frequencyBins = 256;
  const noiseFloorDb = -88;
  const peakDb = noiseFloorDb + parameters.snr + 10;
  const sigmaHz = parameters.bandwidth / 2.2;

  const rows: number[][] = [];
  for (let t = 0; t < timeSlices; t += 1) {
    const driftHz = Math.sin((t / timeSlices) * Math.PI * 2) * 8e3;
    const fade = 0.82 + 0.18 * Math.sin((t / timeSlices) * Math.PI * 4 + 0.4);
    const row: number[] = [];
    for (let f = 0; f < frequencyBins; f += 1) {
      const frequency = -sampleRate / 2 + (f / frequencyBins) * sampleRate;
      const detune = frequency - (parameters.carrierOffset + driftHz);
      const peakLinear =
        10 ** (peakDb / 10) * fade * Math.exp(-(detune * detune) / (2 * sigmaHz * sigmaHz));
      const noiseLinear = 10 ** ((noiseFloorDb + gaussian(rng) * 1.6) / 10);
      row.push(10 * Math.log10(peakLinear + Math.max(noiseLinear, 1e-18)));
    }
    rows.push(row);
  }
  return rows;
}

function buildConstellation(rng: () => number): ConstellationPoint[] {
  const scale = 1 / Math.SQRT2;
  const sigma = 0.09;
  const targets: Array<[number, number]> = [
    [scale, scale],
    [scale, -scale],
    [-scale, scale],
    [-scale, -scale],
  ];

  const points: ConstellationPoint[] = [];
  for (let n = 0; n < 200; n += 1) {
    const [i0, q0] = targets[n % 4];
    points.push({
      i: i0 + gaussian(rng) * sigma,
      q: q0 + gaussian(rng) * sigma,
    });
  }
  return points;
}

function buildHypotheses(symbolRate: number): HypothesisCandidate[] {
  return [
    {
      id: 'hyp-qpsk',
      modulation: 'QPSK',
      symbolRate,
      confidenceScore: 0.942,
      details: 'Gray-mapped QPSK; tight 4-cluster I/Q occupancy.',
      status: 'success',
    },
    {
      id: 'hyp-bpsk',
      modulation: 'BPSK',
      symbolRate,
      confidenceScore: 0.038,
      details: 'Residual 2-cluster energy after QPSK decision.',
      status: 'pending',
    },
    {
      id: 'hyp-8psk',
      modulation: '8-PSK',
      symbolRate,
      confidenceScore: 0.012,
      details: 'No stable 8-ary ring structure detected.',
      status: 'failed',
    },
    {
      id: 'hyp-16qam',
      modulation: '16-QAM',
      symbolRate,
      confidenceScore: 0.008,
      details: 'Amplitude histogram inconsistent with square 16-QAM.',
      status: 'failed',
    },
  ];
}

function buildPipeline(): PipelineStage[] {
  return [
    { label: 'INGESTED', status: 'complete' },
    { label: 'DSP ANALYSIS', status: 'complete' },
    { label: 'MODULATION', status: 'complete' },
    { label: 'DEMODULATION', status: 'active' },
    { label: 'FEC', status: 'pending' },
    { label: 'VALIDATION', status: 'pending' },
  ];
}

export function generateMockSignal(): AnalyzedSignal {
  const rng = createRng(MOCK_SEED);
  const metadata = buildMetadata();
  const parameters = buildParameters();

  return {
    metadata,
    spectrum: buildSpectrum(rng, metadata.sampleRate, parameters),
    waterfall: buildWaterfall(rng, metadata.sampleRate, parameters),
    parameters,
    constellation: buildConstellation(rng),
    hypotheses: buildHypotheses(parameters.symbolRate),
    pipeline: buildPipeline(),
  };
}
