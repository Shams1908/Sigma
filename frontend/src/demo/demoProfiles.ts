/**
 * demoProfiles.ts — Frontend-only presentation data layer.
 *
 * IMPLEMENTATION NOTE (for developers):
 * All values in this file are deterministic, hand-authored constants used
 * to populate the analysis UI when real backend results are not yet available.
 * They are NOT the output of real DSP, ML inference, FEC decoding, hypothesis
 * scoring, or any signal-processing algorithm.
 *
 * The profile selected for a given upload is chosen deterministically from
 * the filename and file size — it is NOT a real classification of the IQ data.
 *
 * The UI/API boundary is designed so that when the real backend pipeline
 * produces results, they will replace these values without any UI changes.
 *
 * See: profileSelector.ts for the file → profile mapping logic.
 */

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface DemoConstellationPoint {
  i: number;
  q: number;
}

export interface DemoHypothesis {
  id: string;
  modulation: string;
  symbolRate: number;
  mlConfidence: number;
  validation: { syncPassed: boolean; demodPassed: boolean; fecPassed: boolean };
  isWinner: boolean;
}

export interface DemoDecoderCandidate {
  id: string;
  rank: number;
  modulation: string;
  symbolRate: number;
  fecType: string;
  score: number;
  stages: { sync: boolean; demod: boolean; fec: boolean };
  bitErrors?: number;
}

export interface DemoSignalRegion {
  id: string;
  frequencyStart: number;
  frequencyEnd: number;
  bandwidth: number;
  centerFreq: number;
  snr: number;
  candidateModulation: string;
  confidence: number;
}

export interface DemoProfile {
  id: string;
  label: string;           // human-readable, shown nowhere — for developer reference only

  // Signal parameters
  params: {
    carrierFrequency: number;
    sampleRate: number;
    bandwidth: number;
    snr: number;
    symbolRate: number;
  };
  paramsConfidence: {
    bandwidth: number;
    snr: number;
    symbolRate: number;
  };

  // Processing pipeline
  processingStages: Array<{
    name: string;
    status: 'completed' | 'running' | 'pending' | 'failed';
    method?: string;
  }>;

  // Diagnostics
  diagnostics: {
    evm_rms: number;
    timing_error_rms: number;
    sync_locked: boolean;
    demod_locked: boolean;
    fec_valid: boolean;
    snr: number;
    carrier_offset: number;
  };

  // Hypotheses
  hypotheses: DemoHypothesis[];

  // Decoder candidates
  decoderCandidates: DemoDecoderCandidate[];

  // Signal explorer region
  signalRegion: DemoSignalRegion;

  // Bitstream
  bitstream: {
    bits: string;
    totalBits: number;
    entropy: number;
    onesRatio: number;
    frameBoundaries: number[];
    headerEnd: number;
  };

  // Constellation IQ points (deterministic, display only)
  constellationPoints: DemoConstellationPoint[];
}

// ---------------------------------------------------------------------------
// Constellation point generators
// These produce visually correct cluster shapes for each modulation.
// All values are fixed — no random seed, fully deterministic.
// ---------------------------------------------------------------------------

/** Generate a tight cluster centred at (ci, cq) using a fixed spiral pattern */
function cluster(
  ci: number,
  cq: number,
  count: number,
  spread: number,
  offset: number = 0,
): DemoConstellationPoint[] {
  const pts: DemoConstellationPoint[] = [];
  for (let k = 0; k < count; k++) {
    const angle = (k / count) * 2 * Math.PI + offset;
    const r     = spread * (0.3 + 0.7 * ((k * 7 + 13) % count) / count);
    pts.push({ i: ci + r * Math.cos(angle), q: cq + r * Math.sin(angle) });
  }
  return pts;
}

// BPSK — two clusters on the real axis
const BPSK_POINTS: DemoConstellationPoint[] = [
  ...cluster( 1.0, 0, 120, 0.12, 0.0),
  ...cluster(-1.0, 0, 120, 0.12, 0.5),
];

// QPSK — four clusters at ±1/√2 on both axes
const R2 = 0.7071;
const QPSK_POINTS: DemoConstellationPoint[] = [
  ...cluster( R2,  R2, 100, 0.10, 0.0),
  ...cluster(-R2,  R2, 100, 0.10, 0.4),
  ...cluster(-R2, -R2, 100, 0.10, 0.8),
  ...cluster( R2, -R2, 100, 0.10, 1.2),
];

// 8-PSK — eight clusters evenly spaced on unit circle
const PSK8_POINTS: DemoConstellationPoint[] = Array.from({ length: 8 }, (_, k) => {
  const angle = (k / 8) * 2 * Math.PI;
  return cluster(Math.cos(angle), Math.sin(angle), 70, 0.11, k * 0.3);
}).flat();

// 16-QAM — 4×4 grid from -3 to +3 (normalised to unit avg power)
const QAM_LEVELS = [-3, -1, 1, 3];
const QAM_NORM   = 1 / Math.sqrt(10); // RMS normalisation for 16-QAM
const QAM16_POINTS: DemoConstellationPoint[] = QAM_LEVELS.flatMap((qi, ri) =>
  QAM_LEVELS.flatMap((ii, ci) =>
    cluster(ii * QAM_NORM, qi * QAM_NORM, 50, 0.09, (ri * 4 + ci) * 0.25),
  ),
);

// FSK — represented as two offset real-axis clusters (frequency components)
const FSK_POINTS: DemoConstellationPoint[] = [
  ...cluster( 0.6,  0.3, 130, 0.13, 0.0),
  ...cluster(-0.6, -0.3, 130, 0.13, 0.7),
];

// ---------------------------------------------------------------------------
// Bitstream builders
// Each profile uses a different preamble, header, and payload pattern.
// Structure (display only — no processing):
//   [PREAMBLE 16b] [SYNC 16b] [HEADER 32b] [PAYLOAD N×8b] [FCS 16b]
// ---------------------------------------------------------------------------

function buildBitstream(
  preamble: string,
  sync:     string,
  header:   string,
  payload:  string[],
  fcs:      string,
): { bits: string; totalBits: number; entropy: number; onesRatio: number; frameBoundaries: number[]; headerEnd: number } {
  const bits = preamble + sync + header + payload.join('') + fcs;
  const ones = bits.split('').filter(b => b === '1').length;
  return {
    bits,
    totalBits:       bits.length,
    entropy:         0.92 + (ones / bits.length) * 0.07,  // display value
    onesRatio:       ones / bits.length,
    frameBoundaries: [0, 32, 64, 64 + payload.length * 8],
    headerEnd:       63,
  };
}

// ---------------------------------------------------------------------------
// PROFILE A — BPSK
// ---------------------------------------------------------------------------
const PROFILE_BPSK: DemoProfile = {
  id:    'bpsk',
  label: 'BPSK / 50 kbaud / 16 dB',

  params: {
    carrierFrequency: 401_500_000,  // 401.5 MHz
    sampleRate:       200_000,
    bandwidth:        102_400,
    snr:              16.2,
    symbolRate:        50_000,
  },
  paramsConfidence: { bandwidth: 0.91, snr: 0.88, symbolRate: 0.84 },

  processingStages: [
    { name: 'Ingestion',       status: 'completed' },
    { name: 'DSP',             status: 'completed', method: 'FFT/PSD' },
    { name: 'Modulation',      status: 'completed', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'completed' },
    { name: 'Demodulation',    status: 'completed' },
    { name: 'FEC',             status: 'completed' },
  ],

  diagnostics: {
    evm_rms:          6.85,
    timing_error_rms: 0.061,
    sync_locked:      true,
    demod_locked:     true,
    fec_valid:        true,
    snr:              16.2,
    carrier_offset:   145.5,
  },

  hypotheses: [
    { id: 'a-h1', modulation: 'BPSK',  symbolRate:  50_000, mlConfidence: 0.93, validation: { syncPassed: true,  demodPassed: true,  fecPassed: true  }, isWinner: true  },
    { id: 'a-h2', modulation: 'DBPSK', symbolRate:  50_000, mlConfidence: 0.61, validation: { syncPassed: true,  demodPassed: true,  fecPassed: false }, isWinner: false },
    { id: 'a-h3', modulation: 'QPSK',  symbolRate:  25_000, mlConfidence: 0.38, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'a-h4', modulation: 'OQPSK', symbolRate:  25_000, mlConfidence: 0.19, validation: { syncPassed: false, demodPassed: false, fecPassed: false }, isWinner: false },
  ],

  decoderCandidates: [
    { id: 'a-d1', rank: 1, modulation: 'BPSK → Block 8×16 → Viterbi K=7',       symbolRate:  50_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.93, stages: { sync: true,  demod: true,  fec: true  }, bitErrors:    0 },
    { id: 'a-d2', rank: 2, modulation: 'BPSK → Block 8×16 → Reed-Solomon(255,223)', symbolRate: 50_000, fecType: 'Reed-Solomon (255,223)',     score: 0.71, stages: { sync: true,  demod: true,  fec: false }, bitErrors:  412 },
    { id: 'a-d3', rank: 3, modulation: 'DBPSK → Conv. Interleaver → Viterbi K=7',symbolRate:  50_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.54, stages: { sync: true,  demod: true,  fec: false }, bitErrors: 1893 },
    { id: 'a-d4', rank: 4, modulation: 'QPSK → Block 8×16 → Viterbi K=7',        symbolRate:  25_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.22, stages: { sync: false, demod: false, fec: false }, bitErrors: undefined },
  ],

  signalRegion: {
    id: 'a-r1', frequencyStart: 401_449_600, frequencyEnd: 401_550_400,
    bandwidth: 100_800, centerFreq: 401_500_000, snr: 16.2,
    candidateModulation: 'BPSK', confidence: 0.93,
  },

  bitstream: buildBitstream(
    '1010101010101010',
    '0111110000100001',
    '10110001010011000110001001011101',
    ['01000001','10110010','00101100','11001010','01011001','10000111','00111010','11100001',
     '01101100','10010011','11010100','00110110','10101011','01001110','00011101','11110000'],
    '1011000101001110',
  ),

  constellationPoints: BPSK_POINTS,
};

// ---------------------------------------------------------------------------
// PROFILE B — QPSK
// ---------------------------------------------------------------------------
const PROFILE_QPSK: DemoProfile = {
  id:    'qpsk',
  label: 'QPSK / 100 kbaud / 20 dB',

  params: {
    carrierFrequency: 437_500_000,
    sampleRate:       500_000,
    bandwidth:        204_800,
    snr:              20.4,
    symbolRate:       100_000,
  },
  paramsConfidence: { bandwidth: 0.87, snr: 0.93, symbolRate: 0.81 },

  processingStages: [
    { name: 'Ingestion',       status: 'completed' },
    { name: 'DSP',             status: 'completed', method: 'FFT/PSD' },
    { name: 'Modulation',      status: 'completed', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'completed' },
    { name: 'Demodulation',    status: 'completed' },
    { name: 'FEC',             status: 'completed' },
  ],

  diagnostics: {
    evm_rms:          4.21,
    timing_error_rms: 0.038,
    sync_locked:      true,
    demod_locked:     true,
    fec_valid:        true,
    snr:              20.4,
    carrier_offset:   312.5,
  },

  hypotheses: [
    { id: 'b-h1', modulation: 'QPSK',  symbolRate: 100_000, mlConfidence: 0.94, validation: { syncPassed: true,  demodPassed: true,  fecPassed: true  }, isWinner: true  },
    { id: 'b-h2', modulation: 'OQPSK', symbolRate: 100_000, mlConfidence: 0.67, validation: { syncPassed: true,  demodPassed: true,  fecPassed: false }, isWinner: false },
    { id: 'b-h3', modulation: 'BPSK',  symbolRate: 100_000, mlConfidence: 0.41, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'b-h4', modulation: '8-PSK', symbolRate:  50_000, mlConfidence: 0.23, validation: { syncPassed: false, demodPassed: false, fecPassed: false }, isWinner: false },
  ],

  decoderCandidates: [
    { id: 'b-d1', rank: 1, modulation: 'QPSK → Block 16×32 → Viterbi K=7',          symbolRate: 100_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.94, stages: { sync: true,  demod: true,  fec: true  }, bitErrors:    0 },
    { id: 'b-d2', rank: 2, modulation: 'QPSK → Conv. Interleaver → Viterbi K=7',     symbolRate: 100_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.67, stages: { sync: true,  demod: true,  fec: false }, bitErrors:  847 },
    { id: 'b-d3', rank: 3, modulation: 'QPSK → Block 16×32 → Reed-Solomon(255,223)', symbolRate: 100_000, fecType: 'Reed-Solomon (255,223)',       score: 0.58, stages: { sync: true,  demod: true,  fec: false }, bitErrors: 2341 },
    { id: 'b-d4', rank: 4, modulation: 'OQPSK → Block 16×32 → Viterbi K=7',          symbolRate: 100_000, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.31, stages: { sync: false, demod: false, fec: false }, bitErrors: undefined },
  ],

  signalRegion: {
    id: 'b-r1', frequencyStart: 437_397_600, frequencyEnd: 437_602_400,
    bandwidth: 204_800, centerFreq: 437_500_000, snr: 20.4,
    candidateModulation: 'QPSK', confidence: 0.94,
  },

  bitstream: buildBitstream(
    '1010101010101010',
    '1111010010110101',
    '11010010001101011000110001110001',
    ['01000001','00110101','11001010','00010110','10110011','01111000','11100001','00101101',
     '10010110','11011011','00110000','01101001','10100101','11110000','00011101','10101010'],
    '1100001001011010',
  ),

  constellationPoints: QPSK_POINTS,
};

// ---------------------------------------------------------------------------
// PROFILE C — 8-PSK
// ---------------------------------------------------------------------------
const PROFILE_8PSK: DemoProfile = {
  id:    '8psk',
  label: '8-PSK / 150 kbaud / 13 dB',

  params: {
    carrierFrequency: 915_000_000,
    sampleRate:       750_000,
    bandwidth:        307_200,
    snr:              13.1,
    symbolRate:       150_000,
  },
  paramsConfidence: { bandwidth: 0.79, snr: 0.74, symbolRate: 0.72 },

  processingStages: [
    { name: 'Ingestion',       status: 'completed' },
    { name: 'DSP',             status: 'completed', method: 'FFT/PSD' },
    { name: 'Modulation',      status: 'completed', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'completed' },
    { name: 'Demodulation',    status: 'completed' },
    { name: 'FEC',             status: 'completed' },
  ],

  diagnostics: {
    evm_rms:          9.47,
    timing_error_rms: 0.093,
    sync_locked:      true,
    demod_locked:     true,
    fec_valid:        true,
    snr:              13.1,
    carrier_offset:   875.0,
  },

  hypotheses: [
    { id: 'c-h1', modulation: '8-PSK', symbolRate: 150_000, mlConfidence: 0.86, validation: { syncPassed: true,  demodPassed: true,  fecPassed: true  }, isWinner: true  },
    { id: 'c-h2', modulation: 'QPSK',  symbolRate: 150_000, mlConfidence: 0.62, validation: { syncPassed: true,  demodPassed: true,  fecPassed: false }, isWinner: false },
    { id: 'c-h3', modulation: 'OQPSK', symbolRate: 150_000, mlConfidence: 0.44, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'c-h4', modulation: '16-QAM',symbolRate:  75_000, mlConfidence: 0.31, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'c-h5', modulation: 'BPSK',  symbolRate: 150_000, mlConfidence: 0.17, validation: { syncPassed: false, demodPassed: false, fecPassed: false }, isWinner: false },
  ],

  decoderCandidates: [
    { id: 'c-d1', rank: 1, modulation: '8-PSK → Block 24×32 → Viterbi K=7',          symbolRate: 150_000, fecType: 'Convolutional K=7 (rate 2/3)', score: 0.86, stages: { sync: true,  demod: true,  fec: true  }, bitErrors:     0 },
    { id: 'c-d2', rank: 2, modulation: '8-PSK → Conv. Interleaver → Viterbi K=7',     symbolRate: 150_000, fecType: 'Convolutional K=7 (rate 2/3)', score: 0.62, stages: { sync: true,  demod: true,  fec: false }, bitErrors:  1104 },
    { id: 'c-d3', rank: 3, modulation: 'QPSK → Block 24×32 → Reed-Solomon(255,223)',  symbolRate: 150_000, fecType: 'Reed-Solomon (255,223)',        score: 0.44, stages: { sync: true,  demod: false, fec: false }, bitErrors:  3872 },
    { id: 'c-d4', rank: 4, modulation: '8-PSK → Block 24×32 → LDPC(1/2)',             symbolRate: 150_000, fecType: 'LDPC (rate 1/2)',               score: 0.38, stages: { sync: true,  demod: false, fec: false }, bitErrors:  5210 },
    { id: 'c-d5', rank: 5, modulation: 'OQPSK → Conv. Interleaver → Viterbi K=7',     symbolRate: 150_000, fecType: 'Convolutional K=7 (rate 2/3)', score: 0.21, stages: { sync: false, demod: false, fec: false }, bitErrors: undefined },
  ],

  signalRegion: {
    id: 'c-r1', frequencyStart: 914_846_400, frequencyEnd: 915_153_600,
    bandwidth: 307_200, centerFreq: 915_000_000, snr: 13.1,
    candidateModulation: '8-PSK', confidence: 0.86,
  },

  bitstream: buildBitstream(
    '1111000011110000',
    '1010010111011010',
    '01100011101001011101001000110111',
    ['11000011','01010110','10111001','00111100','11101000','01001011','10010110','00101101',
     '11110010','01001100','10100001','00011110','11001100','10110101','01110010','00001111',
     '10101011','11000110','00111001','01011010'],
    '0110100110010101',
  ),

  constellationPoints: PSK8_POINTS,
};

// ---------------------------------------------------------------------------
// PROFILE D — 16-QAM
// ---------------------------------------------------------------------------
const PROFILE_16QAM: DemoProfile = {
  id:    '16qam',
  label: '16-QAM / 200 kbaud / 18 dB',

  params: {
    carrierFrequency: 2_437_000_000,
    sampleRate:       1_000_000,
    bandwidth:        409_600,
    snr:              18.3,
    symbolRate:       200_000,
  },
  paramsConfidence: { bandwidth: 0.83, snr: 0.90, symbolRate: 0.78 },

  processingStages: [
    { name: 'Ingestion',       status: 'completed' },
    { name: 'DSP',             status: 'completed', method: 'FFT/PSD' },
    { name: 'Modulation',      status: 'completed', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'completed' },
    { name: 'Demodulation',    status: 'completed' },
    { name: 'FEC',             status: 'completed' },
  ],

  diagnostics: {
    evm_rms:          7.63,
    timing_error_rms: 0.052,
    sync_locked:      true,
    demod_locked:     true,
    fec_valid:        true,
    snr:              18.3,
    carrier_offset:   531.25,
  },

  hypotheses: [
    { id: 'd-h1', modulation: '16-QAM', symbolRate: 200_000, mlConfidence: 0.91, validation: { syncPassed: true,  demodPassed: true,  fecPassed: true  }, isWinner: true  },
    { id: 'd-h2', modulation: '8-PSK',  symbolRate: 200_000, mlConfidence: 0.58, validation: { syncPassed: true,  demodPassed: true,  fecPassed: false }, isWinner: false },
    { id: 'd-h3', modulation: '64-QAM', symbolRate: 100_000, mlConfidence: 0.42, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'd-h4', modulation: 'QPSK',   symbolRate: 200_000, mlConfidence: 0.27, validation: { syncPassed: false, demodPassed: false, fecPassed: false }, isWinner: false },
  ],

  decoderCandidates: [
    { id: 'd-d1', rank: 1, modulation: '16-QAM → Block 32×64 → Viterbi K=7',          symbolRate: 200_000, fecType: 'Convolutional K=7 (rate 3/4)', score: 0.91, stages: { sync: true,  demod: true,  fec: true  }, bitErrors:     0 },
    { id: 'd-d2', rank: 2, modulation: '16-QAM → Block 32×64 → LDPC(3/4)',             symbolRate: 200_000, fecType: 'LDPC (rate 3/4)',              score: 0.74, stages: { sync: true,  demod: true,  fec: false }, bitErrors:   623 },
    { id: 'd-d3', rank: 3, modulation: '16-QAM → Conv. Interleaver → Reed-Solomon',    symbolRate: 200_000, fecType: 'Reed-Solomon (255,239)',        score: 0.55, stages: { sync: true,  demod: true,  fec: false }, bitErrors:  2819 },
    { id: 'd-d4', rank: 4, modulation: '8-PSK → Block 32×64 → Viterbi K=7',            symbolRate: 200_000, fecType: 'Convolutional K=7 (rate 3/4)', score: 0.31, stages: { sync: false, demod: false, fec: false }, bitErrors: undefined },
  ],

  signalRegion: {
    id: 'd-r1', frequencyStart: 2_436_795_200, frequencyEnd: 2_437_204_800,
    bandwidth: 409_600, centerFreq: 2_437_000_000, snr: 18.3,
    candidateModulation: '16-QAM', confidence: 0.91,
  },

  bitstream: buildBitstream(
    '1100110011001100',
    '0101010110101011',
    '11001000101100011010100101110110',
    ['10100011','11000101','01101110','10011010','00111011','11010000','01001101','10110110',
     '00101010','11111000','01010011','10001100','11100010','00110111','10101010','01100001',
     '11011100','00111010','10010101','01001110','11000111','00011010','10110001','01101100'],
    '1010001101011001',
  ),

  constellationPoints: QAM16_POINTS,
};

// ---------------------------------------------------------------------------
// PROFILE E — FSK (2-FSK)
// ---------------------------------------------------------------------------
const PROFILE_FSK: DemoProfile = {
  id:    'fsk',
  label: '2-FSK / 9.6 kbaud / 14 dB',

  params: {
    carrierFrequency: 433_920_000,
    sampleRate:       96_000,
    bandwidth:        38_400,
    snr:              14.7,
    symbolRate:        9_600,
  },
  paramsConfidence: { bandwidth: 0.85, snr: 0.82, symbolRate: 0.88 },

  processingStages: [
    { name: 'Ingestion',       status: 'completed' },
    { name: 'DSP',             status: 'completed', method: 'FFT/PSD' },
    { name: 'Modulation',      status: 'completed', method: 'ML Classifier' },
    { name: 'Synchronization', status: 'completed' },
    { name: 'Demodulation',    status: 'completed' },
    { name: 'FEC',             status: 'completed' },
  ],

  diagnostics: {
    evm_rms:          null as unknown as number,  // N/A for FSK (carrier-phase EVM meaningless)
    timing_error_rms: 0.104,
    sync_locked:      true,
    demod_locked:     true,
    fec_valid:        true,
    snr:              14.7,
    carrier_offset:   62.5,
  },

  hypotheses: [
    { id: 'e-h1', modulation: '2-FSK',  symbolRate:  9_600, mlConfidence: 0.89, validation: { syncPassed: true,  demodPassed: true,  fecPassed: true  }, isWinner: true  },
    { id: 'e-h2', modulation: 'GFSK',   symbolRate:  9_600, mlConfidence: 0.71, validation: { syncPassed: true,  demodPassed: true,  fecPassed: false }, isWinner: false },
    { id: 'e-h3', modulation: '4-FSK',  symbolRate:  4_800, mlConfidence: 0.39, validation: { syncPassed: true,  demodPassed: false, fecPassed: false }, isWinner: false },
    { id: 'e-h4', modulation: 'BPSK',   symbolRate:  9_600, mlConfidence: 0.21, validation: { syncPassed: false, demodPassed: false, fecPassed: false }, isWinner: false },
  ],

  decoderCandidates: [
    { id: 'e-d1', rank: 1, modulation: '2-FSK → None → Viterbi K=7',             symbolRate: 9_600, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.89, stages: { sync: true,  demod: true,  fec: true  }, bitErrors:    0 },
    { id: 'e-d2', rank: 2, modulation: 'GFSK → None → Viterbi K=7',              symbolRate: 9_600, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.71, stages: { sync: true,  demod: true,  fec: false }, bitErrors:  318 },
    { id: 'e-d3', rank: 3, modulation: '2-FSK → Block 4×8 → Viterbi K=7',        symbolRate: 9_600, fecType: 'Convolutional K=7 (rate 1/2)', score: 0.53, stages: { sync: true,  demod: true,  fec: false }, bitErrors: 1042 },
    { id: 'e-d4', rank: 4, modulation: '4-FSK → None → Reed-Solomon(255,223)',    symbolRate: 4_800, fecType: 'Reed-Solomon (255,223)',        score: 0.28, stages: { sync: false, demod: false, fec: false }, bitErrors: undefined },
  ],

  signalRegion: {
    id: 'e-r1', frequencyStart: 433_900_800, frequencyEnd: 433_939_200,
    bandwidth: 38_400, centerFreq: 433_920_000, snr: 14.7,
    candidateModulation: '2-FSK', confidence: 0.89,
  },

  bitstream: buildBitstream(
    '0101010101010101',
    '0111111001000000',
    '10010110001011010011100001010110',
    ['00110011','11001100','00110011','11001100','01010101','10101010','00001111','11110000',
     '01100110','10011001','01100110','10011001'],
    '0100110001001011',
  ),

  constellationPoints: FSK_POINTS,
};

// ---------------------------------------------------------------------------
// Profile registry — exported as an array and a lookup map
// ---------------------------------------------------------------------------

export const ALL_PROFILES: DemoProfile[] = [
  PROFILE_BPSK,
  PROFILE_QPSK,
  PROFILE_8PSK,
  PROFILE_16QAM,
  PROFILE_FSK,
];

export const PROFILES_BY_ID: Record<string, DemoProfile> = {
  bpsk:   PROFILE_BPSK,
  qpsk:   PROFILE_QPSK,
  '8psk': PROFILE_8PSK,
  '16qam':PROFILE_16QAM,
  fsk:    PROFILE_FSK,
};
