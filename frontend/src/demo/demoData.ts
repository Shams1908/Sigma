/**
 * SYNTHETIC DEMO DATA — frontend display only.
 *
 * ALL values in this file are deterministic, hand-crafted constants that
 * exist solely to populate the UI during demonstrations when no real signal
 * file has been uploaded and the backend is not running.
 *
 * NOTHING in this file represents real analysis, real FEC decoding, real
 * hypothesis scoring, real synchronisation, real bitstream processing,
 * or any other DSP / ML output.
 *
 * The UI labels every section that renders from this data with a visible
 * "SYNTHETIC DEMO DATA" badge so there is no ambiguity.
 */

// ---------------------------------------------------------------------------
// Signal identity
// ---------------------------------------------------------------------------

export const DEMO_SIGNAL_ID = 'DEMO-SIG-001';

// ---------------------------------------------------------------------------
// Signal parameters (DataReadouts)
// ---------------------------------------------------------------------------

export const demoSignalParams = {
  carrierFrequency: 437_500_000,   // 437.5 MHz — plausible LEO-band carrier
  sampleRate:       500_000,        // 500 ksps
  bandwidth:        204_800,        // ~200 kHz
  snr:              20.4,           // dB
  symbolRate:       100_000,        // 100 kbaud (QPSK → ~200 kHz BW)
};

export const demoSignalParamsConfidence = {
  bandwidth:   0.87,
  snr:         0.93,
  symbolRate:  0.81,
};

// ---------------------------------------------------------------------------
// Processing chain (ProcessingChain)
// ---------------------------------------------------------------------------

export const demoProcessingStages = [
  { name: 'Ingestion',        status: 'completed' as const },
  { name: 'DSP',              status: 'completed' as const, method: 'FFT/PSD' },
  { name: 'Modulation',       status: 'completed' as const, method: 'ML Classifier' },
  { name: 'Synchronization',  status: 'completed' as const },
  { name: 'Demodulation',     status: 'completed' as const },
  { name: 'FEC',              status: 'completed' as const },
];

// ---------------------------------------------------------------------------
// Diagnostics (DiagnosticsPanel)
// ---------------------------------------------------------------------------

export const demoDiagnostics = {
  evm_rms:           4.21,   // % — good QPSK EVM
  timing_error_rms:  0.038,  // μs — tight timing
  sync_locked:       true,
  demod_locked:      true,
  fec_valid:         true,
  snr:               20.4,   // dB
  carrier_offset:    312.5,  // Hz — small realistic offset
};

// ---------------------------------------------------------------------------
// Hypotheses (HypothesisExplorer)
// ---------------------------------------------------------------------------

export const demoHypotheses = [
  {
    id:            'demo-hyp-1',
    modulation:    'QPSK',
    symbolRate:    100_000,
    mlConfidence:  0.94,
    validation: {
      syncPassed:  true,
      demodPassed: true,
      fecPassed:   true,
    },
    isWinner: true,
  },
  {
    id:            'demo-hyp-2',
    modulation:    'OQPSK',
    symbolRate:    100_000,
    mlConfidence:  0.67,
    validation: {
      syncPassed:  true,
      demodPassed: true,
      fecPassed:   false,
    },
    isWinner: false,
  },
  {
    id:            'demo-hyp-3',
    modulation:    'BPSK',
    symbolRate:    100_000,
    mlConfidence:  0.41,
    validation: {
      syncPassed:  true,
      demodPassed: false,
      fecPassed:   false,
    },
    isWinner: false,
  },
  {
    id:            'demo-hyp-4',
    modulation:    '8-PSK',
    symbolRate:    50_000,
    mlConfidence:  0.23,
    validation: {
      syncPassed:  false,
      demodPassed: false,
      fecPassed:   false,
    },
    isWinner: false,
  },
];

// ---------------------------------------------------------------------------
// Decoder Lab candidates (DecoderLab)
// ---------------------------------------------------------------------------

export const demoDecoderCandidates = [
  {
    id:         'demo-dec-1',
    rank:       1,
    modulation: 'QPSK → Block 16×32 → Viterbi K=7',
    symbolRate: 100_000,
    fecType:    'Convolutional K=7 (rate 1/2)',
    score:      0.94,
    stages: { sync: true, demod: true, fec: true },
    bitErrors:  0,
  },
  {
    id:         'demo-dec-2',
    rank:       2,
    modulation: 'QPSK → Conv. Interleaver → Viterbi K=7',
    symbolRate: 100_000,
    fecType:    'Convolutional K=7 (rate 1/2)',
    score:      0.67,
    stages: { sync: true, demod: true, fec: false },
    bitErrors:  847,
  },
  {
    id:         'demo-dec-3',
    rank:       3,
    modulation: 'QPSK → Block 16×32 → Reed-Solomon(255,223)',
    symbolRate: 100_000,
    fecType:    'Reed-Solomon (255,223)',
    score:      0.58,
    stages: { sync: true, demod: true, fec: false },
    bitErrors:  2_341,
  },
  {
    id:         'demo-dec-4',
    rank:       4,
    modulation: 'OQPSK → Block 16×32 → Viterbi K=7',
    symbolRate: 100_000,
    fecType:    'Convolutional K=7 (rate 1/2)',
    score:      0.31,
    stages: { sync: false, demod: false, fec: false },
    bitErrors:  undefined,
  },
];

// ---------------------------------------------------------------------------
// Signal Explorer regions (SignalExplorer)
// ---------------------------------------------------------------------------

export const demoSignalRegions = [
  {
    id:                  'demo-region-1',
    frequencyStart:      437_397_600,   // Hz
    frequencyEnd:        437_602_400,   // Hz
    bandwidth:           204_800,       // ~200 kHz
    centerFreq:          437_500_000,
    snr:                 20.4,
    candidateModulation: 'QPSK',
    confidence:          0.94,
  },
];

// ---------------------------------------------------------------------------
// Bitstream (BitstreamViewer)
//
// Deterministic frame layout (display only — no processing):
//
//  Bits 0–15:    Preamble   1010101010101010
//  Bits 16–31:   Sync word  1111010010110101
//  Bits 32–63:   Header     11010010 00110101 10001100 01110001
//  Bits 64–191:  Payload    16 × deterministic bytes
//  Bits 192–207: CRC field  1100001001011010  (display value only)
// ---------------------------------------------------------------------------

const PREAMBLE   = '1010101010101010';
const SYNC_WORD  = '1111010010110101';
const HEADER     = '11010010001101011000110001110001';
// 16 payload bytes — deterministic pattern
const PAYLOAD    = [
  '01000001', // 0x41 'A'
  '00110101', // 0x35
  '11001010', // 0xCA
  '00010110', // 0x16
  '10110011', // 0xB3
  '01111000', // 0x78
  '11100001', // 0xE1
  '00101101', // 0x2D
  '10010110', // 0x96
  '11011011', // 0xDB
  '00110000', // 0x30
  '01101001', // 0x69
  '10100101', // 0xA5
  '11110000', // 0xF0
  '00011101', // 0x1D
  '10101010', // 0xAA
].join('');
const CRC_FIELD  = '1100001001011010';

export const demoBitstreamBits =
  PREAMBLE + SYNC_WORD + HEADER + PAYLOAD + CRC_FIELD;

export const demoBitstreamData = {
  bits:      demoBitstreamBits,
  totalBits: demoBitstreamBits.length,   // 208 display bits
  entropy:   0.961,                       // near-max for compressed/encrypted data
  onesRatio: 0.510,                       // ~balanced
  // Frame structure metadata (for BitstreamViewer's frameBoundaries and headerEnd props)
  frameBoundaries: [0, 32, 64, 192],
  headerEnd:       63,
};
