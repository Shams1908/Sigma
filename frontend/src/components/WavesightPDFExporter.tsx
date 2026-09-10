import { RefObject } from 'react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { captureComponentCanvas } from '../utils/canvasExtractor';

interface ExportParameters {
  sampleRate: number;
  bandwidth: number;
  snr: number;
  symbolRate: number;
  carrierOffset: number;
}

interface Hypothesis {
  id: string;
  modulation: string;
  symbolRate: number;
  mlConfidence: number;
  validation: {
    syncPassed: boolean;
    demodPassed: boolean;
    fecPassed: boolean;
  };
  isWinner: boolean;
}

interface Diagnostics {
  evm_rms: number | null;
  timing_error_rms: number | null;
  sync_locked: boolean;
  demod_locked: boolean;
  fec_valid: boolean;
  snr: number;
  carrier_offset: number;
}

interface VisualizationRefs {
  spectrumRef: RefObject<HTMLDivElement>;
  waterfallRef: RefObject<HTMLDivElement>;
  waveformRef: RefObject<HTMLDivElement>;
  constellationRef: RefObject<HTMLDivElement>;
}

interface StatisticalFingerprint {
  crestFactor: number;
  kurtosis: number;
  skewness: number;
}

function calculateStatistics(iData: number[], qData: number[]): StatisticalFingerprint {
  const n = Math.min(iData.length, qData.length);
  if (n === 0) return { crestFactor: 0, kurtosis: 0, skewness: 0 };

  const magnitudes = [];
  for (let i = 0; i < n; i++) {
    magnitudes.push(Math.sqrt(iData[i] * iData[i] + qData[i] * qData[i]));
  }

  const peakPower = Math.max(...magnitudes.map(m => m * m));
  const avgPower = magnitudes.reduce((sum, m) => sum + m * m, 0) / n;
  const crestFactor = avgPower > 0 ? 10 * Math.log10(peakPower / avgPower) : 0;

  const mean = magnitudes.reduce((sum, m) => sum + m, 0) / n;
  const variance = magnitudes.reduce((sum, m) => sum + Math.pow(m - mean, 2), 0) / n;
  const stdDev = Math.sqrt(variance);

  const m3 = magnitudes.reduce((sum, m) => sum + Math.pow(m - mean, 3), 0) / n;
  const m4 = magnitudes.reduce((sum, m) => sum + Math.pow(m - mean, 4), 0) / n;

  const skewness = stdDev > 0 ? m3 / Math.pow(stdDev, 3) : 0;
  const kurtosis = variance > 0 ? m4 / Math.pow(variance, 2) - 3 : 0;

  return {
    crestFactor: isFinite(crestFactor) ? crestFactor : 0,
    kurtosis: isFinite(kurtosis) ? kurtosis : 0,
    skewness: isFinite(skewness) ? skewness : 0
  };
}

function generateExecutiveSummary(
  parameters: ExportParameters,
  diagnostics: Diagnostics | null,
  hypotheses: Hypothesis[]
): string {
  const syncStatus = diagnostics?.sync_locked ? 'LOCKED' : 'UNLOCKED';
  const demodStatus = diagnostics?.demod_locked ? 'LOCKED' : 'UNLOCKED';
  const topHypothesis = hypotheses.find(h => h.isWinner) || hypotheses[0];
  const modulation = topHypothesis ? topHypothesis.modulation : 'UNKNOWN';
  const confidence = topHypothesis ? (topHypothesis.mlConfidence * 100).toFixed(1) : '0';

  const snrQuality = parameters.snr > 20 ? 'EXCELLENT' : 
                     parameters.snr > 10 ? 'GOOD' : 
                     parameters.snr > 5 ? 'MODERATE' : 'POOR';

  return `AUTOMATED ASSESSMENT: Signal exhibits ${snrQuality.toLowerCase()} SNR characteristics (${parameters.snr.toFixed(1)} dB). ` +
         `Synchronization status: ${syncStatus}. Demodulation status: ${demodStatus}. ` +
         `Primary modulation hypothesis: ${modulation} with ${confidence}% confidence. ` +
         `${hypotheses.length} candidate${hypotheses.length !== 1 ? 's' : ''} evaluated.`;
}

async function captureCanvasImage(componentRef: RefObject<HTMLDivElement>): Promise<string | null> {
  try {
    if (!componentRef.current) return null;
    return await captureComponentCanvas(componentRef.current);
  } catch (error) {
    console.error('Failed to capture canvas:', error);
    return null;
  }
}

export async function exportWavesightPDF(
  signalId: string,
  fileName: string,
  parameters: ExportParameters,
  hypotheses: Hypothesis[],
  diagnostics: Diagnostics | null,
  visualizationRefs: VisualizationRefs,
  iData?: number[],
  qData?: number[]
): Promise<void> {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 15;

  doc.setFillColor(5, 5, 5);
  doc.rect(0, 0, pageWidth, pageHeight, 'F');

  doc.setFillColor(0, 229, 255);
  doc.rect(margin, margin, pageWidth - 2 * margin, 15, 'F');
  
  doc.setFont('courier', 'bold');
  doc.setFontSize(18);
  doc.setTextColor(5, 5, 5);
  doc.text('WAVESIGHT SIGNAL FORENSICS REPORT', pageWidth / 2, margin + 10, { align: 'center' });

  let yPos = margin + 20;

  doc.setFillColor(15, 15, 15);
  doc.rect(margin, yPos, pageWidth - 2 * margin, 12, 'F');
  
  doc.setFont('courier', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(0, 229, 255);
  doc.text(`SIGNAL_ID: ${signalId}`, margin + 5, yPos + 4);
  doc.text('|||||||||||||||||||', margin + 5, yPos + 8);

  yPos += 17;

  doc.setFontSize(7);
  doc.setTextColor(180, 180, 180);
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  doc.text(`Generated: ${timestamp}`, margin, yPos);
  yPos += 4;
  doc.text(`File: ${fileName}`, margin, yPos);
  yPos += 4;
  doc.text(`Signal ID: ${signalId}`, margin, yPos);
  yPos += 8;

  const executiveSummary = generateExecutiveSummary(parameters, diagnostics, hypotheses);
  doc.setFontSize(8);
  doc.setTextColor(255, 255, 255);
  doc.setFont('courier', 'bold');
  doc.text('AUTOMATED EXECUTIVE SUMMARY', margin, yPos);
  yPos += 5;
  doc.setFont('courier', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(200, 200, 200);
  const summaryLines = doc.splitTextToSize(executiveSummary, pageWidth - 2 * margin);
  doc.text(summaryLines, margin, yPos);
  yPos += summaryLines.length * 3 + 5;

  doc.setFont('courier', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(0, 229, 255);
  doc.text('SIGNAL PARAMETERS', margin, yPos);
  yPos += 7;

  const paramTableData = [
    ['PARAMETER', 'VALUE', 'UNIT'],
    ['Sample Rate', (parameters.sampleRate / 1000).toFixed(2), 'ksps'],
    ['Bandwidth', (parameters.bandwidth / 1000).toFixed(2), 'kHz'],
    ['SNR', parameters.snr.toFixed(2), 'dB'],
    ['Symbol Rate', (parameters.symbolRate / 1000).toFixed(2), 'ksps'],
    ['Carrier Offset', parameters.carrierOffset.toFixed(2), 'Hz']
  ];

  autoTable(doc, {
    startY: yPos,
    head: [paramTableData[0]],
    body: paramTableData.slice(1),
    theme: 'plain',
    styles: {
      font: 'courier',
      fontSize: 7,
      textColor: [220, 220, 220],
      fillColor: [5, 5, 5],
      lineColor: [50, 50, 50],
      lineWidth: 0.1
    },
    headStyles: {
      fillColor: [0, 229, 255],
      textColor: [5, 5, 5],
      fontStyle: 'bold',
      fontSize: 8
    },
    alternateRowStyles: {
      fillColor: [15, 15, 15]
    },
    margin: { left: margin, right: margin }
  });

  yPos = (doc as any).lastAutoTable.finalY + 10;

  if (iData && qData && iData.length > 0 && qData.length > 0) {
    const stats = calculateStatistics(iData, qData);
    
    doc.setFont('courier', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(178, 0, 255);
    doc.text('STATISTICAL FINGERPRINT', margin, yPos);
    yPos += 7;

    const statsTableData = [
      ['METRIC', 'VALUE', 'INTERPRETATION'],
      ['Crest Factor', stats.crestFactor.toFixed(3), 'dB PAPR'],
      ['Kurtosis', stats.kurtosis.toFixed(3), 'Signal Impulsiveness'],
      ['Skewness', stats.skewness.toFixed(3), 'Asymmetry Measure']
    ];

    autoTable(doc, {
      startY: yPos,
      head: [statsTableData[0]],
      body: statsTableData.slice(1),
      theme: 'plain',
      styles: {
        font: 'courier',
        fontSize: 7,
        textColor: [220, 220, 220],
        fillColor: [5, 5, 5],
        lineColor: [50, 50, 50],
        lineWidth: 0.1
      },
      headStyles: {
        fillColor: [178, 0, 255],
        textColor: [255, 255, 255],
        fontStyle: 'bold',
        fontSize: 8
      },
      alternateRowStyles: {
        fillColor: [15, 15, 15]
      },
      margin: { left: margin, right: margin }
    });

    yPos = (doc as any).lastAutoTable.finalY + 10;
  }

  if (diagnostics) {
    doc.setFont('courier', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(0, 229, 255);
    doc.text('DIAGNOSTICS', margin, yPos);
    yPos += 7;

    const diagTableData = [
      ['METRIC', 'STATUS/VALUE'],
      ['Synchronization', diagnostics.sync_locked ? '✓ LOCKED' : '✗ UNLOCKED'],
      ['Demodulation', diagnostics.demod_locked ? '✓ LOCKED' : '✗ UNLOCKED'],
      ['FEC Validation', diagnostics.fec_valid ? '✓ VALID' : '✗ INVALID'],
      ['EVM RMS', diagnostics.evm_rms !== null ? diagnostics.evm_rms.toFixed(4) : 'N/A'],
      ['Timing Error', diagnostics.timing_error_rms !== null ? `${diagnostics.timing_error_rms.toFixed(3)} μs` : 'N/A']
    ];

    autoTable(doc, {
      startY: yPos,
      head: [diagTableData[0]],
      body: diagTableData.slice(1),
      theme: 'plain',
      styles: {
        font: 'courier',
        fontSize: 7,
        textColor: [220, 220, 220],
        fillColor: [5, 5, 5],
        lineColor: [50, 50, 50],
        lineWidth: 0.1
      },
      headStyles: {
        fillColor: [0, 229, 255],
        textColor: [5, 5, 5],
        fontStyle: 'bold',
        fontSize: 8
      },
      alternateRowStyles: {
        fillColor: [15, 15, 15]
      },
      margin: { left: margin, right: margin }
    });

    yPos = (doc as any).lastAutoTable.finalY + 10;
  }

  doc.addPage();
  yPos = margin;

  doc.setFont('courier', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(0, 229, 255);
  doc.text('HIGH-FIDELITY VISUAL EXPORT', pageWidth / 2, yPos, { align: 'center' });
  yPos += 10;

  const spectrumImg = await captureCanvasImage(visualizationRefs.spectrumRef);
  const waterfallImg = await captureCanvasImage(visualizationRefs.waterfallRef);
  const waveformImg = await captureCanvasImage(visualizationRefs.waveformRef);
  const constellationImg = await captureCanvasImage(visualizationRefs.constellationRef);

  const imgWidth = (pageWidth - 3 * margin) / 2;
  const imgHeight = imgWidth * 0.75;

  if (spectrumImg) {
    doc.addImage(spectrumImg, 'JPEG', margin, yPos, imgWidth, imgHeight);
  }

  if (waterfallImg) {
    doc.addImage(waterfallImg, 'JPEG', margin + imgWidth + margin, yPos, imgWidth, imgHeight);
  }

  yPos += imgHeight + 5;

  if (waveformImg) {
    doc.addImage(waveformImg, 'JPEG', margin, yPos, imgWidth, imgHeight);
  }

  if (constellationImg) {
    doc.addImage(constellationImg, 'JPEG', margin + imgWidth + margin, yPos, imgWidth, imgHeight);
  }

  if (hypotheses && hypotheses.length > 0) {
    doc.addPage();
    yPos = margin;

    doc.setFont('courier', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(0, 229, 255);
    doc.text('HYPOTHESIS VALIDATION RESULTS', margin, yPos);
    yPos += 7;

    const hypTableData = [
      ['RANK', 'MODULATION', 'SYMBOL RATE', 'CONFIDENCE', 'SYNC', 'DEMOD', 'FEC']
    ];

    hypotheses.slice(0, 15).forEach((hyp, idx) => {
      hypTableData.push([
        (idx + 1).toString(),
        hyp.modulation,
        `${(hyp.symbolRate / 1000).toFixed(1)} ksps`,
        `${(hyp.mlConfidence * 100).toFixed(1)}%`,
        hyp.validation.syncPassed ? '✓' : '✗',
        hyp.validation.demodPassed ? '✓' : '✗',
        hyp.validation.fecPassed ? '✓' : '✗'
      ]);
    });

    autoTable(doc, {
      startY: yPos,
      head: [hypTableData[0]],
      body: hypTableData.slice(1),
      theme: 'plain',
      styles: {
        font: 'courier',
        fontSize: 6,
        textColor: [220, 220, 220],
        fillColor: [5, 5, 5],
        lineColor: [50, 50, 50],
        lineWidth: 0.1,
        halign: 'center'
      },
      headStyles: {
        fillColor: [0, 229, 255],
        textColor: [5, 5, 5],
        fontStyle: 'bold',
        fontSize: 7
      },
      alternateRowStyles: {
        fillColor: [15, 15, 15]
      },
      margin: { left: margin, right: margin }
    });
  }

  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(6);
    doc.setTextColor(100, 100, 100);
    doc.setFont('courier', 'italic');
    doc.text(
      'Generated by WAVESIGHT - Signal Intelligence & Forensic Analysis Platform',
      pageWidth / 2,
      pageHeight - 5,
      { align: 'center' }
    );
    doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 5, { align: 'right' });
  }

  doc.save(`WAVESIGHT_${signalId}_${Date.now()}.pdf`);
}
