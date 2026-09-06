"""
PDF Export - Comprehensive signal analysis report with embedded graphs.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_pdf_report(
    signal_id: str,
    analysis_id: str,
    filename: str,
    iq_data: np.ndarray,
    sample_rate: float,
    parameters: dict,
    hypotheses: list,
    diagnostics: dict | None,
) -> bytes:
    """
    Generate comprehensive PDF report with all analysis data and graphs.
    
    Returns: PDF bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#06b6d4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#06b6d4'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    story.append(Paragraph("SIGMA SIGNAL ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Paragraph(f"<b>File:</b> {filename}", styles['Normal']))
    story.append(Paragraph(f"<b>Signal ID:</b> {signal_id}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis ID:</b> {analysis_id}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("SIGNAL PARAMETERS", heading_style))
    
    param_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Sample Rate', f"{parameters.get('sampleRate', 0)/1000:.2f}", 'ksps'],
        ['Bandwidth', f"{parameters.get('bandwidth', 0)/1000:.2f}", 'kHz'],
        ['SNR', f"{parameters.get('snr', 0):.2f}", 'dB'],
        ['Symbol Rate', f"{parameters.get('symbolRate', 0)/1000:.2f}", 'ksps'],
        ['Carrier Offset', f"{parameters.get('carrierOffset', 0):.2f}", 'Hz'],
    ]
    
    param_table = Table(param_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 0.3*inch))
    
    if diagnostics:
        story.append(Paragraph("DIAGNOSTICS", heading_style))
        
        diag_data = [
            ['Metric', 'Status/Value'],
            ['Synchronization', '✓ LOCKED' if diagnostics.get('sync_locked') else '✗ UNLOCKED'],
            ['Demodulation', '✓ LOCKED' if diagnostics.get('demod_locked') else '✗ UNLOCKED'],
            ['FEC Validation', '✓ VALID' if diagnostics.get('fec_valid') else '✗ INVALID'],
            ['EVM RMS', f"{diagnostics.get('evm_rms', 0):.3f}" if diagnostics.get('evm_rms') else 'N/A'],
            ['Timing Error', f"{diagnostics.get('timing_error_rms', 0):.3f} μs" if diagnostics.get('timing_error_rms') else 'N/A'],
        ]
        
        diag_table = Table(diag_data, colWidths=[2.5*inch, 2.5*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())
    story.append(Paragraph("VISUALIZATIONS", heading_style))
    
    iq_complex = iq_data[0, :] + 1j * iq_data[1, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    from dsp.psd import estimate_psd
    freqs, psd = estimate_psd(iq_complex[:50000], sample_rate)
    axes[0, 0].plot(freqs/1e3, psd, color='#06b6d4', linewidth=0.8)
    axes[0, 0].set_xlabel('Frequency (kHz)', color='white')
    axes[0, 0].set_ylabel('Power (dB)', color='white')
    axes[0, 0].set_title('Power Spectral Density', color='#06b6d4', fontsize=10)
    axes[0, 0].set_facecolor('#0a0a0a')
    axes[0, 0].tick_params(colors='white')
    axes[0, 0].grid(True, alpha=0.2, color='gray')
    
    from dsp.spectrogram import compute_spectrogram
    freqs_spec, times_spec, Sxx = compute_spectrogram(iq_complex[:50000], sample_rate, nperseg=128)
    im = axes[0, 1].imshow(Sxx, aspect='auto', cmap='viridis', origin='lower',
                           extent=[times_spec[0]*1000, times_spec[-1]*1000, 
                                  freqs_spec[0]/1e3, freqs_spec[-1]/1e3])
    axes[0, 1].set_xlabel('Time (ms)', color='white')
    axes[0, 1].set_ylabel('Frequency (kHz)', color='white')
    axes[0, 1].set_title('Spectrogram', color='#06b6d4', fontsize=10)
    axes[0, 1].set_facecolor('#0a0a0a')
    axes[0, 1].tick_params(colors='white')
    
    decimation = max(1, len(iq_complex) // 5000)
    time_axis = np.arange(len(iq_complex[::decimation])) * decimation / sample_rate * 1000
    axes[1, 0].plot(time_axis, iq_data[0, ::decimation][:len(time_axis)], 
                   color='#06b6d4', linewidth=0.5, label='I', alpha=0.8)
    axes[1, 0].plot(time_axis, iq_data[1, ::decimation][:len(time_axis)], 
                   color='#a78bfa', linewidth=0.5, label='Q', alpha=0.8)
    axes[1, 0].set_xlabel('Time (ms)', color='white')
    axes[1, 0].set_ylabel('Amplitude', color='white')
    axes[1, 0].set_title('Time Domain Waveform', color='#06b6d4', fontsize=10)
    axes[1, 0].set_facecolor('#0a0a0a')
    axes[1, 0].tick_params(colors='white')
    axes[1, 0].legend(facecolor='#1a1a1a', edgecolor='gray', labelcolor='white')
    axes[1, 0].grid(True, alpha=0.2, color='gray')
    
    constellation_decimation = max(1, len(iq_complex) // 1000)
    axes[1, 1].scatter(iq_data[0, ::constellation_decimation][:1000], 
                      iq_data[1, ::constellation_decimation][:1000],
                      c='#06b6d4', s=1, alpha=0.6)
    axes[1, 1].set_xlabel('In-Phase (I)', color='white')
    axes[1, 1].set_ylabel('Quadrature (Q)', color='white')
    axes[1, 1].set_title('IQ Constellation', color='#06b6d4', fontsize=10)
    axes[1, 1].set_facecolor('#0a0a0a')
    axes[1, 1].tick_params(colors='white')
    axes[1, 1].grid(True, alpha=0.2, color='gray')
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, facecolor='#0a0a0a', edgecolor='none')
    img_buffer.seek(0)
    plt.close()
    
    img = Image(img_buffer, width=7*inch, height=5.6*inch)
    story.append(img)
    
    if hypotheses:
        story.append(PageBreak())
        story.append(Paragraph("HYPOTHESIS VALIDATION RESULTS", heading_style))
        
        hyp_data = [['Rank', 'Modulation', 'Symbol Rate', 'Confidence', 'Sync', 'Demod', 'FEC']]
        
        for i, hyp in enumerate(hypotheses[:10], 1):
            hyp_data.append([
                str(i),
                hyp.get('modulation', 'N/A'),
                f"{hyp.get('symbolRate', 0)/1000:.1f} ksps",
                f"{hyp.get('mlConfidence', 0)*100:.1f}%",
                '✓' if hyp.get('validation', {}).get('syncPassed') else '✗',
                '✓' if hyp.get('validation', {}).get('demodPassed') else '✗',
                '✓' if hyp.get('validation', {}).get('fecPassed') else '✗',
            ])
        
        hyp_table = Table(hyp_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1*inch, 0.6*inch, 0.6*inch, 0.6*inch])
        hyp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(hyp_table)
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<i>Generated by SIGMA - Signal Intelligence & Guided Modulation Analysis</i>", 
                          ParagraphStyle('Footer', parent=styles['Normal'], 
                                       fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
