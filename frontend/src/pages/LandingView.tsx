import { useState, useEffect } from 'react';
import AeroShards from '../components/AeroShards';
import Particles from '../components/Particles';
import Topography from '../components/Topography';

interface LandingViewProps {
  onLaunch: () => void;
}

export default function LandingView({ onLaunch }: LandingViewProps) {
  const [activeSection, setActiveSection] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      const windowHeight = window.innerHeight;
      const section = Math.floor(scrollPosition / windowHeight);
      setActiveSection(section);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="relative bg-slate-900 text-slate-200">
      <header className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-teal-400 to-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-slate-900 font-bold text-sm">Σ</span>
            </div>
            <span className="text-xl font-bold text-white tracking-tight">SIGMA</span>
          </div>
          
          <nav className="hidden md:flex items-center gap-8">
            <a href="#hero" className="text-sm text-slate-400 hover:text-teal-400 transition-colors duration-200">Home</a>
            <a href="#problem" className="text-sm text-slate-400 hover:text-teal-400 transition-colors duration-200">Problem</a>
            <a href="#feasibility" className="text-sm text-slate-400 hover:text-teal-400 transition-colors duration-200">Feasibility</a>
            <a href="#impact" className="text-sm text-slate-400 hover:text-teal-400 transition-colors duration-200">Impact</a>
          </nav>

          <button
            onClick={onLaunch}
            className="px-6 py-2 bg-teal-500 hover:bg-teal-400 text-slate-900 font-semibold text-sm rounded-lg transition-all duration-200 transform hover:scale-105"
          >
            Launch
          </button>
        </div>
      </header>

      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 transition-opacity duration-[1500ms] ease-in-out" style={{ opacity: activeSection === 0 ? 1 : 0 }}>
          <AeroShards />
        </div>
        <div className="absolute inset-0 transition-opacity duration-[1500ms] ease-in-out" style={{ opacity: activeSection === 1 ? 1 : 0 }}>
          <Particles />
        </div>
        <div className="absolute inset-0 transition-opacity duration-[1500ms] ease-in-out" style={{ opacity: activeSection === 2 ? 1 : 0 }}>
          <Topography />
        </div>
        <div className="absolute inset-0 transition-opacity duration-[1500ms] ease-in-out" style={{ opacity: activeSection === 3 ? 1 : 0 }}>
          <Particles />
        </div>
      </div>
      
      <div className="relative z-10 pt-16">
        <section id="hero" className="min-h-screen flex flex-col items-center justify-center px-6 snap-start">
          <div className="max-w-4xl text-center space-y-8 transform transition-all duration-1000 opacity-0 translate-y-10 animate-[fadeInUp_1s_ease-out_forwards]">
            <h1 className="text-7xl font-bold tracking-tight text-white mb-4 animate-[fadeIn_1.5s_ease-out]">
              SIGMA
            </h1>
            <p className="text-2xl text-slate-300 font-light leading-relaxed animate-[fadeIn_1.5s_ease-out_0.3s_both]">
              Automated model for analysis of .IQ and .wav files
            </p>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed animate-[fadeIn_1.5s_ease-out_0.6s_both]">
              A signals intelligence platform that validates radio frequency modulation schemes through closed-loop hypothesis testing and forward error correction verification.
            </p>
            <button
              onClick={onLaunch}
              className="mt-12 px-12 py-4 bg-teal-500 hover:bg-teal-400 text-slate-900 font-semibold text-lg rounded-lg transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-teal-500/30 animate-[fadeIn_1.5s_ease-out_0.9s_both]"
            >
              Launch Workstation
            </button>
            <div className="mt-16 animate-bounce">
              <svg className="w-8 h-8 mx-auto text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
          </div>
        </section>

        <section id="problem" className="min-h-screen flex items-center justify-center px-6 py-20 snap-start">
          <div className="max-w-6xl w-full">
            <h2 className="text-4xl font-bold text-center mb-16 text-white transform transition-all duration-700 opacity-0 translate-y-10" 
                style={{ animation: activeSection >= 1 ? 'fadeInUp 0.8s ease-out forwards' : 'none' }}>
              The Problem & Solution
            </h2>
            
            <div className="grid md:grid-cols-2 gap-12">
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-8 border border-red-500/30 transform transition-all duration-700 hover:scale-105 hover:border-red-400/50"
                   style={{ animation: activeSection >= 1 ? 'fadeInLeft 0.8s ease-out 0.2s forwards' : 'none', opacity: activeSection >= 1 ? 1 : 0 }}>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                  <h3 className="text-2xl font-bold text-red-400">Traditional Analysis</h3>
                </div>
                <ul className="space-y-4 text-slate-300">
                  <li className="flex items-start gap-3">
                    <span className="text-red-400 mt-1">×</span>
                    <span>Trial-and-error manual parameter adjustment</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-red-400 mt-1">×</span>
                    <span>Uncertainty chain: one wrong guess breaks everything downstream</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-red-400 mt-1">×</span>
                    <span>AI confidence scores without physical validation</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-red-400 mt-1">×</span>
                    <span>Hours spent on false positives and dead ends</span>
                  </li>
                </ul>
              </div>

              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-8 border border-teal-500/30 transform transition-all duration-700 hover:scale-105 hover:border-teal-400/50"
                   style={{ animation: activeSection >= 1 ? 'fadeInRight 0.8s ease-out 0.4s forwards' : 'none', opacity: activeSection >= 1 ? 1 : 0 }}>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-3 h-3 bg-teal-400 rounded-full animate-pulse"></div>
                  <h3 className="text-2xl font-bold text-teal-400">SIGMA Approach</h3>
                </div>
                <ul className="space-y-4 text-slate-300">
                  <li className="flex items-start gap-3">
                    <span className="text-teal-400 mt-1">✓</span>
                    <span>Automated Signal Hypothesis Engine</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-teal-400 mt-1">✓</span>
                    <span>Closed-loop validation: ML proposes, demodulator proves</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-teal-400 mt-1">✓</span>
                    <span>Forward Error Correction verification against real Viterbi decoding</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-teal-400 mt-1">✓</span>
                    <span>Explainable evidence trail for every validated hypothesis</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="mt-12 bg-slate-800/30 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/20 transform transition-all duration-700 hover:border-blue-400/40"
                 style={{ animation: activeSection >= 1 ? 'fadeInUp 0.8s ease-out 0.6s forwards' : 'none', opacity: activeSection >= 1 ? 1 : 0 }}>
              <h4 className="text-xl font-bold text-blue-400 mb-4">The Uncertainty Chain</h4>
              <p className="text-slate-300 leading-relaxed">
                In RF signal analysis, parameters are interdependent. If you guess the wrong sample rate, your carrier frequency estimate will be off. If the carrier frequency is wrong, synchronization fails. If synchronization fails, demodulation produces garbage. SIGMA breaks this chain by systematically testing hypotheses and validating each stage with physical signal processing, not just statistical confidence.
              </p>
            </div>
          </div>
        </section>

        <section id="feasibility" className="min-h-screen flex items-center justify-center px-6 py-20 snap-start">
          <div className="max-w-6xl w-full">
            <h2 className="text-4xl font-bold text-center mb-16 text-white transform transition-all duration-700"
                style={{ animation: activeSection >= 2 ? 'fadeInUp 0.8s ease-out forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
              Feasibility & Viability
            </h2>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-10 border border-slate-700/50 transform transition-all duration-700"
                 style={{ animation: activeSection >= 2 ? 'fadeIn 0.8s ease-out 0.2s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
              <h3 className="text-2xl font-bold text-teal-400 mb-8 text-center">
                Signal Processing Pipeline
              </h3>

              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.3s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 1</div>
                  <h4 className="text-lg font-bold text-white mb-3">Ingestion</h4>
                  <p className="text-slate-400 text-sm">
                    Parse .IQ and .wav files, extract metadata, normalize sample format
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.4s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 2</div>
                  <h4 className="text-lg font-bold text-white mb-3">DSP Analysis</h4>
                  <p className="text-slate-400 text-sm">
                    FFT, PSD, spectrogram, bandwidth, SNR, carrier offset estimation
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.5s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 3</div>
                  <h4 className="text-lg font-bold text-white mb-3">Feature Extraction</h4>
                  <p className="text-slate-400 text-sm">
                    Statistical, spectral, and cyclostationary features for ML classifier
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.6s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 4</div>
                  <h4 className="text-lg font-bold text-white mb-3">ML Classification</h4>
                  <p className="text-slate-400 text-sm">
                    Modulation recognition with confidence scoring
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.7s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 5</div>
                  <h4 className="text-lg font-bold text-white mb-3">Synchronization</h4>
                  <p className="text-slate-400 text-sm">
                    Carrier recovery, timing recovery, matched filtering
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 transform transition-all duration-500 hover:scale-105 hover:border-teal-400/50 hover:shadow-lg hover:shadow-teal-500/20"
                     style={{ animation: activeSection >= 2 ? 'fadeInUp 0.6s ease-out 0.8s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                  <div className="text-blue-400 font-mono text-sm mb-2">STAGE 6</div>
                  <h4 className="text-lg font-bold text-white mb-3">Validation</h4>
                  <p className="text-slate-400 text-sm">
                    Demodulation attempt, FEC verification, hypothesis ranking
                  </p>
                </div>
              </div>

              <div className="mt-10 pt-8 border-t border-slate-700"
                   style={{ animation: activeSection >= 2 ? 'fadeIn 0.8s ease-out 1s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                <h4 className="text-xl font-bold text-white mb-6">Parameter Extraction</h4>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">Sampling Frequency</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">Carrier Frequency</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">Modulation Type</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">Symbol Rate</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">FEC Scheme</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-400 rounded-full"></div>
                    <span className="text-slate-300 font-mono text-sm">Signal Bandwidth</span>
                  </div>
                </div>
              </div>

              <div className="mt-8 bg-blue-500/10 border border-blue-500/30 rounded-xl p-6 transform transition-all duration-700 hover:border-blue-400/50"
                   style={{ animation: activeSection >= 2 ? 'fadeInUp 0.8s ease-out 1.2s forwards' : 'none', opacity: activeSection >= 2 ? 1 : 0 }}>
                <h4 className="text-lg font-bold text-blue-400 mb-3">MVP Validation Scope</h4>
                <p className="text-slate-300 leading-relaxed">
                  The minimum viable product focuses on BPSK and QPSK modulation schemes with convolutional FEC codes. Each hypothesis undergoes real Viterbi decoding with syndrome checking to confirm validity. This provides a concrete foundation for expanding to higher-order modulations.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section id="impact" className="min-h-screen flex items-center justify-center px-6 py-20 snap-start">
          <div className="max-w-6xl w-full">
            <h2 className="text-4xl font-bold text-center mb-16 text-white transform transition-all duration-700"
                style={{ animation: activeSection >= 3 ? 'fadeInUp 0.8s ease-out forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
              Impact & Benefits
            </h2>

            <div className="bg-gradient-to-br from-teal-500/20 to-blue-500/20 backdrop-blur-sm rounded-3xl p-12 border border-teal-500/30 mb-12 transform transition-all duration-700 hover:scale-[1.02]"
                 style={{ animation: activeSection >= 3 ? 'fadeIn 0.8s ease-out 0.2s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
              <div className="text-center mb-8">
                <div className="inline-block bg-teal-400/10 border border-teal-400/30 rounded-full px-6 py-2 mb-6 animate-pulse">
                  <span className="text-teal-400 font-bold text-sm tracking-wider">CORE USP</span>
                </div>
                <h3 className="text-3xl font-bold text-white mb-4">
                  Closed-Loop Signal Hypothesis Validation
                </h3>
                <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
                  SIGMA doesn't just predict modulation schemes—it proves them. Every hypothesis is physically tested through the complete signal chain: synchronization, demodulation, and forward error correction.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6 mt-10">
                <div className="bg-slate-900/50 rounded-xl p-6 transform transition-all duration-500 hover:scale-105 hover:border hover:border-teal-400/30"
                     style={{ animation: activeSection >= 3 ? 'fadeInLeft 0.6s ease-out 0.4s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
                  <h4 className="text-lg font-bold text-teal-400 mb-3">Proof Over Confidence</h4>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    Traditional systems output percentages like "82% QPSK" without verification. SIGMA attempts actual demodulation and FEC decoding. If Viterbi syndrome checks pass, the hypothesis is proven—not guessed.
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 transform transition-all duration-500 hover:scale-105 hover:border hover:border-teal-400/30"
                     style={{ animation: activeSection >= 3 ? 'fadeInRight 0.6s ease-out 0.5s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
                  <h4 className="text-lg font-bold text-teal-400 mb-3">Explainable Evidence</h4>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    Every validated signal comes with a complete evidence trail: which synchronization method succeeded, what demodulator configuration worked, and which FEC parameters decoded cleanly.
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 transform transition-all duration-500 hover:scale-105 hover:border hover:border-teal-400/30"
                     style={{ animation: activeSection >= 3 ? 'fadeInLeft 0.6s ease-out 0.6s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
                  <h4 className="text-lg font-bold text-teal-400 mb-3">Automation at Scale</h4>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    Analysts can process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically, testing combinations that humans might never consider.
                  </p>
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 transform transition-all duration-500 hover:scale-105 hover:border hover:border-teal-400/30"
                     style={{ animation: activeSection >= 3 ? 'fadeInRight 0.6s ease-out 0.7s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
                  <h4 className="text-lg font-bold text-teal-400 mb-3">Reduced False Positives</h4>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. A hypothesis either fully decodes or it doesn't—there's no ambiguity.
                  </p>
                </div>
              </div>
            </div>

            <div className="text-center"
                 style={{ animation: activeSection >= 3 ? 'fadeInUp 0.8s ease-out 0.9s forwards' : 'none', opacity: activeSection >= 3 ? 1 : 0 }}>
              <button
                onClick={onLaunch}
                className="px-12 py-4 bg-teal-500 hover:bg-teal-400 text-slate-900 font-semibold text-lg rounded-lg transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-teal-500/30"
              >
                Launch Workstation
              </button>
            </div>
          </div>
        </section>

        <footer className="relative z-10 border-t border-slate-800 py-8">
          <div className="max-w-6xl mx-auto px-6 text-center text-slate-500 text-sm">
            <p>SIGMA — Signal Intelligence & Guided Modulation Analysis</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
