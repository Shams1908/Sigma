import { motion, useScroll, useTransform, useMotionValue, useSpring, useInView } from 'framer-motion';
import { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Topography from '../components/Topography';

interface LandingViewProps {
  onLaunch: () => void;
}

export default function LandingView({ onLaunch }: LandingViewProps) {
  const navigate = useNavigate();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  
  const handleLaunch = () => {
    navigate('/workstation');
  };
  const heroRef = useRef(null);
  const problemRef = useRef(null);
  const pipelineRef = useRef(null);
  const impactRef = useRef(null);

  const heroInView = useInView(heroRef, { once: true, amount: 0.3 });
  const problemInView = useInView(problemRef, { once: true, amount: 0.2 });
  const pipelineInView = useInView(pipelineRef, { once: true, amount: 0.1 });
  const impactInView = useInView(impactRef, { once: true, amount: 0.1 });

  const { scrollYProgress } = useScroll();
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 100, damping: 30 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: e.clientX,
        y: e.clientY
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const magneticEffect = (ref: React.RefObject<HTMLElement>, strength = 20) => {
    if (!ref.current) return { x: 0, y: 0 };
    const rect = ref.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const distX = mousePosition.x - centerX;
    const distY = mousePosition.y - centerY;
    const distance = Math.sqrt(distX * distX + distY * distY);
    if (distance < 150) {
      const factor = (150 - distance) / 150;
      return { x: distX * factor * (strength / 150), y: distY * factor * (strength / 150) };
    }
    return { x: 0, y: 0 };
  };

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden antialiased">
      {/* WebGL Topography Background */}
      <div className="fixed inset-0 z-0">
        <Topography
          lowColor="#0a0014"
          midColor="#6d28d9"
          highColor="#d8b4fe"
          speed={0.35}
          bands={2.0}
          thickness={0.012}
          glow={0.8}
          pixelSize={1.0}
          mouseInteraction={true}
          mouseRadius={0.3}
          mouseStrength={0.4}
        />
        {/* Removed radial vignette - full brightness topography across entire screen */}
      </div>

      {/* Floating Solid Navbar */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
        className="fixed top-6 left-0 right-0 z-50 flex justify-center px-6"
      >
        <motion.div 
          className="bg-[#0A0A0A] border border-[#222222] rounded-full px-8 py-4 shadow-2xl shadow-purple-900/20 max-w-fit"
          style={{
            boxShadow: '0 0 40px rgba(109, 40, 217, 0.15), 0 20px 40px rgba(0,0,0,0.5)'
          }}
        >
          <div className="flex items-center gap-8">
            {/* Logo */}
            <motion.div 
              className="text-xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent"
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
            >
              SIGMA
            </motion.div>
            
            {/* Nav Links */}
            <div className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-400">
              {['Features', 'Pipeline', 'Impact'].map((item, i) => (
                <motion.a 
                  key={item}
                  href={`#${item.toLowerCase()}`} 
                  className="hover:text-white transition-colors duration-300 relative group whitespace-nowrap"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  whileHover={{ y: -2 }}
                >
                  {item}
                  <motion.div 
                    className="absolute -bottom-1 left-0 w-0 h-[2px] bg-purple-500 group-hover:w-full transition-all duration-300"
                  />
                </motion.a>
              ))}
            </div>
            
            {/* Launch Button */}
            <motion.button
              onClick={handleLaunch}
              className="bg-white text-black px-6 py-2 rounded-full font-semibold text-sm transition-all duration-300 relative overflow-hidden"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <span className="relative z-10">Launch</span>
              <motion.div 
                className="absolute inset-0 bg-gradient-to-r from-purple-400 to-purple-600"
                initial={{ x: '100%' }}
                whileHover={{ x: 0 }}
                transition={{ duration: 0.3 }}
              />
            </motion.button>
          </div>
        </motion.div>
      </motion.nav>

      <div className="relative z-10">
        {/* HERO SECTION */}
        <section 
          ref={heroRef}
          className="min-h-screen flex flex-col items-center justify-center px-6 relative"
        >
          <motion.div 
            className="max-w-6xl w-full space-y-12"
            initial={{ opacity: 0 }}
            animate={heroInView ? { opacity: 1 } : { opacity: 0 }}
            transition={{ duration: 1 }}
          >
            {/* Top Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={heroInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex justify-center"
            >
              <div className="bg-[#0A0A0A] border border-purple-800/50 rounded-full px-6 py-2 inline-flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                <span className="text-purple-400 font-mono text-sm tracking-wider">SIGMA</span>
              </div>
            </motion.div>

            {/* Massive Headline with Gradient Clip */}
            <div className="text-center">
              <motion.h1
                className="text-[5rem] md:text-[8rem] lg:text-[10rem] font-black leading-[0.85] tracking-tighter"
                initial={{ opacity: 0, y: 40 }}
                animate={heroInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
                transition={{ duration: 0.8, delay: 0.5 }}
              >
                {['Automated', 'signal', 'intelligence', '& analysis'].map((word, i) => (
                  <motion.span
                    key={word}
                    className="block bg-gradient-to-br from-white via-purple-200 to-purple-500 bg-clip-text text-transparent"
                    initial={{ opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
                    animate={heroInView ? { opacity: 1, x: 0 } : { opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
                    transition={{ duration: 0.8, delay: 0.7 + i * 0.1 }}
                    style={{
                      textShadow: '0 0 80px rgba(168, 85, 247, 0.4)'
                    }}
                  >
                    {word}
                  </motion.span>
                ))}
              </motion.h1>
            </div>

            {/* Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={heroInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 1.2 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              {/* Primary Button */}
              <motion.button
                onClick={handleLaunch}
                className="group relative px-10 py-4 bg-white text-black font-bold text-lg rounded-lg overflow-hidden"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  boxShadow: '0 0 40px rgba(255,255,255,0.3), 0 20px 40px rgba(0,0,0,0.5)'
                }}
              >
                <span className="relative z-10 flex items-center gap-2">
                  Launch Workstation
                  <motion.svg
                    className="w-5 h-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    initial={{ x: 0 }}
                    whileHover={{ x: 5 }}
                    transition={{ duration: 0.3 }}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </motion.svg>
                </span>
              </motion.button>

              {/* Secondary Button */}
              <motion.button
                className="px-10 py-4 bg-[#111111] text-white font-bold text-lg rounded-lg border border-[#333333] hover:border-purple-500 transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
                }}
              >
                View Documentation
              </motion.button>
            </motion.div>

            {/* Scroll Indicator */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={heroInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.6, delay: 1.5 }}
              className="absolute bottom-10 left-1/2 -translate-x-1/2"
            >
              <motion.div
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="flex flex-col items-center gap-2 text-gray-500"
              >
                <span className="text-xs font-mono tracking-wider">SCROLL</span>
                <div className="w-[2px] h-12 bg-gradient-to-b from-purple-500 to-transparent" />
              </motion.div>
            </motion.div>
          </motion.div>
        </section>

        {/* PROBLEM & SOLUTION SECTION */}
        <section 
          id="features"
          ref={problemRef}
          className="min-h-screen flex items-center justify-center px-6 py-20 relative"
        >
          <div className="max-w-7xl w-full">
            {/* Section Title */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={problemInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-20"
            >
              <h2 className="text-6xl md:text-7xl font-black text-center mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                The Problem &<br />Solution
              </h2>
              <p className="text-center text-gray-400 text-lg max-w-2xl mx-auto">
                Traditional RF analysis is broken. SIGMA fixes it.
              </p>
            </motion.div>

            {/* Comparison Cards */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Traditional Analysis Card */}
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -50 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="bg-[#0A0A0A] border border-[#222222] rounded-3xl p-10 relative overflow-hidden"
                style={{
                  boxShadow: '0 0 60px rgba(239, 68, 68, 0.1)'
                }}
              >
                {/* Red accent glow */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-red-500/10 rounded-full blur-[100px]" />
                
                <div className="relative z-10">
                  <div className="flex items-center gap-3 mb-8">
                    <motion.div 
                      className="w-3 h-3 bg-red-500 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    <h3 className="text-3xl font-bold text-red-400">Traditional Analysis</h3>
                  </div>

                  <ul className="space-y-5">
                    {[
                      'Trial-and-error manual parameter adjustment',
                      'Uncertainty chain: one wrong guess breaks everything downstream',
                      'AI confidence scores without physical validation',
                      'Hours spent on false positives and dead ends'
                    ].map((item, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: -20 }}
                        animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
                        transition={{ duration: 0.4, delay: 0.4 + i * 0.1 }}
                        className="flex items-start gap-4 text-gray-300"
                      >
                        <span className="text-red-400 text-xl mt-1 font-bold">×</span>
                        <span className="leading-relaxed">{item}</span>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>

              {/* SIGMA Approach Card */}
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 50 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="bg-[#0A0A0A] border border-[#222222] rounded-3xl p-10 relative overflow-hidden"
                style={{
                  boxShadow: '0 0 60px rgba(20, 184, 166, 0.15)'
                }}
              >
                {/* Teal accent glow */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-teal-500/10 rounded-full blur-[100px]" />
                
                <div className="relative z-10">
                  <div className="flex items-center gap-3 mb-8">
                    <motion.div 
                      className="w-3 h-3 bg-teal-400 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    <h3 className="text-3xl font-bold text-teal-400">SIGMA Approach</h3>
                  </div>

                  <ul className="space-y-5">
                    {[
                      'Automated Signal Hypothesis Engine',
                      'Closed-loop validation: ML proposes, demodulator proves',
                      'Forward Error Correction verification against real Viterbi decoding',
                      'Explainable evidence trail for every validated hypothesis'
                    ].map((item, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: 20 }}
                        animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 20 }}
                        transition={{ duration: 0.4, delay: 0.4 + i * 0.1 }}
                        className="flex items-start gap-4 text-gray-300"
                      >
                        <span className="text-teal-400 text-xl mt-1 font-bold">✓</span>
                        <span className="leading-relaxed">{item}</span>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            </div>

            {/* Uncertainty Chain Callout */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={problemInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.6, delay: 0.8 }}
              className="mt-12 bg-[#0A0A0A] border border-blue-900/50 rounded-2xl p-8"
              style={{
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.1)'
              }}
            >
              <h4 className="text-2xl font-bold text-blue-400 mb-4 flex items-center gap-3">
                <div className="w-2 h-8 bg-blue-500 rounded" />
                The Uncertainty Chain
              </h4>
              <p className="text-gray-300 leading-relaxed text-lg">
                In RF signal analysis, parameters are interdependent. If you guess the wrong sample rate, 
                your carrier frequency estimate will be off. If the carrier frequency is wrong, synchronization 
                fails. If synchronization fails, demodulation produces garbage. SIGMA breaks this chain by 
                systematically testing hypotheses and validating each stage with physical signal processing, 
                not just statistical confidence.
              </p>
            </motion.div>
          </div>
        </section>

        {/* PIPELINE SECTION */}
        <section 
          id="pipeline"
          ref={pipelineRef}
          className="min-h-screen flex items-center justify-center px-6 py-20"
        >
          <div className="max-w-7xl w-full">
            {/* Title */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={pipelineInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-20 text-center"
            >
              <h2 className="text-6xl md:text-7xl font-black mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                Signal Processing<br />Pipeline
              </h2>
            </motion.div>

            {/* Pipeline Stages - Bento Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { stage: 1, title: 'Ingestion', desc: 'Parse .IQ and .wav files, extract metadata, normalize sample format', color: 'blue' },
                { stage: 2, title: 'DSP Analysis', desc: 'FFT, PSD, spectrogram, bandwidth, SNR, carrier offset estimation', color: 'cyan' },
                { stage: 3, title: 'Feature Extraction', desc: 'Statistical, spectral, and cyclostationary features for ML classifier', color: 'teal' },
                { stage: 4, title: 'ML Classification', desc: 'Modulation recognition with confidence scoring', color: 'green' },
                { stage: 5, title: 'Synchronization', desc: 'Carrier recovery, timing recovery, matched filtering', color: 'yellow' },
                { stage: 6, title: 'Validation', desc: 'Demodulation attempt, FEC verification, hypothesis ranking', color: 'purple' }
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 30 }}
                  animate={pipelineInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
                  transition={{ duration: 0.5, delay: 0.2 + i * 0.1 }}
                  className="bg-[#0A0A0A] border border-[#222222] rounded-2xl p-8 relative overflow-hidden group hover:border-teal-900 transition-colors duration-300"
                  whileHover={{ y: -5 }}
                >
                  <div className={`absolute top-0 right-0 w-32 h-32 bg-${item.color}-500/5 rounded-full blur-[60px]`} />
                  
                  <div className="relative z-10">
                    <div className="text-teal-400 font-mono text-sm mb-3 tracking-wider">
                      STAGE {item.stage}
                    </div>
                    <h4 className="text-2xl font-bold text-white mb-4">
                      {item.title}
                    </h4>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      {item.desc}
                    </p>
                  </div>

                  {/* Hover effect line */}
                  <motion.div 
                    className="absolute bottom-0 left-0 h-[2px] w-0 bg-gradient-to-r from-teal-500 to-blue-500 group-hover:w-full transition-all duration-500"
                  />
                </motion.div>
              ))}
            </div>

            {/* Parameter Extraction */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={pipelineInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.6, delay: 0.8 }}
              className="mt-12 bg-[#0A0A0A] border border-[#222222] rounded-2xl p-10"
            >
              <h4 className="text-3xl font-bold text-white mb-8">Parameter Extraction</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                {[
                  'Sampling Frequency',
                  'Carrier Frequency',
                  'Modulation Type',
                  'Symbol Rate',
                  'FEC Scheme',
                  'Signal Bandwidth'
                ].map((param, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={pipelineInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
                    transition={{ duration: 0.3, delay: 1 + i * 0.05 }}
                    className="flex items-center gap-3"
                  >
                    <motion.div 
                      className="w-2 h-2 bg-teal-400 rounded-full"
                      animate={{ scale: [1, 1.5, 1] }}
                      transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                    />
                    <span className="text-gray-300 font-mono text-sm">{param}</span>
                  </motion.div>
                ))}
              </div>

              <div className="mt-10 pt-8 border-t border-[#222222]">
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
                  <h5 className="text-xl font-bold text-blue-400 mb-3">MVP Validation Scope</h5>
                  <p className="text-gray-300 leading-relaxed">
                    The minimum viable product focuses on BPSK and QPSK modulation schemes with convolutional 
                    FEC codes. Each hypothesis undergoes real Viterbi decoding with syndrome checking to confirm 
                    validity. This provides a concrete foundation for expanding to higher-order modulations.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* IMPACT SECTION */}
        <section 
          id="impact"
          ref={impactRef}
          className="min-h-screen flex items-center justify-center px-6 py-20"
        >
          <div className="max-w-7xl w-full">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={impactInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-16 text-center"
            >
              <h2 className="text-6xl md:text-7xl font-black mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                Impact &<br />Benefits
              </h2>
            </motion.div>

            {/* Core USP */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={impactInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="mb-12 bg-gradient-to-br from-[#0A0A0A] to-[#111111] border border-teal-900/50 rounded-3xl p-12 relative overflow-hidden"
              style={{
                boxShadow: '0 0 80px rgba(20, 184, 166, 0.2)'
              }}
            >
              <div className="absolute top-0 right-0 w-96 h-96 bg-teal-500/10 rounded-full blur-[120px]" />
              
              <div className="relative z-10 text-center mb-10">
                <motion.div 
                  className="inline-block bg-teal-400/10 border border-teal-400/30 rounded-full px-6 py-2 mb-6"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 3, repeat: Infinity }}
                >
                  <span className="text-teal-400 font-bold text-sm tracking-wider">CORE USP</span>
                </motion.div>
                <h3 className="text-4xl md:text-5xl font-black text-white mb-6 leading-tight">
                  Closed-Loop Signal<br />Hypothesis Validation
                </h3>
                <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
                  SIGMA doesn't just predict modulation schemes—it proves them. Every hypothesis is 
                  physically tested through the complete signal chain: synchronization, demodulation, 
                  and forward error correction.
                </p>
              </div>

              {/* Benefits Grid */}
              <div className="grid md:grid-cols-2 gap-6">
                {[
                  {
                    title: 'Proof Over Confidence',
                    desc: 'Traditional systems output percentages like "82% QPSK" without verification. SIGMA attempts actual demodulation and FEC decoding. If Viterbi syndrome checks pass, the hypothesis is proven—not guessed.'
                  },
                  {
                    title: 'Explainable Evidence',
                    desc: 'Every validated signal comes with a complete evidence trail: which synchronization method succeeded, what demodulator configuration worked, and which FEC parameters decoded cleanly.'
                  },
                  {
                    title: 'Automation at Scale',
                    desc: 'Analysts can process hundreds of unknown signals without manual parameter tuning. The hypothesis engine explores the parameter space systematically, testing combinations that humans might never consider.'
                  },
                  {
                    title: 'Reduced False Positives',
                    desc: 'By requiring physical demodulation success, SIGMA eliminates the false confidence of pure ML classifiers. A hypothesis either fully decodes or it does not - there is no ambiguity.'
                  }
                ].map((benefit, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    animate={impactInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                    transition={{ duration: 0.5, delay: 0.6 + i * 0.1 }}
                    className="bg-[#0A0A0A] border border-[#222222] rounded-xl p-6 hover:border-teal-900 transition-colors duration-300"
                  >
                    <h4 className="text-xl font-bold text-teal-400 mb-3">{benefit.title}</h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{benefit.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Final CTA */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={impactInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.6, delay: 1 }}
              className="text-center"
            >
              <motion.button
                onClick={handleLaunch}
                className="group relative px-12 py-5 bg-white text-black font-bold text-xl rounded-lg overflow-hidden"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  boxShadow: '0 0 60px rgba(255,255,255,0.4), 0 20px 60px rgba(0,0,0,0.6)'
                }}
              >
                <span className="relative z-10 flex items-center gap-3">
                  Launch Workstation
                  <motion.svg
                    className="w-6 h-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </motion.svg>
                </span>
              </motion.button>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="relative z-10 border-t border-[#222222] py-12 mt-20">
          <div className="max-w-7xl mx-auto px-6 text-center">
            <motion.p 
              className="text-gray-500 text-sm font-mono"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
            >
              SIGMA — Signal Intelligence & Guided Modulation Analysis
            </motion.p>
          </div>
        </footer>
      </div>
    </div>
  );
}
