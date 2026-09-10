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

  // Parallax offsets — each section heading drifts upward as the page scrolls,
  // giving the continuous background a sense of depth. Ranges are intentionally
  // subtle so they never fight the WebGL background.
  const problemHeadingY  = useTransform(smoothProgress, [0.05, 0.35], [24, -24]);
  const pipelineHeadingY = useTransform(smoothProgress, [0.30, 0.65], [24, -24]);
  const impactHeadingY   = useTransform(smoothProgress, [0.60, 0.95], [24, -24]);

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
              WaveSight
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
              {/* About WaveSight — links to the existing /info page */}
              <motion.a
                href="/info#about"
                className="hover:text-white transition-colors duration-300 relative group whitespace-nowrap"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                whileHover={{ y: -2 }}
              >
                About WaveSight
                <motion.div
                  className="absolute -bottom-1 left-0 w-0 h-[2px] bg-purple-500 group-hover:w-full transition-all duration-300"
                />
              </motion.a>
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
                <span className="text-purple-400 font-mono text-sm tracking-wider">WaveSight</span>
              </div>
            </motion.div>

            {/* Massive Headline with Gradient Clip */}
            <div className="text-center">
              <motion.h1
                className="text-[3rem] md:text-[5rem] lg:text-[6rem] font-bold leading-[1.05] tracking-wide"
                style={{ fontFamily: '"Orbitron", sans-serif', fontWeight: 700, letterSpacing: '0.03em' }}
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

            {/* Supporting text */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={heroInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.9, delay: 1.15, ease: 'easeOut' }}
              className="text-center text-gray-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed"
            >
              From raw RF data to validated signal hypotheses, WaveSight structures
              and automates the analysis path.
            </motion.p>

            {/* Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={heroInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 1.4 }}
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
            </motion.div>

            {/* Scroll Indicator */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={heroInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.6, delay: 1.8 }}
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
            {/* Section Title — parallax on scroll */}
            <motion.div
              style={{ y: problemHeadingY }}
              initial={{ opacity: 0, y: 30 }}
              animate={problemInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-20"
            >
              <h2 className="text-6xl md:text-7xl font-black text-center mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                The Problem &<br />Solution
              </h2>
              <p className="text-center text-gray-400 text-lg max-w-2xl mx-auto">
                RF analysis often depends on iterative parameter tuning and hypotheses
                that are difficult to validate. WaveSight turns that process into a
                structured, evidence-driven pipeline.
              </p>
            </motion.div>

            {/* Comparison Cards */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Traditional Analysis Card — enters from left with perspective tilt */}
              <motion.div
                initial={{ opacity: 0, x: -60, rotateY: -8 }}
                animate={problemInView ? { opacity: 1, x: 0, rotateY: 0 } : { opacity: 0, x: -60, rotateY: -8 }}
                transition={{ duration: 0.7, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
                className="bg-[#0A0A0A] border border-[#222222] rounded-3xl p-10 relative overflow-hidden"
                style={{
                  boxShadow: '0 0 60px rgba(239, 68, 68, 0.1)',
                  transformPerspective: 1200,
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

                  <ul className="space-y-6">
                    {[
                      {
                        title: 'Trial-and-error parameter tuning',
                        sub:   'Analysts repeatedly adjust parameters to find a workable interpretation.',
                      },
                      {
                        title: 'Uncertainty propagates downstream',
                        sub:   'An incorrect early assumption can affect every later processing stage.',
                      },
                      {
                        title: 'Hypotheses are difficult to validate',
                        sub:   'A modulation guess may look plausible without proving it is correct.',
                      },
                      {
                        title: 'False positives create dead ends',
                        sub:   'Incorrect interpretations lead to wasted analysis time and repeated manual tuning.',
                      },
                    ].map((item, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: -16 }}
                        animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -16 }}
                        transition={{ duration: 0.45, delay: 0.35 + i * 0.12, ease: 'easeOut' }}
                        className="flex items-start gap-4"
                      >
                        <span className="text-red-400 text-xl mt-0.5 font-bold flex-shrink-0">×</span>
                        <div>
                          <p className="text-white font-semibold leading-snug">{item.title}</p>
                          <p className="text-gray-400 text-sm leading-relaxed mt-0.5">{item.sub}</p>
                        </div>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>

              {/* SIGMA Approach Card — enters from right with perspective tilt */}
              <motion.div
                initial={{ opacity: 0, x: 60, rotateY: 8 }}
                animate={problemInView ? { opacity: 1, x: 0, rotateY: 0 } : { opacity: 0, x: 60, rotateY: 8 }}
                transition={{ duration: 0.7, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  boxShadow: '0 0 60px rgba(20, 184, 166, 0.15)',
                  transformPerspective: 1200,
                }}
                className="bg-[#0A0A0A] border border-[#222222] rounded-3xl p-10 relative overflow-hidden"
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
                    <h3 className="text-3xl font-bold text-teal-400">WaveSight Approach</h3>
                  </div>

                  <ul className="space-y-6">
                    {[
                      {
                        title: 'Automated signal analysis',
                        sub:   'Extracts measurable characteristics from raw IQ/WAV data to establish a reliable starting point.',
                      },
                      {
                        title: 'Hypothesis-driven processing',
                        sub:   'Uses extracted evidence to investigate plausible signal interpretations instead of relying only on manual trial and error.',
                      },
                      {
                        title: 'Closed-loop validation',
                        sub:   'Candidate interpretations are tested through downstream synchronization, demodulation and reconstruction evidence.',
                      },
                      {
                        title: 'Explainable evidence',
                        sub:   'Validated hypotheses are supported by observable signal-processing results rather than unexplained confidence scores.',
                      },
                    ].map((item, i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: 16 }}
                        animate={problemInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 16 }}
                        transition={{ duration: 0.45, delay: 0.45 + i * 0.12, ease: 'easeOut' }}
                        className="flex items-start gap-4"
                      >
                        <span className="text-teal-400 text-xl mt-0.5 font-bold flex-shrink-0">✓</span>
                        <div>
                          <p className="text-white font-semibold leading-snug">{item.title}</p>
                          <p className="text-gray-400 text-sm leading-relaxed mt-0.5">{item.sub}</p>
                        </div>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            </div>

            {/* Uncertainty Chain Callout — fades in after the cards settle */}
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={problemInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
              transition={{ duration: 0.7, delay: 1.05, ease: 'easeOut' }}
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
                fails. If synchronization fails, demodulation produces garbage. WaveSight breaks this chain by 
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
            {/* Title — parallax on scroll */}
            <motion.div
              style={{ y: pipelineHeadingY }}
              initial={{ opacity: 0, y: 30 }}
              animate={pipelineInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-20 text-center"
            >
              <h2 className="text-6xl md:text-7xl font-black mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                Signal Processing<br />Pipeline
              </h2>
            </motion.div>

            {/* Pipeline Stages — each card clips up from below with increasing delay */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                {
                  stage: 1, color: 'blue',
                  title: 'Ingestion',
                  desc: 'Load IQ/WAV recordings and prepare raw samples for analysis.',
                },
                {
                  stage: 2, color: 'cyan',
                  title: 'DSP Analysis',
                  desc: 'Analyze time- and frequency-domain behaviour to characterize the signal using FFT, PSD, and spectrogram techniques.',
                },
                {
                  stage: 3, color: 'teal',
                  title: 'Feature Extraction',
                  desc: 'Extract measurable signal characteristics that guide downstream hypothesis generation.',
                },
                {
                  stage: 4, color: 'green',
                  title: 'ML Classification',
                  desc: 'Use extracted features to assist modulation recognition and generate candidate hypotheses.',
                },
                {
                  stage: 5, color: 'yellow',
                  title: 'Synchronization',
                  desc: 'Recover timing and carrier alignment required for reliable downstream processing.',
                },
                {
                  stage: 6, color: 'purple',
                  title: 'Validation',
                  desc: 'Test candidate interpretations against downstream processing and reconstruction evidence.',
                },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 40, scale: 0.97 }}
                  animate={pipelineInView
                    ? { opacity: 1, y: 0, scale: 1 }
                    : { opacity: 0, y: 40, scale: 0.97 }}
                  transition={{
                    duration: 0.55,
                    delay: 0.1 + i * 0.09,
                    ease: [0.22, 1, 0.36, 1],
                  }}
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
              <h4 className="text-3xl font-bold text-white mb-3">Parameter Extraction</h4>
              <p className="text-gray-400 text-sm leading-relaxed mb-8">
                Convert raw signal observations into measurable parameters that guide hypothesis generation and validation.
              </p>
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
            {/* Heading — parallax on scroll */}
            <motion.div
              style={{ y: impactHeadingY }}
              initial={{ opacity: 0, y: 30 }}
              animate={impactInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
              transition={{ duration: 0.8 }}
              className="mb-16 text-center"
            >
              <h2 className="text-6xl md:text-7xl font-black mb-4 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent">
                Impact &<br />Benefits
              </h2>
            </motion.div>

            {/* Core USP — introductory card, visually distinct, keeps existing entrance animation */}
            <motion.div
              initial={{ opacity: 0, scale: 0.96, filter: 'blur(6px)' }}
              animate={impactInView
                ? { opacity: 1, scale: 1, filter: 'blur(0px)' }
                : { opacity: 0, scale: 0.96, filter: 'blur(6px)' }}
              transition={{ duration: 0.9, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              className="mb-8 bg-gradient-to-br from-[#0A0A0A] to-[#111111] border border-teal-900/50 rounded-3xl p-12 relative overflow-hidden"
              style={{ boxShadow: '0 0 80px rgba(20, 184, 166, 0.2)' }}
            >
              <div className="absolute top-0 right-0 w-96 h-96 bg-teal-500/10 rounded-full blur-[120px]" />

              <div className="relative z-10 text-center">
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
                  WaveSight does not stop at predicting a modulation scheme. Candidate hypotheses
                  are carried through the signal-processing chain and tested against downstream
                  evidence before being considered validated.
                </p>
              </div>
            </motion.div>

            {/* Four individual benefit cards — 2×2 grid, styled like Pipeline cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
              {[
                {
                  title: 'Proof Over Confidence',
                  desc: 'Replace unexplained prediction confidence with evidence from measurable signal-processing outcomes. A hypothesis is not treated as correct simply because a model considers it likely; it must produce consistent downstream results.',
                },
                {
                  title: 'Explainable Evidence',
                  desc: 'Trace why a signal hypothesis was accepted or rejected through observable processing results. WaveSight connects the hypothesis to the evidence produced throughout the analysis chain instead of presenting an unexplained final prediction.',
                },
                {
                  title: 'Automation at Scale',
                  desc: 'Reduce repetitive manual tuning by structuring the signal-analysis process into a consistent pipeline. This makes complex RF analysis more systematic and scalable across multiple recordings.',
                },
                {
                  title: 'Reduced False Positives',
                  desc: 'Use downstream validation to distinguish plausible hypotheses from interpretations that fail during processing. This helps prevent incorrect signal classifications from being treated as validated results.',
                },
              ].map((benefit, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 30, scale: 0.97 }}
                  animate={impactInView
                    ? { opacity: 1, y: 0, scale: 1 }
                    : { opacity: 0, y: 30, scale: 0.97 }}
                  transition={{
                    duration: 0.55,
                    delay: 0.4 + i * 0.1,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  className="bg-[#0A0A0A] border border-[#222222] rounded-2xl p-8 relative overflow-hidden group hover:border-teal-900 transition-colors duration-300"
                  whileHover={{ y: -5 }}
                >
                  {/* Subtle accent glow — same pattern as Pipeline cards */}
                  <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/5 rounded-full blur-[60px]" />

                  <div className="relative z-10">
                    <h4 className="text-2xl font-bold text-teal-400 mb-4">{benefit.title}</h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{benefit.desc}</p>
                  </div>

                  {/* Bottom hover line — same as Pipeline cards */}
                  <motion.div
                    className="absolute bottom-0 left-0 h-[2px] w-0 bg-gradient-to-r from-teal-500 to-blue-500 group-hover:w-full transition-all duration-500"
                  />
                </motion.div>
              ))}
            </div>

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

        {/* ── NEW SIGMA FOOTER ── */}
        <footer className="relative z-10 border-t border-[#222222] mt-20">
          <div className="max-w-7xl mx-auto px-6 py-12">
            {/* Top row: brand + nav links */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 mb-10">
              {/* Brand */}
              <div className="flex-shrink-0">
                <div className="text-xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent mb-1">
                  WaveSight
                </div>
                <p className="text-gray-500 text-xs font-mono tracking-wide">
                  Signal Intelligence &amp; Guided Modulation Analysis
                </p>
              </div>

              {/* Navigation links */}
              <nav className="flex flex-wrap gap-x-8 gap-y-3">
                {[
                  { label: 'About WaveSight',     href: '/info#about'     },
                  { label: 'Problem Statement',  href: '/info#problem'   },
                  { label: 'Problem & Solution', href: '/info#solution'  },
                  { label: 'Use Cases',          href: '/info#use-cases' },
                  { label: 'Privacy Policy',     href: '/info#privacy'   },
                ].map(({ label, href }) => (
                  <a
                    key={href}
                    href={href}
                    className="text-gray-400 text-sm hover:text-purple-400 transition-colors duration-200 whitespace-nowrap"
                  >
                    {label}
                  </a>
                ))}
              </nav>
            </div>

            {/* Divider */}
            <div className="border-t border-[#1a1a1a]" />

            {/* Bottom row: copyright */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-6">
              <motion.p
                className="text-gray-600 text-xs font-mono"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                transition={{ duration: 0.6 }}
              >
                © {new Date().getFullYear()} WaveSight Project. Prototype / research demonstration.
              </motion.p>
              <a
                href="/info"
                className="text-gray-600 text-xs font-mono hover:text-purple-400 transition-colors duration-200"
              >
                Info &amp; Policies ↗
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
