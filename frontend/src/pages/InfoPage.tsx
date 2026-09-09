import { motion, useInView } from 'framer-motion';
import { useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Topography from '../components/Topography';

// ─── Shared reveal animation ────────────────────────────────────────────────
// Reuses the same framer-motion pattern as the landing page: fade + slight
// upward drift, fires once when the element enters the viewport.
function RevealSection({
  children,
  delay = 0,
  className = '',
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.15 });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
      transition={{ duration: 0.65, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ─── Small card component — matches existing Pipeline / Impact card style ───
function InfoCard({
  title,
  body,
  accent = 'teal',
  delay = 0,
}: {
  title: string;
  body: string;
  accent?: 'teal' | 'purple' | 'blue' | 'red';
  delay?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.2 });

  const accentClasses: Record<string, string> = {
    teal:   'text-teal-400',
    purple: 'text-purple-400',
    blue:   'text-blue-400',
    red:    'text-red-400',
  };
  const hoverBorder: Record<string, string> = {
    teal:   'hover:border-teal-900',
    purple: 'hover:border-purple-900',
    blue:   'hover:border-blue-900',
    red:    'hover:border-red-900',
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30, scale: 0.97 }}
      animate={inView ? { opacity: 1, y: 0, scale: 1 } : { opacity: 0, y: 30, scale: 0.97 }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`bg-[#0A0A0A] border border-[#222222] rounded-2xl p-8 relative overflow-hidden
                  group transition-colors duration-300 ${hoverBorder[accent]}`}
      whileHover={{ y: -4 }}
    >
      {/* Subtle corner glow — same as pipeline cards */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/5 rounded-full blur-[60px]" />
      <div className="relative z-10">
        <h4 className={`text-xl font-bold mb-3 ${accentClasses[accent]}`}>{title}</h4>
        <p className="text-gray-400 text-sm leading-relaxed">{body}</p>
      </div>
      {/* Bottom hover sweep line — same as pipeline cards */}
      <motion.div className="absolute bottom-0 left-0 h-[2px] w-0 bg-gradient-to-r from-teal-500 to-blue-500 group-hover:w-full transition-all duration-500" />
    </motion.div>
  );
}

// ─── Section heading — same gradient style as landing page h2 elements ──────
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-4xl md:text-5xl font-black mb-6 bg-gradient-to-br from-white to-gray-500 bg-clip-text text-transparent leading-tight">
      {children}
    </h2>
  );
}

// ─── Pipeline step visual for the Solution section ──────────────────────────
const PIPELINE_STEPS = [
  { label: 'RAW IQ / WAV',                  note: 'Input recordings'                           },
  { label: 'SIGNAL ANALYSIS',               note: 'DSP characterization (FFT, PSD, spectrogram)' },
  { label: 'FEATURE EXTRACTION',            note: 'Measurable signal characteristics'           },
  { label: 'HYPOTHESIS GENERATION',         note: 'Candidate modulation interpretations'        },
  { label: 'SYNCHRONIZATION & PROCESSING',  note: 'Carrier / timing recovery'                  },
  { label: 'VALIDATION',                    note: 'Downstream evidence testing'                 },
];

// ─── Main page ───────────────────────────────────────────────────────────────
export default function InfoPage() {
  const navigate   = useNavigate();
  const location   = useLocation();

  // Smooth-scroll to the anchor after the page mounts (hash navigation)
  useEffect(() => {
    if (!location.hash) return;
    const id = location.hash.replace('#', '');
    // Give the page a tick to paint before scrolling
    const timer = setTimeout(() => {
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 120);
    return () => clearTimeout(timer);
  }, [location.hash]);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden antialiased">

      {/* ── Same WebGL Topography background as the landing page ────────── */}
      <div className="fixed inset-0 z-0">
        <Topography
          lowColor="#0a0014"
          midColor="#6d28d9"
          highColor="#d8b4fe"
          speed={0.25}
          bands={2.0}
          thickness={0.012}
          glow={0.8}
          pixelSize={1.0}
          mouseInteraction={true}
          mouseRadius={0.3}
          mouseStrength={0.4}
        />
      </div>

      {/* ── Sticky top bar: wordmark + Back button ──────────────────────── */}
      <motion.header
        initial={{ y: -60, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
        className="fixed top-6 left-0 right-0 z-50 flex justify-center px-6"
      >
        <div
          className="bg-[#0A0A0A] border border-[#222222] rounded-full px-8 py-4 flex items-center gap-8 max-w-fit"
          style={{ boxShadow: '0 0 40px rgba(109, 40, 217, 0.15), 0 20px 40px rgba(0,0,0,0.5)' }}
        >
          {/* Wordmark — same as navbar */}
          <span className="text-xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
            WaveSight
          </span>

          {/* Section quick-links — hidden on small screens */}
          <nav className="hidden md:flex items-center gap-5 text-sm font-medium text-gray-400">
            {[
              ['About',    '#about'    ],
              ['Solution', '#solution' ],
              ['Use Cases','#use-cases'],
              ['Privacy',  '#privacy'  ],
            ].map(([label, hash]) => (
              <a
                key={hash}
                href={hash}
                className="hover:text-white transition-colors duration-200 relative group whitespace-nowrap"
              >
                {label}
                <span className="absolute -bottom-1 left-0 w-0 h-[2px] bg-purple-500 group-hover:w-full transition-all duration-300" />
              </a>
            ))}
          </nav>

          {/* Back button — same style as Landing nav Launch button */}
          <motion.button
            onClick={() => navigate('/')}
            className="bg-white text-black px-5 py-2 rounded-full font-semibold text-sm transition-all duration-300 relative overflow-hidden whitespace-nowrap"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="relative z-10">← Back to WaveSight</span>
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-purple-400 to-purple-600"
              initial={{ x: '100%' }}
              whileHover={{ x: 0 }}
              transition={{ duration: 0.3 }}
            />
          </motion.button>
        </div>
      </motion.header>

      {/* ── Page content ────────────────────────────────────────────────── */}
      <div className="relative z-10 pt-32 pb-24">
        <div className="max-w-4xl mx-auto px-6 space-y-32">

          {/* ══════════════════════════════════════════════════════════════
              A — ABOUT
          ══════════════════════════════════════════════════════════════ */}
          <section id="about">
            <RevealSection>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                <span className="text-purple-400 font-mono text-sm tracking-wider">ABOUT</span>
              </div>
              <SectionHeading>About WaveSight</SectionHeading>
              <p className="text-gray-300 text-lg leading-relaxed mb-8">
                WaveSight (Signal Intelligence &amp; Guided Modulation Analysis) is an automated RF signal
                analysis platform designed to turn raw signal recordings into structured, testable signal
                hypotheses.
              </p>
            </RevealSection>

            <RevealSection delay={0.1}>
              <div
                className="bg-[#0A0A0A] border border-[#222222] rounded-2xl p-8 relative overflow-hidden"
                style={{ boxShadow: '0 0 40px rgba(109, 40, 217, 0.08)' }}
              >
                <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/5 rounded-full blur-[80px]" />
                <div className="relative z-10">
                  <h3 className="text-xl font-bold text-purple-400 mb-5">WaveSight is designed to:</h3>
                  <ul className="space-y-4">
                    {[
                      ['Analyze IQ/WAV signal recordings',           'Process raw captured RF data into a form the pipeline can act on.'],
                      ['Extract measurable signal characteristics',  'Derive observable properties such as frequency content, bandwidth and spectral density.'],
                      ['Assist modulation identification',           'Use extracted features to generate and rank candidate modulation hypotheses.'],
                      ['Structure the analysis pipeline',            'Provide a consistent, stage-by-stage workflow that connects ingestion to downstream validation.'],
                      ['Validate candidate interpretations',         'Test hypotheses against downstream signal-processing evidence before accepting them.'],
                    ].map(([title, sub], i) => (
                      <motion.li
                        key={i}
                        initial={{ opacity: 0, x: -16 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.4, delay: 0.05 * i, ease: 'easeOut' }}
                        className="flex items-start gap-4"
                      >
                        <span className="text-teal-400 font-bold text-lg flex-shrink-0 mt-0.5">→</span>
                        <div>
                          <p className="text-white font-semibold leading-snug">{title}</p>
                          <p className="text-gray-400 text-sm leading-relaxed mt-0.5">{sub}</p>
                        </div>
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </div>
            </RevealSection>
          </section>

          {/* ══════════════════════════════════════════════════════════════
              C — PROBLEM & SOLUTION
          ══════════════════════════════════════════════════════════════ */}
          <section id="solution">
            <RevealSection>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
                <span className="text-teal-400 font-mono text-sm tracking-wider">APPROACH</span>
              </div>
              <SectionHeading>From Signal Analysis to Validation</SectionHeading>
              <p className="text-gray-300 text-lg leading-relaxed mb-10">
                WaveSight is designed around a structured, evidence-driven workflow rather than treating a
                predicted modulation classification as the final answer.
              </p>
            </RevealSection>

            {/* Pipeline flow diagram */}
            <RevealSection delay={0.1}>
              <div
                className="bg-[#0A0A0A] border border-[#222222] rounded-2xl p-8 mb-10"
                style={{ boxShadow: '0 0 40px rgba(20, 184, 166, 0.08)' }}
              >
                <div className="relative">
                  {PIPELINE_STEPS.map((step, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.45, delay: 0.08 * i, ease: 'easeOut' }}
                      className="flex items-start gap-4 mb-0"
                    >
                      {/* Vertical connector */}
                      <div className="flex flex-col items-center flex-shrink-0">
                        <div
                          className="w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-bold font-mono"
                          style={{
                            backgroundColor: i === PIPELINE_STEPS.length - 1 ? '#14b8a6' : '#111',
                            borderColor:     i === PIPELINE_STEPS.length - 1 ? '#14b8a6' : '#333',
                            color:           i === PIPELINE_STEPS.length - 1 ? '#000' : '#9ca3af',
                          }}
                        >
                          {i + 1}
                        </div>
                        {i < PIPELINE_STEPS.length - 1 && (
                          <div className="w-[2px] h-10 bg-gradient-to-b from-[#333] to-[#1a1a1a]" />
                        )}
                      </div>

                      <div className="pb-8">
                        <p className="text-white font-bold tracking-wide text-sm">{step.label}</p>
                        <p className="text-gray-500 text-xs mt-0.5">{step.note}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </RevealSection>

            <RevealSection delay={0.15}>
              <div className="bg-[#0A0A0A] border border-teal-900/50 rounded-2xl p-8 relative overflow-hidden"
                style={{ boxShadow: '0 0 40px rgba(20, 184, 166, 0.12)' }}
              >
                <div className="absolute top-0 right-0 w-64 h-64 bg-teal-500/5 rounded-full blur-[80px]" />
                <div className="relative z-10">
                  <div className="w-2 h-8 bg-teal-500 rounded mb-4" />
                  <p className="text-gray-200 text-lg leading-relaxed italic">
                    "WaveSight moves from <span className="text-white font-semibold">'What might this signal be?'</span> toward{' '}
                    <span className="text-teal-400 font-semibold">'Does this hypothesis remain consistent when tested through the signal-processing chain?'</span>"
                  </p>
                </div>
              </div>
            </RevealSection>
          </section>

          {/* ══════════════════════════════════════════════════════════════
              D — USE CASES
          ══════════════════════════════════════════════════════════════ */}
          <section id="use-cases">
            <RevealSection>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                <span className="text-purple-400 font-mono text-sm tracking-wider">USE CASES</span>
              </div>
              <SectionHeading>Where WaveSight Can Be Useful</SectionHeading>
              <p className="text-gray-400 text-base leading-relaxed mb-10">
                The following represent informational use cases that illustrate where a structured
                RF analysis pipeline may be helpful. They are not claims of validated deployment.
              </p>
            </RevealSection>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {[
                {
                  title: 'RF Signal Analysis',
                  body:  'Assist analysts in examining unfamiliar IQ/WAV recordings and extracting meaningful signal characteristics using a consistent, structured pipeline.',
                  accent: 'teal',
                },
                {
                  title: 'Spectrum & Modulation Investigation',
                  body:  'Support structured investigation of candidate signal characteristics and modulation hypotheses, providing observable evidence rather than unexplained predictions.',
                  accent: 'teal',
                },
                {
                  title: 'Research & Education',
                  body:  'Provide a practical environment for understanding signal-processing pipelines, hypothesis generation and downstream validation techniques.',
                  accent: 'purple',
                },
                {
                  title: 'Large-Scale Signal Screening',
                  body:  'Help reduce repetitive manual analysis when multiple recordings need to be examined using a consistent workflow, making the process more systematic and repeatable.',
                  accent: 'purple',
                },
                {
                  title: 'Signal Intelligence Workflows',
                  body:  'Provide structured evidence that can help analysts compare and investigate competing interpretations of RF signals based on measurable processing outcomes.',
                  accent: 'blue',
                },
              ].map((uc, i) => (
                <InfoCard key={i} title={uc.title} body={uc.body} accent={uc.accent as 'teal' | 'purple' | 'blue' | 'red'} delay={0.07 * i} />
              ))}
            </div>
          </section>

          {/* ══════════════════════════════════════════════════════════════
              E — PRIVACY POLICY
          ══════════════════════════════════════════════════════════════ */}
          <section id="privacy">
            <RevealSection>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span className="text-blue-400 font-mono text-sm tracking-wider">LEGAL</span>
              </div>
              <SectionHeading>Privacy Policy</SectionHeading>
              <div
                className="bg-blue-500/10 border border-blue-500/30 rounded-xl px-6 py-4 mb-10"
              >
                <p className="text-blue-300 text-sm leading-relaxed">
                  <strong className="text-blue-200">Prototype notice:</strong> WaveSight is provided as a
                  research demonstration and prototype. Data handling, storage and retention depend on
                  the specific deployment configuration. This policy describes general practices for the
                  default prototype environment.
                </p>
              </div>
            </RevealSection>

            <div className="space-y-6">
              {[
                {
                  title:   'Information We Handle',
                  body:    'When using this platform you may upload RF signal recordings (IQ or WAV format) for analysis. The platform processes these files locally through the configured backend. No personal account information or identifying data is collected as part of normal operation.',
                  accent:  'blue',
                },
                {
                  title:   'Uploaded Signal Data',
                  body:    'Signal files uploaded through the platform are passed to the backend processing pipeline for analysis. Whether uploaded files are stored persistently, cached temporarily or discarded after processing depends on the deployment configuration of the backend. This prototype does not make guarantees about file retention.',
                  accent:  'blue',
                },
                {
                  title:   'How Information Is Used',
                  body:    'Uploaded signal data is used solely for the purpose of performing the signal analysis pipeline: ingestion, DSP characterization, feature extraction, modulation classification and hypothesis validation. Results are returned to the requesting session.',
                  accent:  'blue',
                },
                {
                  title:   'Data Storage & Retention',
                  body:    'This project is provided as a demonstration and prototype environment. The default configuration does not explicitly guarantee permanent deletion or indefinite retention of uploaded files. Users deploying WaveSight in a production environment are responsible for configuring appropriate data retention policies.',
                  accent:  'purple',
                },
                {
                  title:   'Third-Party Services',
                  body:    'The default deployment of this platform does not send signal data to third-party external services. All processing is performed by the local backend. If a specific deployment integrates third-party services, the operator of that deployment is responsible for disclosing those integrations.',
                  accent:  'purple',
                },
                {
                  title:   'Security',
                  body:    'As a prototype application, WaveSight does not implement production-grade authentication, authorization or encryption by default. This platform should not be used to process sensitive or classified signal data unless the deployment has been configured with appropriate security controls.',
                  accent:  'red',
                },
                {
                  title:   'Changes to This Policy',
                  body:    'This policy reflects the prototype state of the project. As the project evolves, this document may be updated. Changes will be reflected in the project repository.',
                  accent:  'blue',
                },
                {
                  title:   'Contact',
                  body:    'For questions about this project or this policy, please refer to the project repository or contact the project maintainer through the repository\'s issue tracker.',
                  accent:  'teal',
                },
              ].map((item, i) => (
                <RevealSection key={i} delay={0.05 * i}>
                  <div className="bg-[#0A0A0A] border border-[#222222] rounded-2xl p-7">
                    <h4 className={`text-lg font-bold mb-3 ${
                      item.accent === 'blue'   ? 'text-blue-400'   :
                      item.accent === 'purple' ? 'text-purple-400' :
                      item.accent === 'red'    ? 'text-red-400'    :
                      'text-teal-400'
                    }`}>
                      {item.title}
                    </h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{item.body}</p>
                  </div>
                </RevealSection>
              ))}
            </div>
          </section>

          {/* ── Footer of the info page ──────────────────────────────── */}
          <RevealSection>
            <div className="border-t border-[#222222] pt-12 flex flex-col sm:flex-row items-center justify-between gap-4">
              <span className="text-xl font-bold tracking-tighter bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
                WaveSight
              </span>
              <p className="text-gray-600 text-xs font-mono text-center sm:text-right">
                © {new Date().getFullYear()} WaveSight Project. Prototype / research demonstration.
              </p>
              <motion.button
                onClick={() => navigate('/')}
                className="text-gray-400 text-sm hover:text-purple-400 transition-colors duration-200 font-mono whitespace-nowrap"
                whileHover={{ x: -3 }}
              >
                ← Back to homepage
              </motion.button>
            </div>
          </RevealSection>

        </div>{/* /max-w-4xl */}
      </div>{/* /relative z-10 */}
    </div>
  );
}
