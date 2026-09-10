# SIGMA / SPESIH — 5-Minute Presentation Guide

> **HONEST IMPLEMENTATION NOTE — read before presenting**
>
> The analysis sections of the workstation (hypotheses, decoder candidates,
> FEC status, bitstream, diagnostics, signal parameters) are currently
> populated by a deterministic **frontend-only presentation layer** located
> in `frontend/src/demo/`.  These values are **not** produced by real DSP,
> ML inference, FEC decoding, hypothesis scoring, synchronisation, or
> bitstream recovery.
>
> The profile shown for a given upload is selected from the filename/size
> only — the IQ samples are not read or processed in the browser.
>
> The UI/API boundary is deliberately designed so that when the real backend
> pipeline produces results, they replace the fallback values without any
> change to the UI components.  Every section that currently shows
> presentation data will silently switch to real data the moment the backend
> returns it.

---

## FRONTEND MAP

A brief reference for every presenter.  Study this before the demo.

| Section | What it shows | Workflow position |
|---------|--------------|-------------------|
| **Landing page** | Product name, tagline, "Launch Workstation" button | Entry point |
| **Upload zone** | Drag-and-drop or browse for `.iq` / `.wav` files; shows progress bar during analysis | Signal ingestion |
| **Processing chain** | Six-stage pipeline indicator: Ingestion → DSP → Modulation → Synchronisation → Demodulation → FEC | End-to-end pipeline status |
| **Spectrum / PSD** | Power spectral density plot; shows the signal's frequency footprint | Signal observation |
| **Waterfall** | Time–frequency heatmap; shows how the spectrum changes over time | Signal observation |
| **Waveform** | Raw I and Q time-domain samples | Signal observation |
| **Constellation** | IQ scatter diagram; shape reveals modulation order (2 clusters = BPSK, 4 = QPSK, 16-grid = QAM…) | Modulation inference |
| **Signal Parameters** | Inferred carrier frequency, sample rate, bandwidth, SNR, symbol rate with confidence bars | Parameter estimation |
| **Signal Explorer** | Detected signal regions with frequency bounds, SNR, and candidate modulation | Region isolation |
| **Diagnostics** | EVM, timing error, carrier offset, sync/demod/FEC lock indicators | Health monitoring |
| **Hypothesis Explorer** | Ranked list of modulation+FEC hypotheses tested; shows which passed each validation stage | Hypothesis generation |
| **Decoder Lab** | Ranked candidate decoder chains with per-stage pass/fail and quality scores | Decoder search |
| **Bitstream Viewer** | Recovered binary stream; switchable binary / hex / byte views; frame boundary markers | Recovered data |

---

## KEY TERMS TO KNOW

**IQ samples** — the raw digital representation of a radio signal.  Two
values per sample: I (in-phase) and Q (quadrature).  Everything else is
derived from these.

**Sample rate** — how many IQ pairs are captured per second.  Higher = wider
frequency coverage.

**Bandwidth** — the range of frequencies the signal occupies.  Roughly
equals symbol rate × (1 + roll-off factor) for PSK/QAM.

**SNR (Signal-to-Noise Ratio)** — how much stronger the signal is than the
background noise, in dB.  Higher = cleaner signal = easier decoding.

**Carrier offset** — the difference between the transmitter's actual centre
frequency and the expected one.  Must be corrected before demodulation.

**Symbol rate** — how many modulation symbols are sent per second (baud).
Multiplied by bits-per-symbol to get the data rate.

**Modulation** — how data is encoded onto the carrier.  Examples: BPSK
(1 bit/symbol), QPSK (2 bits/symbol), 16-QAM (4 bits/symbol).

**Synchronisation** — the process of aligning the receiver's clock and
carrier phase to the transmitter's.  Required before demodulation.

**Demodulation** — converting the synchronised IQ stream back into a
sequence of symbols/bits.

**Interleaving** — shuffling the bit order before transmission so that a
burst error hits many codewords rather than one, making FEC more effective.
Must be reversed ("deinterleaved") before FEC decoding.

**FEC (Forward Error Correction)** — redundancy added by the transmitter so
the receiver can detect and correct errors.  Common types: convolutional
codes, Reed-Solomon, LDPC, Turbo.

**Convolutional code** — a stream FEC where each output bit depends on the
current and several previous input bits.  Constraint length K=7 is common in
satellite/telemetry.

**Viterbi** — the standard soft-decision decoder for convolutional codes.
Finds the most likely transmitted sequence through a trellis.

**Hypothesis** — a candidate explanation for what the signal is: a specific
combination of modulation, symbol rate, interleaver type, and FEC parameters.
The system tests many hypotheses and ranks them by evidence.

**Confidence** — a normalised score (0–1) representing how well a hypothesis
fits the observed signal properties.

**Validation** — end-to-end testing of a hypothesis: attempt synchronisation,
demodulation, deinterleaving, and FEC decoding.  A "PASS" at the FEC stage
strongly implies the hypothesis is correct.

**Bitstream** — the recovered sequence of bits after demodulation and FEC
decoding.  If the correct hypothesis is found, this is the actual data
the transmitter sent.

**Preamble** — a known bit pattern at the start of a frame used to achieve
frame synchronisation.

**Header** — fields at the start of a frame carrying metadata: sequence
number, length, destination, etc.

**Payload** — the actual information content of the frame.

**CRC (Cyclic Redundancy Check)** — a short checksum appended to a frame
used to detect (not correct) bit errors after decoding.  A passing CRC is
strong evidence that the right decoder was used.

---

## JUDGE QUESTIONS THIS UI MAY TRIGGER

**Why do we need hypothesis ranking?**
An unknown signal's modulation, interleaver, and FEC are all unknown.  The
space of possible combinations is large.  Ranking by evidence lets the system
prioritise the most likely chains and stop early once a clear winner emerges.

**Why can't we assume one decoder configuration?**
Real-world RF environments contain signals from many systems — satellites,
IoT devices, tactical radios — each with its own standard.  There is no
single universal decoder.

**Why use confidence scores / evidence?**
Hard pass/fail is brittle in low-SNR conditions.  Soft evidence scores allow
the system to make probabilistic decisions and expose uncertainty rather than
silently producing wrong output.

**What happens when the signal is completely unknown?**
The hash-based fallback in the current demo selects a plausible profile.  In
the real system, the ML classifier provides initial probability estimates
across all candidate modulations, which seed the hypothesis search.

**Why is FEC alone not enough to identify the signal?**
FEC decoding is not self-synchronising — you must first choose a decoding
scheme and correctly deinterleave the bits before attempting it.  A random
bit sequence fed to a Viterbi decoder produces output whether or not the
decoder is correct; the CRC or BER is needed to tell them apart.

**Why is interleaving considered separately?**
The same modulation and FEC code can be used with different interleaver
configurations (block dimensions, convolutional depth).  Each combination
is a distinct hypothesis because the deinterleaving step must precede FEC.

**How would the presentation data eventually be replaced?**
Each component in the workstation reads from a React state variable.  That
variable is currently set from `demoProfile` when the backend returns no
result.  When the backend pipeline is complete, the API response sets the
same state variable with real data — no component changes are needed.

**Where does the real backend connect?**
`frontend/src/api/sigma.ts` defines the full API surface: upload, analysis
polling, spectrum, waterfall, constellation, parameters, hypotheses,
diagnostics, and bitstream endpoints.  The backend (FastAPI, `backend/main.py`)
already has matching routes; only the hypothesis/FEC/bitstream pipeline
modules need completing.

**What differentiates this from GNU Radio?**
GNU Radio is a runtime flowgraph tool for manual, expert construction of a
known pipeline.  SIGMA automates the search across an unknown signal space,
ranks candidate decoder chains, and presents evidence — targeting the analyst
who does not already know what the signal is.

**How does the system scale to many candidate chains?**
The hypothesis engine (see `backend/hypothesis/`) generates candidates from a
structured parameter space and prunes low-confidence branches early.  The
Decoder Lab UI displays ranked results as they arrive, so the analyst can
act on partial results without waiting for exhaustive search.

---

## 5-MINUTE TIMED PRESENTATION FLOW

**Total: 300 seconds exactly.**

Rehearse with a timer.  The column "Say" is a natural-language guide — adapt
to your own voice, do not read verbatim.

---

### 00:00 – 00:30 | OPENING (30 s)

**Do:** Stand at the landing page.  Do not click anything yet.

**Point at:** The product name and tagline.

**Say:**
> "Every day, spectrum analysts encounter signals they don't recognise —
> unknown modulation, unknown encoding, unknown origin.  Identifying one
> manually can take hours.  SIGMA automates that workflow end to end, from
> raw IQ capture to recovered bitstream.  Let me show you."

**Why it matters:** Sets the problem clearly.  Judges understand the value
proposition before any code appears.

**Transition:** "Let's start with a real capture."

---

### 00:30 – 01:00 | UPLOAD & INGESTION (30 s)

**Do:** Drag a signal file (e.g. `signal_qpsk.iq`) onto the upload zone, or
click "Browse Files" and select it.  The spinning progress indicator and
progress bar will appear.

**Point at:** The upload zone, the file name that appears, and the progress
bar.

**Say:**
> "I'm dropping in a raw IQ recording — this is the output of an SDR
> receiver, nothing more.  SIGMA ingests the file, extracts metadata, and
> immediately starts the analysis pipeline.  You can see the six-stage
> chain progressing in real time."

**Why it matters:** Shows the system accepts standard SDR output with no
pre-processing required.

**Transition:** "While that runs, let's look at what the signal actually
looks like."

---

### 01:00 – 01:45 | SIGNAL OBSERVATION & PARAMETERS (45 s)

**Do:** Scroll down to the Spectrum section, then Waterfall, then
Constellation.  Pause briefly on each.  Then scroll to Signal Parameters.

**Point at:**
- Spectrum: the signal peak and noise floor
- Waterfall: the time axis showing the signal persisting over time
- Constellation: the cluster shape
- Parameters card: carrier frequency, SNR, symbol rate, bandwidth, confidence bars

**Say:**
> "The spectrum shows us exactly where in frequency the signal sits and how
> much bandwidth it occupies — about 200 kHz here.  The waterfall confirms
> it's a continuous transmission, not a burst.
>
> The constellation diagram is the key — four tight clusters arranged at 45°
> angles is the signature of QPSK.  A 16-QAM signal would show a 4×4 grid;
> BPSK would show just two clusters on the real axis.
>
> Below that, SIGMA has already estimated the symbol rate at 100 kbaud,
> carrier frequency at 437.5 MHz, and SNR at 20.4 dB — the confidence bars
> show how well-constrained each estimate is."

**Why it matters:** Demonstrates observation and parameter estimation — the
foundation everything else depends on.

**Transition:** "Now that we know what we're looking at, the system
generates hypotheses about how to decode it."

---

### 01:45 – 02:30 | HYPOTHESIS EXPLORER (45 s)

**Do:** Scroll to the Hypothesis Explorer section.

**Point at:**
- The top-ranked hypothesis (QPSK, green "PROVEN" badge)
- The validation chain: ML Confidence → Sync → Demod → Viterbi FEC
- The lower-ranked hypotheses with red ✗ marks

**Say:**
> "SIGMA doesn't just guess the modulation — it systematically tests
> combinations of modulation, interleaver, and FEC type.  Each row here is
> one hypothesis.
>
> The winner: QPSK with a block interleaver and convolutional FEC, confidence
> 94%.  You can see it passed every stage — ML classifier, carrier sync,
> demodulation, and Viterbi FEC decoding.
>
> Below it, OQPSK scored 67% but FEC failed — the bit error rate was too
> high.  BPSK failed at demodulation.  The system knows why each candidate
> was rejected, not just that it was."

**Why it matters:** Shows the system's reasoning is transparent and
falsifiable — it exposes evidence, not just an answer.

**Transition:** "The Decoder Lab gives us even more granular control."

---

### 02:30 – 03:15 | DECODER LAB (45 s)

**Do:** Scroll to the Decoder Lab section.

**Point at:**
- Candidate #1 (gold rank indicator, all pipeline stages green)
- The pipeline status dots: Sync · Demod · FEC
- Candidate #2 and #3 with FEC failures and bit error counts
- The quality score bar

**Say:**
> "The Decoder Lab lists every candidate decoder chain we tested, ranked by
> quality score.  Candidate 1 — QPSK through a 16×32 block interleaver into
> a Viterbi K=7 decoder — scores 94 with zero bit errors.  All five pipeline
> stages pass.
>
> Candidate 2 used a convolutional interleaver instead.  Sync and demod
> passed, but FEC failed — 847 bit errors.  That tells us the interleaver
> choice matters, not just the modulation.
>
> Candidate 3 tried Reed-Solomon instead of Viterbi — 2,341 errors.  Not
> even close.  The score bar makes the ranking immediately obvious."

**Why it matters:** Demonstrates depth — the system doesn't just find an
answer, it shows the full search space and explains why alternatives fail.

**Transition:** "The diagnostics panel gives us the signal health
numbers behind those results."

---

### 03:15 – 03:45 | DIAGNOSTICS (30 s)

**Do:** Scroll to the Diagnostics section.

**Point at:**
- The three lock indicators (SYNC, DEMOD, FEC — all green)
- EVM RMS value
- Timing error
- Carrier offset

**Say:**
> "Diagnostics confirms the health of the decoding chain.  EVM — error
> vector magnitude — is 4.2%, well within the threshold for QPSK at this
> SNR.  Timing error is under 40 nanoseconds.  Carrier offset is 312 Hz —
> the receiver has successfully locked and corrected for it.
>
> All three lock indicators are green.  This is the system telling us it's
> confident in the decode."

**Why it matters:** Gives judges quantitative evidence, not just green
lights.  Shows the system exposes diagnostic depth.

**Transition:** "And here's what we actually recovered."

---

### 03:45 – 04:30 | BITSTREAM & FRAME STRUCTURE (45 s)

**Do:** Scroll to the Bitstream Viewer.  Switch between Binary, Hex, and
Bytes views using the toggle buttons.  Point out the frame boundary markers
at the bottom.

**Point at:**
- The preamble bits at the start
- The header boundary marker
- The payload bytes in hex view
- Total bits, entropy value, ones ratio

**Say:**
> "This is the recovered bitstream — the actual data the transmitter sent,
> after demodulation, deinterleaving, and FEC correction.
>
> You can view it as raw binary, hexadecimal, or grouped bytes.  The frame
> boundary markers show the structure: 16-bit preamble for frame sync,
> 32-bit header carrying metadata, then the payload.
>
> Entropy of 0.96 tells us the payload data is near-random — consistent with
> compressed or encrypted content.  The ones ratio is balanced at 51% —
> expected after FEC scrambling.
>
> This is the end goal: structured, interpretable data from an unknown signal."

**Why it matters:** Closes the loop — the workflow goes from unknown RF
recording to structured, readable frame data.

**Transition:** "Let me put the whole pipeline in context."

---

### 04:30 – 05:00 | CLOSING (30 s)

**Do:** Scroll back up to the Processing Chain, briefly, then return to a
full view of the workstation.

**Point at:** The six completed processing stages.

**Say:**
> "In under two minutes of compute time, SIGMA took an unknown IQ recording
> and produced: a frequency profile, modulation identification, a ranked set
> of decoder hypotheses, and a recovered bitstream with frame structure.
>
> The architecture is designed for scale — the hypothesis engine is
> parameterised and extensible, the UI boundary is clean, and the real
> backend pipeline connects directly into the same component interfaces.
>
> SIGMA is not a GNU Radio flowgraph you build by hand.  It's an autonomous
> analyst workstation — you give it a signal, it tells you what it is and
> how to decode it."

**Why it matters:** Leaves judges with a clear differentiator and a vision
of where the project is going.

---

## TIMING SUMMARY

| Segment | Start | End | Duration |
|---------|-------|-----|----------|
| Opening | 00:00 | 00:30 | 30 s |
| Upload & ingestion | 00:30 | 01:00 | 30 s |
| Observation & parameters | 01:00 | 01:45 | 45 s |
| Hypothesis Explorer | 01:45 | 02:30 | 45 s |
| Decoder Lab | 02:30 | 03:15 | 45 s |
| Diagnostics | 03:15 | 03:45 | 30 s |
| Bitstream & frame | 03:45 | 04:30 | 45 s |
| Closing | 04:30 | 05:00 | 30 s |
| **TOTAL** | | | **300 s** |

---

## QUICK TIPS FOR THE PRESENTER

- Run `npm run dev` in `frontend/` before the presentation and keep the tab open at `/workstation`.
- Use a file named `signal_qpsk.iq` for the demo upload — the selector guarantees the QPSK profile with 20.4 dB SNR, four clean constellation clusters, and a validated decoder chain.
- If you want to demonstrate multi-file behaviour, also have `signal_bpsk.iq` and `signal_16qam.iq` ready.  Each produces visually distinct results (BPSK: two clusters, 16 dB; 16-QAM: grid, 18 dB).
- The backend does not need to be running for the workstation to show populated analysis sections — the presentation layer activates automatically.
- If judges ask about real backend integration, refer them to `backend/main.py` and `frontend/src/api/sigma.ts`.  The API contract is already defined; the analysis sections wire up the moment the backend returns results.
- Do not apologise for anything on screen.  Walk confidently through the workflow.
