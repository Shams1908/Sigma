"""
QAM demodulator — 16-QAM stub.

OUT OF SCOPE for the current implementation phase.
16-QAM demodulation is explicitly deferred until BPSK and QPSK are
fully exercised in production and testing.

When this is implemented, it should:
  1. Accept synchronised complex symbols (post matched-filter + carrier recovery).
  2. Apply a 16-QAM Gray-coded decision map (4×4 grid, unit average power).
  3. Return a DemodResult consistent with psk_demod.DemodResult.
  4. Be wired into demodulate() in __init__.py alongside BPSK/QPSK.

Do not call this module from the hypothesis evaluator until it is implemented.
"""
from __future__ import annotations


class QAMDemodNotImplemented(NotImplementedError):
    """Raised when 16-QAM demodulation is requested before implementation."""


def demod_16qam(_symbols):  # noqa: ANN001
    """Placeholder — 16-QAM demodulation is not yet implemented."""
    raise QAMDemodNotImplemented(
        "16-QAM demodulation is out of scope for this phase. "
        "See module docstring for implementation notes."
    )
