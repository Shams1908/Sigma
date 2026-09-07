"""Hard and soft BPSK/QPSK symbol demodulators."""

import numpy as np

from .models import DemodulationResult


def _validate_symbols(symbols: np.ndarray) -> np.ndarray:
	values = np.asarray(symbols, dtype=np.complex128)
	if values.ndim != 1 or values.size == 0:
		raise ValueError("symbols must be a non-empty one-dimensional array")
	if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
		raise ValueError("symbols must contain only finite values")
	return values


def _validate_noise_variance(noise_variance: float | None) -> float:
	if noise_variance is None:
		return 1.0
	value = float(noise_variance)
	if not np.isfinite(value) or value <= 0:
		raise ValueError("noise_variance must be a finite positive number")
	return value


class BPSKDemodulator:
	"""Demodulate BPSK symbols using 0 -> +1 and 1 -> -1 mapping."""

	modulation = "BPSK"

	def demodulate(
		self,
		symbols: np.ndarray,
		*,
		noise_variance: float | None = None,
	) -> DemodulationResult:
		values = _validate_symbols(symbols)
		variance = _validate_noise_variance(noise_variance)
		soft_bits = 2.0 * values.real / variance
		bits = (soft_bits < 0).astype(np.uint8)
		return DemodulationResult(
			modulation=self.modulation,
			bits=bits,
			symbols=values,
			soft_bits=soft_bits,
			metrics={"decision_margin": float(np.mean(np.abs(values.real)))},
		)


class QPSKDemodulator:
	"""Demodulate Gray-coded QPSK symbols.

	The mapping is 00 -> (+,+), 01 -> (-,+), 11 -> (-,-), 10 -> (+,-).
	"""

	modulation = "QPSK"

	def demodulate(
		self,
		symbols: np.ndarray,
		*,
		noise_variance: float | None = None,
	) -> DemodulationResult:
		values = _validate_symbols(symbols)
		variance = _validate_noise_variance(noise_variance)
		scale = 2.0 * np.sqrt(2.0) / variance
		# This ordering matches the documented Gray map: the first bit is
		# selected by the imaginary axis and the second by the real axis.
		soft_bits = np.column_stack((scale * values.imag, scale * values.real)).reshape(-1)
		bits = (soft_bits < 0).astype(np.uint8)
		return DemodulationResult(
			modulation=self.modulation,
			bits=bits,
			symbols=values,
			soft_bits=soft_bits,
			metrics={"decision_margin": float(np.mean(np.abs(soft_bits)))},
		)
