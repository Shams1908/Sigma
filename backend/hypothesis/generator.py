"""
Candidate hypothesis generator and bounded search-space expansion for SIGMA.
Expands ML classification probabilities, DSP parameter estimations (M8),
and decoder configuration candidates into bounded, deterministic hypothesis candidate sets.
"""
from dataclasses import dataclass, field
import hashlib
import math
from typing import List, Dict, Optional, Any, Tuple, Union

try:
    from backend.core.config import settings, HypothesisSearchSettings
    from backend.hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceStatus,
        EvidenceComponent,
    )
    from backend.hypothesis.evaluator import (
        create_ml_evidence,
        create_symbol_rate_evidence,
        create_snr_evidence,
    )
except ImportError:
    from core.config import settings, HypothesisSearchSettings
    from hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceStatus,
        EvidenceComponent,
    )
    from hypothesis.evaluator import (
        create_ml_evidence,
        create_symbol_rate_evidence,
        create_snr_evidence,
    )


@dataclass(frozen=True)
class SyncSearchConfig:
    """
    Typed synchronization configuration assumption for candidate search.
    Represents parameters corresponding to capabilities under backend/synchronization/.
    """
    carrier_recovery: str = "costas"        # "costas", "pll", "none"
    timing_recovery: str = "gardner"         # "gardner", "early_late", "mueller_muller", "none"
    matched_filter: str = "rrc"              # "rrc", "rc", "gaussian", "none"
    filter_rolloff: float = 0.35             # Excess bandwidth / alpha for RRC/RC
    loop_bandwidth: float = 0.01             # PLL/timing loop normalized bandwidth
    damping_factor: float = 0.707            # Loop damping factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "carrier_recovery": self.carrier_recovery,
            "timing_recovery": self.timing_recovery,
            "matched_filter": self.matched_filter,
            "filter_rolloff": float(self.filter_rolloff),
            "loop_bandwidth": float(self.loop_bandwidth),
            "damping_factor": float(self.damping_factor),
        }


# Default synchronization preset
DEFAULT_SYNC_CONFIG = SyncSearchConfig(
    carrier_recovery="costas",
    timing_recovery="gardner",
    matched_filter="rrc",
)


@dataclass
class HypothesisSearchConfig:
    """
    Search configuration controlling bounded candidate hypothesis generation.
    """
    # ML search
    top_k_modulations: int = 5
    min_ml_probability: float = 0.01
    include_fallback_modulation: Optional[str] = None
    supported_modulations: Optional[List[str]] = None

    # Symbol-rate search
    symbol_rate_offsets: Optional[List[float]] = None
    symbol_rate_step_count: int = 1
    symbol_rate_uncertainty_multiplier: float = 1.0
    fallback_uncertainty_fraction: float = 0.05
    max_symbol_rate_candidates: int = 3

    # Synchronization search
    sync_configurations: Optional[List[Optional[Union[SyncSearchConfig, Dict[str, Any]]]]] = None

    # Coding, Interleaving & CRC search (identifiers only; defaults to [None])
    fec_candidates: Optional[List[Optional[str]]] = None
    interleaver_candidates: Optional[List[Optional[str]]] = None
    crc_candidates: Optional[List[Optional[str]]] = None

    # Total bounds
    max_candidates: int = 24

    @classmethod
    def from_settings(cls, search_settings: Optional[HypothesisSearchSettings] = None) -> "HypothesisSearchConfig":
        """Constructs search config utilizing system-level settings."""
        cfg = search_settings or getattr(settings, "HYPOTHESIS_SEARCH", None)
        if cfg is None:
            return cls()
        return cls(
            top_k_modulations=cfg.default_top_k_modulations,
            min_ml_probability=cfg.min_ml_probability,
            max_symbol_rate_candidates=cfg.max_symbol_rate_candidates,
            fallback_uncertainty_fraction=cfg.fallback_uncertainty_fraction,
            max_candidates=cfg.max_candidates,
        )


@dataclass(frozen=True)
class HypothesisSearchSummary:
    """
    Lightweight, serializable summary explaining candidate search-space expansion and pruning.
    """
    input_modulation_count: int
    retained_modulation_count: int
    retained_modulations: List[str]
    symbol_rate_candidates: List[float]
    sync_configuration_count: int
    fec_configuration_count: int
    interleaver_configuration_count: int
    total_combinations_before_pruning: int
    candidates_after_deduplication: int
    final_candidate_count: int
    max_candidates_limit: int
    pruned_by_limit: bool
    crc_configuration_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_modulation_count": self.input_modulation_count,
            "retained_modulation_count": self.retained_modulation_count,
            "retained_modulations": list(self.retained_modulations),
            "symbol_rate_candidates": [float(r) for r in self.symbol_rate_candidates],
            "sync_configuration_count": self.sync_configuration_count,
            "fec_configuration_count": self.fec_configuration_count,
            "interleaver_configuration_count": self.interleaver_configuration_count,
            "crc_configuration_count": self.crc_configuration_count,
            "total_combinations_before_pruning": self.total_combinations_before_pruning,
            "candidates_after_deduplication": self.candidates_after_deduplication,
            "final_candidate_count": self.final_candidate_count,
            "max_candidates_limit": self.max_candidates_limit,
            "pruned_by_limit": self.pruned_by_limit,
        }


def generate_deterministic_candidate_id(
    modulation: str,
    symbol_rate: float,
    fec_config: Optional[str] = None,
    interleaver_config: Optional[str] = None,
    sync_assumptions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a deterministic, reproducible candidate ID without random UUIDs.
    Identical search inputs produce identical candidate IDs across runs.
    """
    clean_mod = modulation.replace("-", "").replace("/", "").lower()
    rounded_rate = int(round(float(symbol_rate)))

    norm_fec = (fec_config or "").strip().lower()
    norm_intl = (interleaver_config or "").strip().lower()

    def _freeze_dict(d: Optional[Dict[str, Any]]) -> str:
        if not d:
            return ""
        items = sorted((str(k), str(v)) for k, v in d.items())
        return ";".join(f"{k}={v}" for k, v in items)

    hash_payload = (
        f"{clean_mod}|{rounded_rate}|{norm_fec}|{norm_intl}|{_freeze_dict(sync_assumptions)}"
    )
    digest = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:8]

    tags = []
    if norm_fec and norm_fec != "none":
        tags.append(norm_fec.replace("/", "_"))
    if norm_intl and norm_intl != "none":
        tags.append(norm_intl.replace("/", "_"))

    tag_str = f"_{'_'.join(tags)}" if tags else ""
    return f"hyp_{clean_mod}_{rounded_rate}{tag_str}_{digest}"


def create_candidate(
    modulation: str,
    symbol_rate: float,
    fec_config: Optional[str] = None,
    interleaver_config: Optional[str] = None,
    sync_assumptions: Optional[Dict[str, Any]] = None,
    evidence: Optional[EvidenceTrace] = None,
    candidate_id: Optional[str] = None,
    details: str = "",
) -> HypothesisCandidate:
    """
    Constructs a well-formed HypothesisCandidate instance.
    Generates a deterministic candidate_id if none is provided.
    """
    if candidate_id is None:
        candidate_id = generate_deterministic_candidate_id(
            modulation=modulation,
            symbol_rate=symbol_rate,
            fec_config=fec_config,
            interleaver_config=interleaver_config,
            sync_assumptions=sync_assumptions,
        )

    if evidence is None:
        evidence = EvidenceTrace()

    return HypothesisCandidate(
        id=candidate_id,
        modulation=modulation,
        symbolRate=float(symbol_rate),
        fec_config=fec_config,
        interleaver_config=interleaver_config,
        sync_assumptions=sync_assumptions,
        evidence=evidence,
        details=details,
        status="pending",
    )


def expand_symbol_rate_candidates(
    estimated_baud: float,
    uncertainty_baud: Optional[float] = None,
    config: Optional[HypothesisSearchConfig] = None,
) -> List[float]:
    """
    Expands a center symbol-rate estimate into a deterministic, bounded candidate set.

    Policy:
      1. Always includes the estimated symbol rate when valid (> 0).
      2. If explicit symbol_rate_offsets are configured, evaluates estimated_baud + offset.
      3. Otherwise, generates steps around estimated_baud using uncertainty_baud (or fallback fraction).
      4. Discards non-positive (<= 0) rates.
      5. Deduplicates rates (rounded to 2 decimals).
      6. Prioritizes rates closest to estimated_baud, bounding to max_symbol_rate_candidates.
      7. Returns the final set sorted ascending for clean determinism.

    Args:
        estimated_baud: Center symbol rate estimate in Baud.
        uncertainty_baud: Estimated uncertainty (std dev) in Baud.
        config: HypothesisSearchConfig options.

    Returns:
        List[float]: Bounded, sorted list of candidate symbol rates in Baud.
    """
    est_b = float(estimated_baud)
    if est_b <= 0 or not math.isfinite(est_b):
        raise ValueError(f"Estimated symbol rate must be strictly positive and finite, got {estimated_baud}")

    if config is None:
        config = HypothesisSearchConfig()

    max_cands = max(1, config.max_symbol_rate_candidates)

    # 1. Determine offsets
    if config.symbol_rate_offsets is not None:
        raw_rates = [est_b + float(off) for off in config.symbol_rate_offsets]
        if est_b not in raw_rates:
            raw_rates.insert(0, est_b)
    else:
        # Determine step size delta
        unc_val = float(uncertainty_baud) if uncertainty_baud is not None and uncertainty_baud > 0 else None
        if unc_val is not None:
            delta = unc_val * max(0.01, config.symbol_rate_uncertainty_multiplier)
        else:
            delta = est_b * max(0.001, config.fallback_uncertainty_fraction)

        step_count = max(0, config.symbol_rate_step_count)
        raw_rates = [est_b + (step * delta) for step in range(-step_count, step_count + 1)]

    # 2. Filter positive and finite
    positive_rates = [r for r in raw_rates if r > 0 and math.isfinite(r)]

    # 3. Deduplicate preserving precision
    seen = set()
    deduped = []
    for r in positive_rates:
        rounded = round(r, 2)
        if rounded not in seen:
            seen.add(rounded)
            deduped.append(rounded)

    # 4. Prioritize closest to estimated_baud, then truncate to max_cands
    deduped.sort(key=lambda r: (abs(r - est_b), r))
    retained = deduped[:max_cands]

    # 5. Final deterministic ascending sort
    return sorted(retained)


def generate_hypothesis_search_with_summary(
    ml_probabilities: Dict[str, float],
    symbol_rate: Optional[float] = None,
    symbol_rate_uncertainty: Optional[float] = None,
    parameter_result: Optional[Any] = None,
    search_config: Optional[HypothesisSearchConfig] = None,
    supported_classes: Optional[List[str]] = None,
) -> Tuple[List[HypothesisCandidate], HypothesisSearchSummary]:
    """
    Expands input ML evidence and symbol-rate estimates into a bounded candidate hypothesis set.
    Returns both the candidate list and the traceability summary.

    Pruning and generation pipeline:
      1. Rank & filter modulations by ML probability and thresholds.
      2. Expand symbol rate candidates around estimate (or extract from parameter_result).
      3. Expand sync configurations.
      4. Expand FEC and interleaver configuration identifiers.
      5. Generate prioritized Cartesian combinations.
      6. Deduplicate using canonical identity keys.
      7. Enforce max_candidates bound strictly.

    Args:
        ml_probabilities: Mapping from modulation name to classification probability.
        symbol_rate: Center symbol rate in Baud (required if parameter_result is None).
        symbol_rate_uncertainty: Uncertainty in Baud.
        parameter_result: Optional ParameterAnalysisResult from M8 inference.
        search_config: Optional HypothesisSearchConfig controlling bounds.
        supported_classes: Optional list of supported modulation class names.

    Returns:
        Tuple[List[HypothesisCandidate], HypothesisSearchSummary]
    """
    if search_config is None:
        search_config = HypothesisSearchConfig()

    # Extract symbol rate from M8 parameter_result if provided
    est_rate = symbol_rate
    est_unc = symbol_rate_uncertainty
    m8_metadata: Dict[str, Any] = {}

    if parameter_result is not None:
        sr_attr = getattr(parameter_result, "symbol_rate", None)
        if sr_attr is not None:
            if hasattr(sr_attr, "estimate"):
                est_rate = float(sr_attr.estimate)
            if hasattr(sr_attr, "uncertainty"):
                est_unc = float(sr_attr.uncertainty)
            if hasattr(sr_attr, "to_dict"):
                m8_metadata["symbol_rate_m8"] = sr_attr.to_dict()

        snr_attr = getattr(parameter_result, "snr_db", None)
        if snr_attr is not None and hasattr(snr_attr, "to_dict"):
            m8_metadata["snr_m8"] = snr_attr.to_dict()

        samp_attr = getattr(parameter_result, "sample_rate", None)
        if samp_attr is not None:
            m8_metadata["sample_rate_hz"] = float(samp_attr)

    if est_rate is None:
        raise ValueError("A valid symbol_rate or parameter_result must be provided")

    # Step 1: Filter and sort modulations by probability
    input_mod_count = len(ml_probabilities) if ml_probabilities else 0
    if not ml_probabilities:
        retained_mods: List[Tuple[str, float]] = []
    else:
        # Filter supported classes if configured
        supp_set = None
        if search_config.supported_modulations:
            supp_set = {m.strip().upper() for m in search_config.supported_modulations}
        elif supported_classes:
            supp_set = {m.strip().upper() for m in supported_classes}

        sorted_mods = sorted(ml_probabilities.items(), key=lambda item: item[1], reverse=True)

        retained_mods = []
        for mod, prob in sorted_mods:
            norm_m = mod.strip().upper()
            if supp_set is not None and norm_m not in supp_set:
                continue
            if prob < search_config.min_ml_probability:
                continue
            retained_mods.append((mod, float(prob)))
            if len(retained_mods) >= search_config.top_k_modulations:
                break

        # Check fallback modulation if all filtered out
        if not retained_mods and search_config.include_fallback_modulation:
            fb = search_config.include_fallback_modulation
            fb_prob = float(ml_probabilities.get(fb, 0.0))
            retained_mods.append((fb, fb_prob))

    # Step 2: Expand symbol rates
    sym_rates = expand_symbol_rate_candidates(
        estimated_baud=est_rate,
        uncertainty_baud=est_unc,
        config=search_config,
    )

    # Step 3: Synchronization configurations
    sync_configs: List[Optional[Dict[str, Any]]] = []
    if search_config.sync_configurations is not None:
        for s in search_config.sync_configurations:
            if s is None:
                sync_configs.append(None)
            elif isinstance(s, SyncSearchConfig):
                sync_configs.append(s.to_dict())
            elif isinstance(s, dict):
                sync_configs.append(dict(s))
    else:
        # Default: single baseline sync configuration
        sync_configs.append(DEFAULT_SYNC_CONFIG.to_dict())

    # Step 4: FEC, Interleaver, and CRC identifiers
    fec_list = search_config.fec_candidates if search_config.fec_candidates else [None]
    intl_list = search_config.interleaver_candidates if search_config.interleaver_candidates else [None]
    crc_list = search_config.crc_candidates if search_config.crc_candidates else [None]

    # Compute pre-pruning combination count
    total_combinations = (
        len(retained_mods)
        * len(sym_rates)
        * len(sync_configs)
        * len(fec_list)
        * len(intl_list)
        * len(crc_list)
    )

    # Step 5 & 6: Cartesian combination generation with prioritized ordering & deduplication
    candidates: List[HypothesisCandidate] = []
    seen_keys = set()
    total_unique_count = 0

    for mod, prob in retained_mods:
        ml_comp = create_ml_evidence(
            candidate_modulation=mod,
            ml_probabilities=ml_probabilities,
            supported_classes=supported_classes,
        )

        # Sort rates by closeness to center estimate for prioritized addition
        for rate in sorted(sym_rates, key=lambda r: abs(r - est_rate)):
            # Symbol-rate evidence: if M8 parameter_result was provided, attach Gaussian agreement
            if parameter_result is not None and est_rate > 0:
                sym_comp = create_symbol_rate_evidence(
                    candidate_baud=rate,
                    estimated_baud=est_rate,
                    uncertainty_baud=est_unc or 0.0,
                    details={"m8_symbol_rate": m8_metadata.get("symbol_rate_m8")},
                )
            else:
                sym_comp = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

            # SNR evidence: if M8 parameter_result was provided, preserve telemetry as NOT_EVALUATED
            if parameter_result is not None and "snr_m8" in m8_metadata:
                snr_m8 = m8_metadata["snr_m8"]
                est_snr = snr_m8.get("estimate") if isinstance(snr_m8, dict) else None
                unc_snr = snr_m8.get("uncertainty") if isinstance(snr_m8, dict) else None
                snr_comp = create_snr_evidence(
                    estimated_snr_db=est_snr,
                    uncertainty_db=unc_snr,
                    status=EvidenceStatus.NOT_EVALUATED,
                    details={"m8_snr": snr_m8},
                )
            else:
                snr_comp = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

            for sync_cfg in sync_configs:
                # Merge M8 metadata into sync assumptions if present
                base_sync: Optional[Dict[str, Any]] = None
                if sync_cfg is not None or m8_metadata:
                    base_sync = dict(sync_cfg) if sync_cfg is not None else {}
                    if m8_metadata:
                        base_sync.update(m8_metadata)
                    base_sync["estimated_symbol_rate_baud"] = float(est_rate)
                    if est_unc is not None:
                        base_sync["symbol_rate_uncertainty_baud"] = float(est_unc)

                for fec in fec_list:
                    for intl in intl_list:
                        for crc in crc_list:
                            active_sync = dict(base_sync) if base_sync is not None else {}
                            if crc is not None and str(crc).lower() != "none":
                                active_sync["crc_scheme"] = str(crc)

                            # Construct evidence container
                            evidence = EvidenceTrace(
                                ml=ml_comp,
                                symbol_rate=sym_comp,
                                snr=snr_comp,
                                constellation=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                                timing=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                                fec=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                                bitstream=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                                interleaver=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                            )

                            # Construct details provenance
                            details_parts = [f"Hypothesis: {mod} at {rate:g} Baud (ML prob: {prob:.4f})"]
                            if fec and str(fec).lower() != "none":
                                details_parts.append(f"FEC: {fec}")
                            if intl and str(intl).lower() != "none":
                                details_parts.append(f"Interleaver: {intl}")
                            if crc and str(crc).lower() != "none":
                                details_parts.append(f"CRC: {crc}")
                            if active_sync and active_sync.get("carrier_recovery"):
                                details_parts.append(
                                    f"Sync: {active_sync['carrier_recovery']}/{active_sync.get('timing_recovery', '')}"
                                )
                            details_str = "; ".join(details_parts)

                            cand = create_candidate(
                                modulation=mod,
                                symbol_rate=rate,
                                fec_config=fec,
                                interleaver_config=intl,
                                sync_assumptions=active_sync if active_sync else None,
                                evidence=evidence,
                                details=details_str,
                            )

                            key = cand.identity_key()
                            if key not in seen_keys:
                                seen_keys.add(key)
                                total_unique_count += 1
                                if len(candidates) < search_config.max_candidates:
                                    candidates.append(cand)

    pruned_by_limit = total_unique_count > len(candidates)

    summary = HypothesisSearchSummary(
        input_modulation_count=input_mod_count,
        retained_modulation_count=len(retained_mods),
        retained_modulations=[m for m, _ in retained_mods],
        symbol_rate_candidates=sym_rates,
        sync_configuration_count=len(sync_configs),
        fec_configuration_count=len(fec_list),
        interleaver_configuration_count=len(intl_list),
        crc_configuration_count=len(crc_list),
        total_combinations_before_pruning=total_combinations,
        candidates_after_deduplication=total_unique_count,
        final_candidate_count=len(candidates),
        max_candidates_limit=search_config.max_candidates,
        pruned_by_limit=pruned_by_limit,
    )

    return candidates, summary


def generate_hypothesis_search(
    ml_probabilities: Dict[str, float],
    symbol_rate: Optional[float] = None,
    symbol_rate_uncertainty: Optional[float] = None,
    parameter_result: Optional[Any] = None,
    search_config: Optional[HypothesisSearchConfig] = None,
    supported_classes: Optional[List[str]] = None,
) -> List[HypothesisCandidate]:
    """
    Generates a bounded, deterministic list of hypothesis candidates from ML probabilities,
    symbol-rate estimates, and decoder configuration space.
    """
    candidates, _ = generate_hypothesis_search_with_summary(
        ml_probabilities=ml_probabilities,
        symbol_rate=symbol_rate,
        symbol_rate_uncertainty=symbol_rate_uncertainty,
        parameter_result=parameter_result,
        search_config=search_config,
        supported_classes=supported_classes,
    )
    return candidates


def generate_hypotheses_from_ml(
    ml_probabilities: Dict[str, float],
    symbol_rate: float,
    supported_classes: Optional[List[str]] = None,
    fec_candidates: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[HypothesisCandidate]:
    """
    Backward-compatible candidate generator from ML classification probabilities and
    single estimated symbol rate.
    """
    config = HypothesisSearchConfig(
        top_k_modulations=top_k,
        max_symbol_rate_candidates=1,
        symbol_rate_offsets=[0.0],
        fec_candidates=fec_candidates if fec_candidates else [None],
        sync_configurations=[None],  # Preserves None sync_assumptions for legacy compatibility
        max_candidates=top_k * (len(fec_candidates) if fec_candidates else 1),
    )
    return generate_hypothesis_search(
        ml_probabilities=ml_probabilities,
        symbol_rate=symbol_rate,
        search_config=config,
        supported_classes=supported_classes,
    )
