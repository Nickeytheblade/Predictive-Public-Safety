"""
consensus_oracle.py

Reference implementation of a security-focused consensus oracle simulator.
Simulates commit-reveal, aggregation, outlier detection, and reputation updates
for multiple operators predicting rare events (e.g., crime probabilities).
"""

import hashlib
import hmac
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -------------------------- Utilities -------------------------- #

def sha256_hex(s: str) -> str:
    """Return SHA-256 hash of a string as hex."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hmac_sign(key: bytes, message: str) -> str:
    """Return HMAC-SHA256 signature."""
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def hmac_verify(key: bytes, message: str, tag: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    return hmac.compare_digest(hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest(), tag)


def z_score_standardize(value: float, mu: float, sigma: float) -> float:
    """Return standardized z-score."""
    if sigma <= 0:
        return 0.0
    return (value - mu) / sigma


# -------------------------- Operator -------------------------- #

@dataclass
class Operator:
    """Represents a single oracle operator node."""
    id: int
    name: str
    secret_key: bytes
    weight: float = 1.0
    bias: float = 0.0
    noise_scale: float = 0.0
    malicious: bool = False

    def produce_forecast(self, covariates: Dict[str, float]) -> float:
        """Generate probabilistic forecast given covariates."""
        coeffs = {"lpr_z": 1.2, "pc_z": 0.9, "intercept": -7.0}
        lin = coeffs["intercept"] + coeffs["lpr_z"] * covariates["lpr_z"] + coeffs["pc_z"] * covariates["pc_z"]
        p_true = 1.0 / (1.0 + math.exp(-lin))
        p = p_true + self.bias + random.gauss(0, self.noise_scale)
        if self.malicious and random.random() < 0.12:
            p = min(max(p * random.uniform(50, 200), 0.0), 1.0)
        return max(0.0, min(1.0, p))


# -------------------------- Consensus Simulator -------------------------- #

class ConsensusOracleSimulator:
    """
    Simulator for multiple oracle operators.
    Includes commit-reveal, aggregation, outlier detection, and reputation update.
    """

    def __init__(
        self,
        operators: List[Operator],
        eta: float = 0.5,
        z_thresh: float = 3.0,
        require_k: Optional[int] = None,
    ):
        self.operators = {op.id: op for op in operators}
        self.N = len(operators)
        self.eta = eta
        self.z_thresh = z_thresh
        self.require_k = require_k or max(3, math.ceil(2 * self.N / 3))
        self.commits: Dict[int, Dict[int, dict]] = {}
        self.reveals: Dict[int, Dict[int, dict]] = {}
        self.round_logs: List[dict] = []
        self.op_history: dict = defaultdict(list)
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(op.weight for op in self.operators.values())
        if total == 0:
            for op in self.operators.values():
                op.weight = 1.0 / self.N
        else:
            for op in self.operators.values():
                op.weight = op.weight / total

    # -------------------- Commit / Reveal -------------------- #

    def commit_phase(self, round_id: int, cov_snapshot: dict):
        """Operators produce commits."""
        M_v = "model_v1.0_hash_example"
        timestamp = int(time.time())
        self.commits[round_id] = {}
        for op in self.operators.values():
            raw_input = json.dumps({"cov": cov_snapshot, "op_id": op.id}, sort_keys=True)
            D_i = sha256_hex(raw_input)
            p_i = op.produce_forecast({"lpr_z": cov_snapshot["lpr_z"], "pc_z": cov_snapshot["pc_z"]})
            meta = json.dumps({"round": round_id, "cell": cov_snapshot.get("cell", "c0")})
            H_i = sha256_hex(json.dumps({"p": p_i, "meta": meta}))
            C_i = sha256_hex("||".join([D_i, M_v, H_i, str(timestamp)]))
            signature = hmac_sign(op.secret_key, C_i)
            self.commits[round_id][op.id] = {
                "D_i": D_i,
                "M_v": M_v,
                "H_i": H_i,
                "C_i": C_i,
                "timestamp": timestamp,
                "signature": signature,
                "p_i_local": p_i,
                "raw_input": raw_input,
            }

    def reveal_phase(self, round_id: int):
        """Operators reveal their forecasts and verify commits."""
        self.reveals[round_id] = {}
        for op in self.operators.values():
            commit = self.commits[round_id][op.id]
            D_i, M_v, p_i, H_i = commit["D_i"], commit["M_v"], commit["p_i_local"], commit["H_i"]
            C_i_expected = sha256_hex("||".join([D_i, M_v, H_i, str(commit["timestamp"])]))
            commit_ok = (commit["C_i"] == C_i_expected)
            sig_ok = hmac_verify(op.secret_key, commit["C_i"], commit["signature"])
            reveal_sig = hmac_sign(op.secret_key, json.dumps({"D_i": D_i, "M_v": M_v, "p_i": p_i}))
            reveal_sig_ok = hmac_verify(op.secret_key, json.dumps({"D_i": D_i, "M_v": M_v, "p_i": p_i}), reveal_sig)
            verified = commit_ok and sig_ok and reveal_sig_ok
            self.reveals[round_id][op.id] = {
                "D_i": D_i,
                "M_v": M_v,
                "p_i": p_i,
                "commit_ok": commit_ok,
                "sig_ok": sig_ok,
                "reveal_sig_ok": reveal_sig_ok,
                "verified": verified,
                "C_i": commit["C_i"],
            }

    # -------------------- Aggregation / Consensus -------------------- #

    def aggregate_and_consensus(self, round_id: int, tau: float = 0.01, delta: float = 1.0, eps: float = 1e-9) -> dict:
        """Aggregate verified reveals and determine consensus."""
        reveals = self.reveals[round_id]
        V = [op_id for op_id, r in reveals.items() if r["verified"]]
        if len(V) < self.require_k:
            record = {"round": round_id, "status": "insufficient_reveals", "num_reveals": len(V)}
            self.round_logs.append(record)
            return record
        p_list = [reveals[op_id]["p_i"] for op_id in V]
        w_arr = np.array([self.operators[op].weight for op in V])
        p_arr = np.array(p_list)
        p_bar = float(np.sum(w_arr * p_arr) / np.sum(w_arr))
        sigma2 = float(np.sum(w_arr * (p_arr - p_bar) ** 2) / np.sum(w_arr))
        sigma = math.sqrt(max(sigma2, 0.0))
        # Outlier detection
        outliers = [op_id for idx, op_id in enumerate(V) if abs(p_arr[idx] - p_bar) / (math.sqrt(sigma2 + eps)) > self.z_thresh]
        V_filtered = [op for op in V if op not in outliers]
        if not V_filtered:
            record = {"round": round_id, "status": "all_outliers", "num_reveals": len(V), "num_outliers": len(outliers)}
            self.round_logs.append(record)
            return record
        p_arr_f = np.array([reveals[op]["p_i"] for op in V_filtered])
        w_arr_f = np.array([self.operators[op].weight for op in V_filtered])
        p_bar_f = float(np.sum(w_arr_f * p_arr_f) / np.sum(w_arr_f))
        sigma2_f = float(np.sum(w_arr_f * (p_arr_f - p_bar_f) ** 2) / np.sum(w_arr_f))
        sigma_f = math.sqrt(max(sigma2_f, 0.0))
        rel_dispersion = sigma_f / (p_bar_f + eps)
        accepted = (p_bar_f >= tau) and (rel_dispersion <= delta) and (len(V_filtered) >= self.require_k)
        record = {
            "round": round_id,
            "status": "accepted" if accepted else "rejected",
            "p_bar_raw": p_bar,
            "sigma_raw": sigma,
            "p_bar_filtered": p_bar_f,
            "sigma_filtered": sigma_f,
            "rel_dispersion": rel_dispersion,
            "num_reveals": len(V),
            "num_outliers": len(outliers),
            "outliers": outliers,
            "verified_ops": V_filtered,
            "all_verified_ops": V,
        }
        self.round_logs.append(record)
        return record

    # -------------------- Reputation Update -------------------- #

    def update_weights(self, round_id: int, y_actual: int) -> dict:
        """Update operator weights using Brier score loss."""
        reveals = self.reveals.get(round_id, {})
        for op_id, r in reveals.items():
            if not r["verified"]:
                continue
            loss = (r["p_i"] - y_actual) ** 2
            op = self.operators[op_id]
            op.weight *= math.exp(-self.eta * loss)
            self.op_history[op_id].append({"round": round_id, "p": r["p_i"], "loss": loss, "y": y_actual})
        self._normalize_weights()
        return {op_id: round(op.weight, 6) for op_id, op in self.operators.items()}

    # -------------------- Full Simulation Round -------------------- #

    def simulate_round(self, round_id: int, cov_snapshot: dict, tau: float = 0.01, delta: float = 1.0) -> dict:
        """Run commit, reveal, aggregation, and reputation update for one round."""
        self.commit_phase(round_id, cov_snapshot)
        self.reveal_phase
