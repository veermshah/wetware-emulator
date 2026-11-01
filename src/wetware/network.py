"""Synaptic network of a 2D dissociated culture.

Holds connectivity, synaptic currents, short-term plasticity (Tsodyks-Markram
per presynaptic cell), pair-based STDP on excitatory synapses, background
drive and cell death. See ``docs/model.md`` for the equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .neurons import NeuronModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class NetworkParams:
    """Tunable parameters of the culture model.

    Distances are in µm, times in ms, currents in Izhikevich units.
    """

    exc_fraction: float = 0.8
    #: Connection probability at zero distance.
    p_connect_max: float = 0.35
    #: Length constant of the exponential distance rule.
    connect_lambda_um: float = 250.0
    #: Mean synaptic weights (log-normal heterogeneity around these).
    w_exc: float = 2.5
    w_inh: float = -4.0
    w_sigma: float = 0.5
    w_max: float = 6.0
    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    #: Short-term plasticity of excitatory output.
    stp_U_exc: float = 0.5
    stp_tau_rec_exc_ms: float = 4000.0
    stp_tau_facil_exc_ms: float = 50.0
    #: Short-term plasticity of inhibitory output.
    stp_U_inh: float = 0.2
    stp_tau_rec_inh_ms: float = 200.0
    stp_tau_facil_inh_ms: float = 400.0
    #: Pair-based STDP on E->* synapses.
    stdp_a_plus: float = 0.010
    stdp_a_minus: float = 0.0105
    stdp_tau_plus_ms: float = 20.0
    stdp_tau_minus_ms: float = 20.0
    #: Background drive: tonic bias (mean, sd) plus white noise.
    bias_mean: float = 0.0
    bias_sd: float = 1.0
    noise_sd: float = 2.0
    #: Fraction of excitatory cells that are spontaneously active pacemakers.
    pacemaker_fraction: float = 0.03
    pacemaker_bias: float = 4.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Network:
    """Connectivity plus synaptic state for ``n`` neurons."""

    def __init__(
        self,
        positions: FloatArray,
        model: NeuronModel,
        params: NetworkParams,
        rng_build: np.random.Generator,
        rng_dyn: np.random.Generator,
        plasticity: bool = True,
    ) -> None:
        self.positions = positions
        self.n = positions.shape[0]
        self.params = p = params
        self.model = model
        self.rng = rng_dyn
        self.plasticity = plasticity

        n = self.n
        n_exc = int(round(p.exc_fraction * n))
        self.is_exc = np.zeros(n, dtype=bool)
        self.is_exc[rng_build.permutation(n)[:n_exc]] = True

        # Distance-dependent random connectivity (no autapses).
        d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
        prob = p.p_connect_max * np.exp(-d / p.connect_lambda_um)
        np.fill_diagonal(prob, 0.0)
        self.conn = rng_build.random((n, n)) < prob  # [pre, post]

        mag = rng_build.lognormal(0.0, p.w_sigma, (n, n))
        w = np.where(self.is_exc[:, None], p.w_exc * mag, p.w_inh * mag)
        # Normalise by expected in-degree so behaviour is stable across n.
        indeg = max(1.0, float(self.conn.sum(0).mean()))
        scale = 60.0 / indeg
        self.w = np.where(self.conn, w * scale, 0.0)  # [pre, post]
        self.w_max = p.w_max * scale
        np.clip(self.w, -np.inf, self.w_max, out=self.w)
        self.exc_conn = self.conn & self.is_exc[:, None]
        self.weight_scale = 1.0

        # Synaptic currents (post side).
        self.i_exc = np.zeros(n)
        self.i_inh = np.zeros(n)

        # STP state (pre side).
        self.stp_U = np.where(self.is_exc, p.stp_U_exc, p.stp_U_inh)
        self.stp_tau_rec = np.where(self.is_exc, p.stp_tau_rec_exc_ms, p.stp_tau_rec_inh_ms)
        self.stp_tau_f = np.where(self.is_exc, p.stp_tau_facil_exc_ms, p.stp_tau_facil_inh_ms)
        self.stp_x = np.ones(n)
        self.stp_u = self.stp_U.copy()

        # STDP traces.
        self.trace_pre = np.zeros(n)
        self.trace_post = np.zeros(n)

        # Background drive.
        self.bias = rng_build.normal(p.bias_mean, p.bias_sd, n)
        exc_idx = np.flatnonzero(self.is_exc)
        n_pace = int(round(p.pacemaker_fraction * exc_idx.size))
        self.pacemakers = rng_build.choice(exc_idx, n_pace, replace=False)
        self.bias[self.pacemakers] = p.pacemaker_bias + rng_build.normal(0, 0.3, n_pace)
        self.drive_scale = 1.0

        self.alive = np.ones(n, dtype=bool)
        self._noise = np.empty((0, n))
        self._noise_i = 0
        model.init_state(self.is_exc, rng_build)

    # ------------------------------------------------------------------ dynamics
    _NOISE_BLOCK = 512

    def _constants(self, dt: float) -> None:
        """Cache per-step decay factors for step size ``dt``."""
        p = self.params
        self._dt = dt
        self._k_exc = 1.0 - dt / p.tau_exc_ms
        self._k_inh = 1.0 - dt / p.tau_inh_ms
        self._k_x = 1.0 - dt / self.stp_tau_rec
        self._k_u = 1.0 - dt / self.stp_tau_f
        self._k_pre = 1.0 - dt / p.stdp_tau_plus_ms
        self._k_post = 1.0 - dt / p.stdp_tau_minus_ms
        self._noise_scale = p.noise_sd / np.sqrt(dt)
        self._current = np.empty(self.n)

    def _next_noise(self) -> FloatArray:
        # Noise is drawn in fixed blocks, so results do not depend on how a
        # simulation is split into calls.
        if self._noise_i >= self._noise.shape[0]:
            self._noise = self.rng.standard_normal((self._NOISE_BLOCK, self.n))
            self._noise_i = 0
        row = self._noise[self._noise_i]
        self._noise_i += 1
        return row

    def step(self, dt: float, stim_dv: FloatArray | None = None) -> IntArray:
        """Advance one step; return indices of neurons that spiked."""
        if dt != getattr(self, "_dt", None):
            self._constants(dt)
        if stim_dv is not None:
            self.model.depolarize(stim_dv)

        cur = self._current
        np.multiply(self._next_noise(), self._noise_scale, out=cur)
        cur += self.i_exc
        cur += self.i_inh
        cur += self.drive_scale * self.bias
        fired = self.model.step(cur, dt)
        fired &= self.alive
        spk = np.flatnonzero(fired)

        # Decay synaptic currents; relax STP variables towards rest.
        self.i_exc *= self._k_exc
        self.i_inh *= self._k_inh
        x, u = self.stp_x, self.stp_u
        x -= 1.0
        x *= self._k_x
        x += 1.0
        u -= self.stp_U
        u *= self._k_u
        u += self.stp_U
        if self.plasticity:
            self.trace_pre *= self._k_pre
            self.trace_post *= self._k_post

        if spk.size:
            # Tsodyks-Markram: facilitate u, then release u*x, deplete x.
            U = self.stp_U[spk]
            us = u[spk]
            us += U * (1.0 - us)
            u[spk] = us
            release = us * x[spk]
            x[spk] -= release
            eff = (release / U) * self.weight_scale
            exc = self.is_exc[spk]
            if exc.all():
                self.i_exc += eff @ self.w[spk]
            elif not exc.any():
                self.i_inh += eff @ self.w[spk]
            else:
                self.i_exc += eff[exc] @ self.w[spk[exc]]
                self.i_inh += eff[~exc] @ self.w[spk[~exc]]
            if self.plasticity:
                self._stdp(spk)
        return spk

    #: Traces below this are treated as zero, which keeps STDP updates sparse.
    TRACE_EPS = 1e-3

    def _stdp(self, spk: IntArray) -> None:
        p = self.params
        w, wmax, eps = self.w, self.w_max, self.TRACE_EPS
        # Depression: an excitatory pre spike after post activity,
        # w[pre, post] -= A- * w_max * trace_post[post].
        pre_e = spk[self.is_exc[spk]]
        if pre_e.size:
            post_act = np.flatnonzero(self.trace_post > eps)
            if post_act.size:
                ix = np.ix_(pre_e, post_act)
                blk = w[ix] - (p.stdp_a_minus * wmax) * self.trace_post[post_act]
                w[ix] = np.where(self.exc_conn[ix], np.clip(blk, 0.0, wmax), 0.0)
        # Potentiation: a post spike after excitatory pre activity,
        # w[pre, post] += A+ * w_max * trace_pre[pre].
        pre_act = np.flatnonzero((self.trace_pre > eps) & self.is_exc)
        if pre_act.size:
            ix = np.ix_(pre_act, spk)
            blk = w[ix] + (p.stdp_a_plus * wmax) * self.trace_pre[pre_act][:, None]
            w[ix] = np.where(self.exc_conn[ix], np.clip(blk, 0.0, wmax), 0.0)
        self.trace_pre[spk] += 1.0
        self.trace_post[spk] += 1.0

    # ------------------------------------------------------------------ health
    def kill(self, mask: NDArray[np.bool_]) -> None:
        """Remove neurons (cell death)."""
        self.alive &= ~mask
        self.model.silence(mask)

    def mean_exc_weight(self) -> float:
        """Mean weight over existing excitatory synapses."""
        return float(self.w[self.exc_conn].mean()) if self.exc_conn.any() else 0.0

    def get_state(self) -> dict[str, Any]:
        return {
            "w": self.w.copy(),
            "i_exc": self.i_exc.copy(),
            "i_inh": self.i_inh.copy(),
            "stp_x": self.stp_x.copy(),
            "stp_u": self.stp_u.copy(),
            "trace_pre": self.trace_pre.copy(),
            "trace_post": self.trace_post.copy(),
            "alive": self.alive.copy(),
            "drive_scale": self.drive_scale,
            "weight_scale": self.weight_scale,
            "rng": self.rng.bit_generator.state,
            "noise": self._noise.copy(),
            "noise_i": self._noise_i,
            "model": self.model.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        for k in ("w", "i_exc", "i_inh", "stp_x", "stp_u", "trace_pre", "trace_post", "alive"):
            setattr(self, k, state[k].copy())
        self.drive_scale = state["drive_scale"]
        self.weight_scale = state["weight_scale"]
        self.rng.bit_generator.state = state["rng"]
        self._noise = state["noise"].copy()
        self._noise_i = state["noise_i"]
        self.model.set_state(state["model"])
