"""Causal Rest-Supervised Nuisance Subspace Estimation (RSNSE) Riemannian organ."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import sosfiltfilt
from scipy.special import erf as scipy_erf
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.riemannian_organs.RiemannianOrgan import RiemannianOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class RestSubspaceNuisanceRiemannianOrgan(RiemannianOrgan):
    """Task-protected rest-nuisance subspace subtraction + L2 LR and LayerNorm-GELU MLP ensemble (rsnse_t1)."""

    ARTIFACT_FILENAME = "rsnse_t1.npz"
    SPEC_BANDS_7_HZ: tuple[tuple[float, float], ...] = (
        (1.0, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 45.0),
        (1.0, 45.0),
    )
    RIEM_BANDS_6_HZ: tuple[tuple[float, float], ...] = (
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 45.0),
        (1.0, 45.0),
    )
    RIEM_BANDS_6_INT: tuple[tuple[int, int], ...] = (
        (4, 8),
        (8, 13),
        (13, 20),
        (20, 30),
        (30, 45),
        (1, 45),
    )
    SLICE_EARLY = slice(0, 192)
    SLICE_LATE = slice(264, 480)

    def __init__(
        self,
        weight_tensors: dict[str, np.ndarray],
        sampling_rate_hz: float = 120.0,
        stream_mode: str = "shipped",
        pc_tau: float = 1.5,
    ) -> None:
        super().__init__(organ_identifier="rest_subspace_nuisance_rsnse")
        self._weights = {k_str: np.asarray(arr).copy() for k_str, arr in weight_tensors.items()}
        self._sampling_rate_hz = ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", 1.0)
        self._stream_mode = str(stream_mode)
        self._pc_tau = float(pc_tau)

        self._inv_std0 = np.asarray(self._weights["inv_std0"], dtype=np.float64)
        self._scale2 = (self._inv_std0 ** 2).astype(np.float64)
        self._w0_6b = np.asarray(self._weights["W0_6b"], dtype=np.float64)
        self._tang_mu = np.asarray(self._weights["tang_mu"], dtype=np.float64)
        self._tang_sd = np.asarray(self._weights["tang_sd"], dtype=np.float64)
        self._p_tang = np.asarray(self._weights["P_tang"], dtype=np.float64)
        self._z1_mu = np.asarray(self._weights["z1_mu"], dtype=np.float64)
        self._z1_sd = np.asarray(self._weights["z1_sd"], dtype=np.float64)

        self._mu_0 = np.asarray(self._weights["mu_0"], dtype=np.float64)
        self._v_basis = np.asarray(self._weights["V"], dtype=np.float64)
        self._u_basis = np.asarray(self._weights["U"], dtype=np.float64)
        self._g_lam = np.asarray(self._weights["G_lam"], dtype=np.float64)
        self._clip_bound = np.asarray(self._weights["clip_bound"], dtype=np.float64)

        self._w_lr = np.asarray(self._weights["W_lr"], dtype=np.float64)
        self._b_lr = np.asarray(self._weights["b_lr"], dtype=np.float64)
        self._mlp_ln_w = np.asarray(self._weights["mlp_ln_w"], dtype=np.float32)
        self._mlp_ln_b = np.asarray(self._weights["mlp_ln_b"], dtype=np.float32)
        self._mlp_w1 = np.asarray(self._weights["mlp_w1"], dtype=np.float32)
        self._mlp_b1 = np.asarray(self._weights["mlp_b1"], dtype=np.float32)
        self._mlp_w2 = np.asarray(self._weights["mlp_w2"], dtype=np.float32)
        self._mlp_b2 = np.asarray(self._weights["mlp_b2"], dtype=np.float32)
        self._n_mlp = int(self._mlp_w1.shape[0])

        self._n0_shrink = float(self._weights["n0_shrink"][0]) if "n0_shrink" in self._weights else 16.0
        self._ema_decay = float(self._weights["ema_decay"][0]) if "ema_decay" in self._weights else 0.95
        self._alpha_ema = 1.0 - self._ema_decay
        self._n_times = int(self._weights["n_times"][0]) if "n_times" in self._weights else 480

        self._sos_7 = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(band_pair) for band_pair in self.SPEC_BANDS_7_HZ],
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self._sos_6 = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(band_pair) for band_pair in self.RIEM_BANDS_6_HZ],
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self._m_sos_7 = SpectralSignalOperator.precomputed_sos_operators(
            self.SPEC_BANDS_7_HZ,
            sample_count=self._n_times,
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self._m_sos_6 = SpectralSignalOperator.precomputed_sos_operators(
            self.RIEM_BANDS_6_HZ,
            sample_count=self._n_times,
            sampling_rate_hz=self._sampling_rate_hz,
        )

        self._n_past = 0
        self._z_ema = self._mu_0.copy()
        self._n_eff_ema = 0.0
        self._z_cum = np.zeros_like(self._mu_0, dtype=np.float64)
        self._u_hist: list[np.ndarray] = []
        self._z_hist: list[np.ndarray] = []
        self._last_z1: np.ndarray | None = None
        self._last_z_star: np.ndarray | None = None
        self.reset_organ_state()

    @property
    def last_z1_features(self) -> np.ndarray | None:
        return self._last_z1

    @property
    def last_debiased_features(self) -> np.ndarray | None:
        return self._last_z_star

    def reset_organ_state(self) -> None:
        self._n_past = 0
        self._z_ema = self._mu_0.copy()
        self._n_eff_ema = 0.0
        self._z_cum = np.zeros_like(self._mu_0, dtype=np.float64)
        self._u_hist = []
        self._z_hist = []
        self._last_z1 = None
        self._last_z_star = None

    @classmethod
    def load_from_npz(cls, rsnse_npz_path: Path) -> RestSubspaceNuisanceRiemannianOrgan:
        resolved_path = ParameterTypeValidator.require_path(rsnse_npz_path, "rsnse_npz_path", must_exist=True)
        with np.load(resolved_path, allow_pickle=False) as archive:
            loaded_weights = {key_str: archive[key_str].copy() for key_str in archive.files}
        return cls(weight_tensors=loaded_weights)

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        out_path = resolved_dir / self.ARTIFACT_FILENAME
        np.savez_compressed(out_path, **self._weights)
        return [out_path]

    @staticmethod
    def _softmax_1d(logits_in: np.ndarray) -> np.ndarray:
        z64 = np.asarray(logits_in, dtype=np.float64)
        z64 = z64 - np.max(z64)
        exps = np.exp(z64)
        return exps / np.sum(exps)

    @staticmethod
    def _norm_lp_1d(probs_in: np.ndarray) -> np.ndarray:
        lp_vec = np.log(np.clip(np.asarray(probs_in, dtype=np.float64), 1e-6, 1.0))
        return lp_vec - np.mean(lp_vec)

    @staticmethod
    def _lw_cov_single(signal_2d: np.ndarray) -> np.ndarray:
        centered = signal_2d - np.mean(signal_2d, axis=-1, keepdims=True)
        chan_cnt, time_cnt = centered.shape
        sample_s = (centered @ centered.T) / float(time_cnt)
        eye_mat = np.eye(chan_cnt, dtype=np.float64)
        mu_tr = float(np.trace(sample_s)) / float(chan_cnt)
        delta_term = float(np.sum((sample_s - mu_tr * eye_mat) ** 2)) / float(chan_cnt)
        sq_sig = centered ** 2
        beta_term = float(np.sum(((sq_sig @ sq_sig.T) / float(time_cnt) - sample_s ** 2))) / float(
            chan_cnt * time_cnt
        )
        beta_term = min(beta_term, delta_term)
        shrink_s = (beta_term / max(delta_term, 1e-300)) if delta_term > 0.0 else 0.0
        return (1.0 - shrink_s) * sample_s + (shrink_s * mu_tr) * eye_mat

    def extract_standardized_661d_features(
        self,
        cleaned_window: np.ndarray,
        known_covariances: dict | None = None,
    ) -> np.ndarray:
        xb = np.asarray(cleaned_window, dtype=np.float64)
        if not np.isfinite(xb).all():
            xb = np.nan_to_num(xb, nan=0.0, posinf=500.0, neginf=-500.0)
        xz = xb - np.mean(xb, axis=-1, keepdims=True)
        x_car = xz - np.mean(xz, axis=0, keepdims=True)

        time_len = xz.shape[-1]
        early_end = min(time_len, max(2, int(round(192.0 * time_len / 480.0))))
        late_start = min(time_len - 2, int(round(264.0 * time_len / 480.0)))
        early_slice = self.SLICE_EARLY if time_len == 480 else slice(0, early_end)
        late_slice = self.SLICE_LATE if time_len == 480 else slice(late_start, time_len)

        if time_len == self._n_times:
            xf_7 = np.matmul(x_car[None], self._m_sos_7)
        else:
            xf_7 = np.stack([sosfiltfilt(s_mat, x_car, axis=-1) for s_mat in self._sos_7], axis=0)
        vf = np.mean(xf_7 ** 2, axis=-1)
        ve = np.mean(xf_7[:6, :, early_slice] ** 2, axis=-1)
        vl = np.mean(xf_7[:6, :, late_slice] ** 2, axis=-1)

        lp_vec = np.log(np.maximum(vf * self._scale2[None, :], 1e-6)).reshape(-1)
        re_vec = (
            np.log(np.maximum(vl * self._scale2[None, :], 1e-6))
            - np.log(np.maximum(ve * self._scale2[None, :], 1e-6))
        ).reshape(-1)
        spec_vec = np.concatenate([lp_vec, re_vec], axis=0)

        known_dict = known_covariances if known_covariances is not None else {}
        chan_cnt = xz.shape[0]
        cb6 = np.empty((6, chan_cnt, chan_cnt), dtype=np.float64)
        missing_idx = [
            bi
            for bi, (bk_int, bk_flt) in enumerate(zip(self.RIEM_BANDS_6_INT, self.RIEM_BANDS_6_HZ))
            if (bk_int not in known_dict and bk_flt not in known_dict)
        ]
        if missing_idx:
            if time_len == self._n_times:
                xf_m = np.matmul(xz[None], self._m_sos_6[missing_idx])
                for loc_i, bi in enumerate(missing_idx):
                    cb6[bi] = self._lw_cov_single(xf_m[loc_i])
            else:
                for bi in missing_idx:
                    cb6[bi] = self._lw_cov_single(sosfiltfilt(self._sos_6[bi], xz, axis=-1))

        for bi, (bk_int, bk_flt) in enumerate(zip(self.RIEM_BANDS_6_INT, self.RIEM_BANDS_6_HZ)):
            if bk_int in known_dict:
                cb6[bi] = known_dict[bk_int]
            elif bk_flt in known_dict:
                cb6[bi] = known_dict[bk_flt]
            elif known_covariances is not None:
                known_covariances[bk_int] = cb6[bi]

        aligned_6 = self._w0_6b @ cb6 @ self._w0_6b
        tang_raw = CovarianceMatrixOperator.project_to_tangent_vector(aligned_6).reshape(-1)
        tang_std = (tang_raw - self._tang_mu) / self._tang_sd
        z1_unstd = np.concatenate([spec_vec, tang_std @ self._p_tang], axis=0)
        return (z1_unstd - self._z1_mu) / self._z1_sd

    def _predict_mlp_ensemble_mean(self, z_star: np.ndarray) -> np.ndarray:
        z32 = z_star.astype(np.float32)
        z_mean = float(np.mean(z32))
        z_var = float(np.mean((z32 - z_mean) ** 2))
        z_norm = (z32 - z_mean) / np.float32(np.sqrt(z_var + 1e-5))

        prob_sum = np.zeros(3, dtype=np.float64)
        inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
        for seed_i in range(self._n_mlp):
            zn = z_norm * self._mlp_ln_w[seed_i] + self._mlp_ln_b[seed_i]
            hid = self._mlp_w1[seed_i] @ zn + self._mlp_b1[seed_i]
            hid_gelu = np.float32(0.5) * hid * (np.float32(1.0) + scipy_erf(hid * inv_sqrt2).astype(np.float32))
            logits = self._mlp_w2[seed_i] @ hid_gelu + self._mlp_b2[seed_i]
            prob_sum += self._softmax_1d(logits)
        return prob_sum / float(self._n_mlp)

    def step_causal_window(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        known_covariances: dict | None = None,
        pseudo_chrono_adapter: Any = None,
    ) -> np.ndarray:
        zt = self.extract_standardized_661d_features(cleaned_window, known_covariances=known_covariances)
        self._last_z1 = zt

        cur_u = (
            None
            if pseudo_chrono_adapter is None
            else getattr(pseudo_chrono_adapter, "cur_u", getattr(pseudo_chrono_adapter, "current_embedding", None))
        )
        if self._n_past == 0:
            z_star = zt
        else:
            gamma_shrink = self._n_eff_ema / (self._n_eff_ema + self._n0_shrink)
            if self._stream_mode == "pc_gema" and pseudo_chrono_adapter is not None and cur_u is not None and self._u_hist:
                u_mat = np.stack(self._u_hist, axis=0)
                sq_dist = np.sum((u_mat - np.asarray(cur_u, dtype=np.float64)[None, :]) ** 2, axis=1)
                w_rs = np.exp(-0.5 * sq_dist / (float(u_mat.shape[1]) * (self._pc_tau ** 2)))
                ws_rs = float(w_rs.sum())
                z_bar = (
                    (w_rs @ np.stack(self._z_hist, axis=0)) / ws_rs
                    if ws_rs > 1e-12
                    else (self._z_cum / float(self._n_past))
                )
            elif self._stream_mode == "cum_gema":
                z_bar = self._z_cum / float(self._n_past)
            else:
                z_bar = self._z_ema

            diff_vec = z_bar - self._mu_0
            orth_diff = diff_vec - self._v_basis @ (self._v_basis.T @ diff_vec)
            a_hat = np.clip(self._g_lam @ diff_vec, -self._clip_bound, self._clip_bound)
            v_bridged = self._v_basis @ (self._v_basis.T @ (self._u_basis @ a_hat))
            z_star = zt - gamma_shrink * (orth_diff + v_bridged)

        self._last_z_star = z_star
        if not is_artifact_flagged:
            if self._n_past == 0:
                self._z_ema = zt.copy()
                self._n_eff_ema = 1.0
            else:
                self._z_ema = self._ema_decay * self._z_ema + self._alpha_ema * zt
                self._n_eff_ema = self._ema_decay * self._n_eff_ema + 1.0
            self._n_past += 1
            self._z_cum = self._z_cum + zt
            if pseudo_chrono_adapter is not None and cur_u is not None:
                max_h = int(getattr(pseudo_chrono_adapter, "max_hist", 512))
                self._u_hist.append(np.asarray(cur_u, dtype=np.float64))
                self._z_hist.append(zt.copy())
                if len(self._u_hist) > max_h:
                    self._u_hist.pop(0)
                    self._z_hist.pop(0)

        probs_lr = self._softmax_1d(self._w_lr @ z_star + self._b_lr)
        if self._n_mlp <= 0:
            return probs_lr
        probs_mlp = self._predict_mlp_ensemble_mean(z_star)
        return self._softmax_1d(0.55 * self._norm_lp_1d(probs_lr) + 0.45 * self._norm_lp_1d(probs_mlp))

    @classmethod
    def _precompute_spec_and_cov6(
        cls,
        windows_x: np.ndarray,
        sos_7: list[np.ndarray],
        sampling_rate_hz: float,
        chunk_size: int = 256,
    ) -> dict[str, np.ndarray]:
        win_cnt, chan_cnt, time_len = windows_x.shape
        early_end = min(time_len, max(2, int(round(192.0 * time_len / 480.0))))
        late_start = min(time_len - 2, int(round(264.0 * time_len / 480.0)))
        early_sl = cls.SLICE_EARLY if time_len == 480 else slice(0, early_end)
        late_sl = cls.SLICE_LATE if time_len == 480 else slice(late_start, time_len)

        v_full = np.empty((win_cnt, 7, chan_cnt), dtype=np.float64)
        v_early = np.empty((win_cnt, 6, chan_cnt), dtype=np.float64)
        v_late = np.empty((win_cnt, 6, chan_cnt), dtype=np.float64)
        cb6 = np.empty((win_cnt, 6, chan_cnt, chan_cnt), dtype=np.float64)

        for s0 in range(0, win_cnt, chunk_size):
            s1 = min(win_cnt, s0 + chunk_size)
            xb = np.asarray(windows_x[s0:s1], dtype=np.float64)
            if not np.isfinite(xb).all():
                xb = np.nan_to_num(xb, nan=0.0, posinf=500.0, neginf=-500.0)
            xz = xb - xb.mean(axis=-1, keepdims=True)
            x_car = xz - xz.mean(axis=1, keepdims=True)
            for b_i, sos_mat in enumerate(sos_7):
                xf = sosfiltfilt(sos_mat, x_car, axis=-1)
                v_full[s0:s1, b_i] = (xf ** 2).mean(axis=-1)
                if b_i < 6:
                    v_early[s0:s1, b_i] = (xf[..., early_sl] ** 2).mean(axis=-1)
                    v_late[s0:s1, b_i] = (xf[..., late_sl] ** 2).mean(axis=-1)
            cb6[s0:s1] = SpectralSignalOperator.compute_band_covariances(
                xz,
                [list(b_pair) for b_pair in cls.RIEM_BANDS_6_HZ],
                sampling_rate_hz=sampling_rate_hz,
                chunk_size=chunk_size,
            )[:, :-1]
        return {"v_full": v_full, "v_early": v_early, "v_late": v_late, "cb6": cb6}

    @staticmethod
    def _assemble_t1_blocks(
        precomp: dict[str, np.ndarray],
        inv_std0: np.ndarray,
        w0_6b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        scale2 = (inv_std0 ** 2)[None, :]
        vf, ve, vl = precomp["v_full"], precomp["v_early"], precomp["v_late"]
        lp_list = [np.log(np.maximum(vf[:, b_i] * scale2, 1e-6)) for b_i in range(7)]
        re_list = [
            np.log(np.maximum(vl[:, b_i] * scale2, 1e-6)) - np.log(np.maximum(ve[:, b_i] * scale2, 1e-6))
            for b_i in range(6)
        ]
        spec_out = np.concatenate(lp_list + re_list, axis=1)
        aligned_6 = w0_6b[None] @ precomp["cb6"] @ w0_6b[None]
        tang_out = CovarianceMatrixOperator.project_to_tangent_vector(aligned_6).reshape(len(vf), -1)
        return spec_out, tang_out

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        mlp_seeds: int = 5,
        random_seed: int = 0,
    ) -> RestSubspaceNuisanceRiemannianOrgan:
        validated_mlp_seeds = ParameterTypeValidator.require_integer(mlp_seeds, "mlp_seeds", 1)
        validated_seed = ParameterTypeValidator.require_integer(random_seed, "random_seed", 0)

        x_tr = np.asarray(corpus_container.eeg_windows, dtype=np.float64)
        y_tr = np.asarray(corpus_container.class_labels, dtype=np.int64)
        sess_tr = np.asarray(corpus_container.session_ids)
        part_tr = np.asarray(corpus_container.participant_ids)
        sfreq_hz = corpus_container.sampling_rate_hz
        n_times = int(x_tr.shape[-1])

        sos_7 = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(b_pair) for b_pair in cls.SPEC_BANDS_7_HZ],
            sampling_rate_hz=sfreq_hz,
        )
        pre_tr = cls._precompute_spec_and_cov6(x_tr, sos_7, sfreq_hz)
        uniq_sess = list(dict.fromkeys(sess_tr.tolist()))

        xz_tr = x_tr - x_tr.mean(axis=-1, keepdims=True)
        x_car_tr = xz_tr - xz_tr.mean(axis=1, keepdims=True)
        var_win = np.maximum((x_car_tr ** 2).mean(axis=-1), 1e-8)
        r0_var = np.mean([np.maximum(var_win[sess_tr == sk].mean(axis=0), 1e-6) for sk in uniq_sess], axis=0)
        inv_std0 = 1.0 / np.sqrt(np.maximum(r0_var, 1e-6))

        cb6_tr = pre_tr["cb6"]
        r0_6b = np.mean([cb6_tr[sess_tr == sk].mean(axis=0) for sk in uniq_sess], axis=0)
        w0_6b = CovarianceMatrixOperator.inverse_matrix_square_root(r0_6b)

        spec_tr, tang_tr = cls._assemble_t1_blocks(pre_tr, inv_std0, w0_6b)
        tang_mu = tang_tr.mean(axis=0)
        tang_sd = np.maximum(tang_tr.std(axis=0), 1e-5)
        tang_std = (tang_tr - tang_mu) / tang_sd
        _, _, vt_t = np.linalg.svd(tang_std, full_matrices=False)
        target_tang_dim = min(128, tang_std.shape[1])
        p_tang = vt_t[:target_tang_dim].T.copy()
        if p_tang.shape[1] < target_tang_dim:
            rng_pad = np.random.default_rng(validated_seed + 137)
            rand_cols = rng_pad.standard_normal((tang_std.shape[1], target_tang_dim - p_tang.shape[1]))
            p_tang, _ = np.linalg.qr(np.concatenate([p_tang, rand_cols], axis=1))
            p_tang = p_tang[:, :target_tang_dim].copy()

        z1_tr_u = np.concatenate([spec_tr, tang_std @ p_tang], axis=1)
        z1_mu = z1_tr_u.mean(axis=0)
        z1_sd = np.maximum(z1_tr_u.std(axis=0), 1e-5)
        z1_tr = (z1_tr_u - z1_mu) / z1_sd
        feat_dim = z1_tr.shape[1]

        sess_cls_means: dict[str, np.ndarray] = {}
        sess_bal_means: dict[str, np.ndarray] = {}
        contrast_rows = []
        for sk in uniq_sess:
            ms = sess_tr == sk
            c_means = [
                z1_tr[ms & (y_tr == c_idx)].mean(axis=0) if np.any(ms & (y_tr == c_idx)) else z1_tr[ms].mean(axis=0)
                for c_idx in range(3)
            ]
            c_arr = np.stack(c_means, axis=0)
            b_mean = c_arr.mean(axis=0)
            sess_cls_means[str(sk)] = c_arr
            sess_bal_means[str(sk)] = b_mean
            for c_idx in range(3):
                contrast_rows.append(c_arr[c_idx] - b_mean)

        mu_0 = np.mean([sess_bal_means[str(sk)] for sk in uniq_sess], axis=0)
        pop_contrasts = np.mean(
            [sess_cls_means[str(sk)] - sess_bal_means[str(sk)][None, :] for sk in uniq_sess],
            axis=0,
        )

        r_tilde_rows = []
        if corpus_container.resting_windows is not None and corpus_container.resting_session_ids is not None:
            pre_rest = cls._precompute_spec_and_cov6(
                np.asarray(corpus_container.resting_windows, dtype=np.float64),
                sos_7,
                sfreq_hz,
            )
            spec_r, tang_r = cls._assemble_t1_blocks(pre_rest, inv_std0, w0_6b)
            z1_rest = (np.concatenate([spec_r, ((tang_r - tang_mu) / tang_sd) @ p_tang], axis=1) - z1_mu) / z1_sd
            rest_sess = np.asarray(corpus_container.resting_session_ids)
            ce_dict, oe_dict = {}, {}
            for sk in uniq_sess:
                rm = rest_sess == sk
                if int(rm.sum()) >= 2:
                    zr_s = z1_rest[rm]
                    mid_r = len(zr_s) // 2
                    ce_dict[str(sk)] = zr_s[:mid_r].mean(axis=0)
                    oe_dict[str(sk)] = zr_s[mid_r:].mean(axis=0)
                elif int(rm.sum()) == 1:
                    ce_dict[str(sk)] = z1_rest[rm][0]
                    oe_dict[str(sk)] = z1_rest[rm][0]
            if len(ce_dict) >= 2:
                ce_bar = np.mean(list(ce_dict.values()), axis=0)
                oe_bar = np.mean(list(oe_dict.values()), axis=0)
                for sk_s in ce_dict:
                    r_tilde_rows.append(ce_dict[sk_s] - ce_bar)
                for sk_s in oe_dict:
                    r_tilde_rows.append(oe_dict[sk_s] - oe_bar)

        if len(r_tilde_rows) < 2:
            for sk in uniq_sess:
                r_tilde_rows.append(sess_bal_means[str(sk)] - mu_0)
        r_tilde = np.stack(r_tilde_rows, axis=0)
        _, _, vt_rest = np.linalg.svd(r_tilde, full_matrices=False)
        k_use = min(4, vt_rest.shape[0])
        u0_mat = vt_rest[:k_use].T.copy()

        _, _, vt_pop = np.linalg.svd(pop_contrasts, full_matrices=False)
        v_pop = vt_pop[:2].T
        c_mat = np.stack(contrast_rows, axis=0)
        c_perp = c_mat - (c_mat @ v_pop) @ v_pop.T
        _, _, vt_rem = np.linalg.svd(c_perp, full_matrices=False)
        n_extra = min(4, vt_rem.shape[0])
        v_mat = np.concatenate([v_pop, vt_rem[:n_extra].T], axis=1)

        a0_mat = u0_mat - v_mat @ (v_mat.T @ u0_mat)
        _, sv_pu, vt_pu = np.linalg.svd(a0_mat, full_matrices=False)
        keep_mask = sv_pu >= 0.25
        if not bool(keep_mask.any()):
            keep_mask[0] = True
        u_mat = u0_mat @ vt_pu.T[:, keep_mask]
        a_mat = u_mat - v_mat @ (v_mat.T @ u_mat)
        k_eff = u_mat.shape[1]
        g_lam = np.linalg.solve(a_mat.T @ a_mat + 0.15 * np.eye(k_eff), a_mat.T)

        a_hat_sess = {str(sk): g_lam @ (z1_tr[sess_tr == sk].mean(axis=0) - mu_0) for sk in uniq_sess}
        a_tr_mat = np.stack([a_hat_sess[str(sk)] for sk in uniq_sess], axis=0)
        ddof_a = 1 if len(uniq_sess) > 1 else 0
        clip_bound = 3.0 * np.maximum(a_tr_mat.std(axis=0, ddof=ddof_a), 0.35)

        z_clean = np.empty_like(z1_tr)
        for sk in uniq_sess:
            ms = sess_tr == sk
            z_clean[ms] = z1_tr[ms] - (u_mat @ a_hat_sess[str(sk)])[None, :]

        _, inv_p, cnt_p = np.unique(part_tr, return_inverse=True, return_counts=True)
        w_part = (1.0 / cnt_p)[inv_p]
        w_part = (w_part / w_part.mean()).astype(np.float64)

        lr_clf = LogisticRegression(C=0.08, max_iter=500, tol=1e-4)
        lr_clf.fit(z_clean, y_tr, sample_weight=w_part)
        w_lr = np.zeros((3, feat_dim), dtype=np.float64)
        b_lr = np.zeros(3, dtype=np.float64)
        for loc_i, cls_val in enumerate(lr_clf.classes_):
            w_lr[int(cls_val)] = lr_clf.coef_[loc_i]
            b_lr[int(cls_val)] = lr_clf.intercept_[loc_i]

        zt_gpu = torch.as_tensor(z_clean, dtype=torch.float32)
        yt_gpu = torch.as_tensor(y_tr, dtype=torch.long)
        wt_gpu = torch.as_tensor(w_part, dtype=torch.float32)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
        mlp_ln_w, mlp_ln_b, mlp_w1, mlp_b1, mlp_w2, mlp_b2 = [], [], [], [], [], []
        bs = min(64, max(4, len(z_clean)))
        epochs = 35

        for sd_idx in range(validated_mlp_seeds):
            cur_seed = 500 + validated_seed + sd_idx
            torch.manual_seed(cur_seed)
            cpu_gen = torch.Generator(device="cpu").manual_seed(cur_seed)
            mlp_seq = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 128),
                nn.GELU(),
                nn.Dropout(0.45),
                nn.Linear(128, 3),
            )
            opt = torch.optim.AdamW(mlp_seq.parameters(), lr=8e-4, weight_decay=3e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((len(z_clean) + bs - 1) // bs))
            mlp_seq.train()
            for _ in range(epochs):
                perm = torch.randperm(len(z_clean), generator=cpu_gen)
                for s0 in range(0, len(z_clean), bs):
                    bi = perm[s0 : s0 + bs]
                    step_loss = (ce_loss(mlp_seq(zt_gpu[bi]), yt_gpu[bi]) * wt_gpu[bi]).mean()
                    opt.zero_grad(set_to_none=True)
                    step_loss.backward()
                    opt.step()
                    sched.step()
            mlp_seq.eval()
            mlp_ln_w.append(mlp_seq[0].weight.detach().numpy().astype(np.float32))
            mlp_ln_b.append(mlp_seq[0].bias.detach().numpy().astype(np.float32))
            mlp_w1.append(mlp_seq[1].weight.detach().numpy().astype(np.float32))
            mlp_b1.append(mlp_seq[1].bias.detach().numpy().astype(np.float32))
            mlp_w2.append(mlp_seq[4].weight.detach().numpy().astype(np.float32))
            mlp_b2.append(mlp_seq[4].bias.detach().numpy().astype(np.float32))

        packed = {
            "n_times": np.array([n_times], dtype=np.int64),
            "n0_shrink": np.array([16.0], dtype=np.float64),
            "ema_decay": np.array([0.95], dtype=np.float64),
            "inv_std0": inv_std0.astype(np.float64),
            "W0_6b": w0_6b.astype(np.float64),
            "tang_mu": tang_mu.astype(np.float64),
            "tang_sd": tang_sd.astype(np.float64),
            "P_tang": p_tang.astype(np.float64),
            "z1_mu": z1_mu.astype(np.float64),
            "z1_sd": z1_sd.astype(np.float64),
            "mu_0": mu_0.astype(np.float64),
            "V": v_mat.astype(np.float64),
            "U": u_mat.astype(np.float64),
            "G_lam": g_lam.astype(np.float64),
            "clip_bound": clip_bound.astype(np.float64),
            "W_lr": w_lr.astype(np.float64),
            "b_lr": b_lr.astype(np.float64),
            "mlp_ln_w": np.stack(mlp_ln_w, axis=0),
            "mlp_ln_b": np.stack(mlp_ln_b, axis=0),
            "mlp_w1": np.stack(mlp_w1, axis=0),
            "mlp_b1": np.stack(mlp_b1, axis=0),
            "mlp_w2": np.stack(mlp_w2, axis=0),
            "mlp_b2": np.stack(mlp_b2, axis=0),
        }
        return cls(weight_tensors=packed, sampling_rate_hz=sfreq_hz)
