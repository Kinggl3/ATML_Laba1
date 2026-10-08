"""
labkit — общий код ЛР1: загрузка данных, окна, метрики, проверка ответов, журнал запусков.

Один и тот же файл используется во всех ноутбуках лабы, чтобы протокол (PROTOCOL.md)
нигде не разъехался: одна подвыборка, одни окна, одни метрики.

На Kaggle: ноутбук lab_00 кладёт этот файл в /kaggle/working; остальные ноутбуки подключают
вывод lab_00 как Input и делают `sys.path.append(<папка с labkit.py>)`.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CONTEXT = 336            # часов контекста (2 недели) — основной вариант
CONTEXTS = (168, 336, 672)  # проверка чувствительности к длине контекста
HORIZON = 24             # часов прогноза
SEASON = 168             # недельная сезонность для наивного прогноза и MASE
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_PER_CLUSTER = 10
ORIGIN_EVERY_DAYS = 6       # каждый 6-й день теста: 15 точек, по очереди попадают на все дни недели
VAL_DEBUG_DAYS = ["2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22", "2026-06-29"]
INVALID_MAX_FACTOR = 3.0  # прогноз > 3 × максимум станции на train считается невалидным


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def find_data_root() -> Path:
    """Папка, где лежат lab/ и graph/. Ищет на Kaggle, затем локально."""
    env = os.environ.get("VKR_DATA_ROOT")
    if env:
        return Path(env)
    hits = glob.glob("/kaggle/input/**/lab/split.json", recursive=True)
    if hits:
        return Path(hits[0]).parent.parent
    for cand in [Path("dataset_kaggle"), Path("../dataset_kaggle"),        # локальная копия датасета
                 Path("data/processed"), Path("../data/processed")]:
        if (cand / "lab" / "split.json").exists():
            return cand
    raise FileNotFoundError("Не найден lab/split.json: подключите датасет vkr-transit-lab или задайте VKR_DATA_ROOT")


@dataclass
class LabData:
    y: pd.DataFrame          # час × станция, входы
    closed: pd.DataFrame     # час × станция, True = закрыта
    cov: pd.DataFrame        # час × ковариаты
    static: pd.DataFrame     # станции (индекс = station_complex_id)
    split: dict
    root: Path

    def part(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        a, b = self.split[name]
        return pd.Timestamp(a), pd.Timestamp(b)

    def train_max(self) -> pd.Series:
        """Максимум входов станции на train (только открытые часы) — для проверки валидности."""
        a, b = self.part("train")
        yy = self.y.loc[a:b].where(~self.closed.loc[a:b])
        return yy.max()


def load(root: Path | None = None) -> LabData:
    root = Path(root) if root else find_data_root()
    lab = root / "lab"
    y = pd.read_parquet(lab / "ridership_wide.parquet")
    closed = pd.read_parquet(lab / "closed_wide.parquet")
    cov = pd.read_parquet(lab / "covariates_time.parquet")
    static = pd.read_parquet(lab / "station_static.parquet")
    static["station_complex_id"] = static["station_complex_id"].astype(str)
    static = static.set_index("station_complex_id")
    split = json.loads((lab / "split.json").read_text(encoding="utf-8"))
    y.columns = y.columns.astype(str)
    closed.columns = closed.columns.astype(str)
    assert list(y.columns) == list(closed.columns), "столбцы ridership_wide и closed_wide не совпадают"
    assert y.index.equals(closed.index), "индексы ridership_wide и closed_wide не совпадают"
    return LabData(y=y, closed=closed, cov=cov, static=static, split=split, root=root)


# ---------------------------------------------------------------------------
# Подвыборка и окна
# ---------------------------------------------------------------------------
def make_subset(d: LabData, n_per_cluster: int = N_PER_CLUSTER, seed: int = SEED) -> list[str]:
    """По n станций из каждого кластера (кластеры посчитаны по train), случайно с фиксированным seed."""
    st = d.static[d.static.index.isin(d.y.columns) & d.static["cluster"].notna()]
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(st["cluster"].unique()):
        ids = sorted(st.index[st["cluster"] == c])
        out += list(rng.choice(ids, size=min(n_per_cluster, len(ids)), replace=False))
    return sorted(out)


def test_origins(d: LabData, every_days: int = ORIGIN_EVERY_DAYS) -> list[pd.Timestamp]:
    """Точки прогноза: каждый every_days-й день теста в 00:00, если впереди целые сутки."""
    a, b = d.part("test")
    days = pd.date_range(a.normalize(), b.normalize(), freq=f"{every_days}D")
    return [t for t in days if t + pd.Timedelta(hours=HORIZON - 1) <= b]


def val_origins() -> list[pd.Timestamp]:
    return [pd.Timestamp(x) for x in VAL_DEBUG_DAYS]


@dataclass
class Windows:
    meta: pd.DataFrame       # station, origin (одна строка на окно)
    ctx: np.ndarray          # [N, context] — история
    target: np.ndarray       # [N, horizon] — факт
    mask: np.ndarray         # [N, horizon] — True = час учитывается в метриках
    ctx_index: list          # временные метки контекста для каждого окна (для ковариат)

    def __len__(self):
        return len(self.meta)


def make_windows(d: LabData, stations: list[str], origins: list[pd.Timestamp],
                 context: int = CONTEXT, horizon: int = HORIZON) -> Windows:
    """Окна «контекст → горизонт» по позиции в почасовом индексе.

    Позиционный сдвиг корректен: индекс — полная сетка часов (весенний несуществующий час
    отсутствует, осенний сдвоенный час — одна строка), поэтому 24 строки = 24 фактических часа.
    """
    idx = d.y.index
    Y = d.y[stations].to_numpy(dtype="float64")
    C = d.closed[stations].to_numpy()
    rows, ctx, tgt, msk, cidx = [], [], [], [], []
    for o in origins:
        p = idx.get_loc(o)
        if p - context < 0 or p + horizon > len(idx):
            continue
        for j, s in enumerate(stations):
            m = ~C[p:p + horizon, j]
            if not m.any():             # станция закрыта весь горизонт — окно не оцениваем
                continue
            rows.append((s, o))
            ctx.append(Y[p - context:p, j])
            tgt.append(Y[p:p + horizon, j])
            msk.append(m)
            cidx.append(idx[p - context:p])
    meta = pd.DataFrame(rows, columns=["station", "origin"])
    return Windows(meta, np.array(ctx), np.array(tgt), np.array(msk), cidx)


# ---------------------------------------------------------------------------
# Прогнозы-ориентиры и проверка ответов
# ---------------------------------------------------------------------------
def seasonal_naive(ctx: np.ndarray, horizon: int = HORIZON, season: int = SEASON) -> np.ndarray:
    """Тот же час неделю назад."""
    start = ctx.shape[1] - season
    return ctx[:, start:start + horizon].copy()


def check_valid(pred, w: Windows, train_max: pd.Series) -> np.ndarray:
    """Валидность окна по правилам протокола (п. 5). pred — список или массив [N, horizon]."""
    ok = np.zeros(len(w), dtype=bool)
    tmax = train_max.reindex(w.meta["station"]).to_numpy()
    for i, p in enumerate(pred):
        try:
            a = np.asarray(p, dtype="float64")
        except (TypeError, ValueError):
            continue
        if a.shape != (HORIZON,) or not np.isfinite(a).all() or (a < 0).any():
            continue
        if np.isfinite(tmax[i]) and (a > INVALID_MAX_FACTOR * tmax[i]).any():
            continue
        ok[i] = True
    return ok


def apply_fallback(pred, valid: np.ndarray, w: Windows) -> np.ndarray:
    """Невалидные окна заменяются сезонным наивным прогнозом."""
    naive = seasonal_naive(w.ctx)
    out = naive.copy()
    for i, p in enumerate(pred):
        if valid[i]:
            out[i] = np.asarray(p, dtype="float64")
    return out


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def _qloss(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def window_stats(w: Windows, point: np.ndarray, quant: np.ndarray | None = None) -> pd.DataFrame:
    """Суммы по окну, из которых складываются все метрики и бутстрэп.

    quant — [N, horizon, len(QUANTILES)] или None.
    """
    m = w.mask
    y = np.where(m, w.target, 0.0)
    e = np.where(m, np.abs(w.target - point), 0.0)
    se = np.where(m, (w.target - point) ** 2, 0.0)
    scale = np.abs(w.ctx[:, SEASON:] - w.ctx[:, :-SEASON]).mean(axis=1)   # MAE наивного на контексте
    n = m.sum(axis=1)
    df = w.meta.copy()
    df["n"] = n
    df["sum_y"] = y.sum(axis=1)
    df["sum_ae"] = e.sum(axis=1)
    df["sum_se"] = se.sum(axis=1)
    df["mase"] = (df["sum_ae"] / n.clip(min=1)) / np.where(scale > 0, scale, np.nan)
    if quant is not None:
        ql = sum(np.where(m, _qloss(w.target, quant[..., k], t), 0.0).sum(axis=1)
                 for k, t in enumerate(QUANTILES))
        df["sum_ql"] = 2 * ql / len(QUANTILES)
        lo, hi = quant[..., QUANTILES.index(0.1)], quant[..., QUANTILES.index(0.9)]
        df["n_cov80"] = (m & (w.target >= lo) & (w.target <= hi)).sum(axis=1)
    return df


def summarize(ws: pd.DataFrame) -> dict:
    n = ws["n"].sum()
    out = {
        "wape": ws["sum_ae"].sum() / ws["sum_y"].sum(),
        "mae": ws["sum_ae"].sum() / n,
        "rmse": np.sqrt(ws["sum_se"].sum() / n),
        "mase": ws["mase"].mean(),
    }
    if "sum_ql" in ws:
        out["wql"] = ws["sum_ql"].sum() / ws["sum_y"].sum()
        out["cov80"] = ws["n_cov80"].sum() / n
        out["cov80_dev"] = abs(out["cov80"] - 0.8)
    return out


def bootstrap_ci(ws: pd.DataFrame, metric: str = "wape", n_boot: int = 1000, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """95% ДИ: бутстрэп по окнам («станция × точка прогноза»)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ws), size=(n_boot, len(ws)))
    vals = [summarize(ws.iloc[i])[metric] for i in idx]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff_ci(ws_a: pd.DataFrame, ws_b: pd.DataFrame, metric: str = "wape", n_boot: int = 1000,
                   seed: int = SEED) -> tuple[float, float, float]:
    """Разность метрики (a − b) и её 95% ДИ на одних и тех же окнах."""
    key = ["station", "origin"]
    a = ws_a.set_index(key).sort_index()
    b = ws_b.set_index(key).reindex(a.index)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = [summarize(a.iloc[i])[metric] - summarize(b.iloc[i])[metric] for i in idx]
    return summarize(a)[metric] - summarize(b)[metric], float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ---------------------------------------------------------------------------
# Ресурсы и журнал
# ---------------------------------------------------------------------------
class Timer:
    """Время на каждое окно (или батч) — для p50/p95."""

    def __init__(self):
        self.times = []

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self._t)

    def per_window_ms(self, n_windows_per_call: int = 1) -> dict:
        t = np.array(self.times) * 1000 / n_windows_per_call
        return {"ms_p50": float(np.median(t)) if len(t) else np.nan,
                "ms_p95": float(np.quantile(t, 0.95)) if len(t) else np.nan,
                "total_s": float(np.sum(self.times))}


def gpu_peak_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return float("nan")


def results_dir() -> Path:
    p = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else Path("results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_run(model: str, mode: str, w: Windows, point: np.ndarray, valid: np.ndarray,
             quant: np.ndarray | None = None, timing: dict | None = None, extra: dict | None = None) -> dict:
    """Сохраняет прогнозы окна и строку в runs.csv; возвращает сводку метрик."""
    rd = results_dir()
    long = w.meta.loc[w.meta.index.repeat(HORIZON)].reset_index(drop=True)
    long["step"] = np.tile(np.arange(1, HORIZON + 1), len(w))
    long["y"] = w.target.ravel()
    long["yhat"] = point.ravel()
    long["in_metric"] = w.mask.ravel()
    long["valid"] = np.repeat(valid, HORIZON)
    if quant is not None:
        for k, t in enumerate(QUANTILES):
            long[f"q{int(t * 100)}"] = quant[..., k].ravel()
    long.to_parquet(rd / f"{model}__{mode}.parquet", index=False)

    ws = window_stats(w, point, quant)
    ws.to_parquet(rd / f"{model}__{mode}__windows.parquet", index=False)
    s = summarize(ws)
    lo, hi = bootstrap_ci(ws, "wape")
    row = {"date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "model": model, "mode": mode,
           "n_windows": len(w), "invalid_share": float(1 - valid.mean()), **s,
           "wape_ci_lo": lo, "wape_ci_hi": hi, "gpu_peak_gb": gpu_peak_gb(),
           **(timing or {}), **(extra or {})}
    runs = rd / "runs.csv"
    pd.DataFrame([row]).to_csv(runs, mode="a", header=not runs.exists(), index=False)
    return row
