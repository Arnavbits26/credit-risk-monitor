import numpy as np
import pandas as pd
import pytest

from src import data, model, monitoring as mon


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=2000)
    assert mon.psi(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert mon.psi(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)) > mon.PSI_RED


def test_psi_categorical_detects_shift():
    a = ["x"] * 80 + ["y"] * 20
    b = ["x"] * 30 + ["y"] * 70
    assert mon.psi_categorical(a, b) > mon.PSI_RED


def test_hhi_bounds_normalised():
    assert mon.hhi([1, 1, 1, 1]) == pytest.approx(0)
    assert mon.hhi([100, 0, 0, 0]) == pytest.approx(1)


def test_hhi_raw_even_split():
    assert mon.hhi([1, 1, 1, 1], normalise=False) == pytest.approx(0.25)


def test_rag_thresholds():
    assert mon.rag(0.05, 0.1, 0.25) == "GREEN"
    assert mon.rag(0.10, 0.1, 0.25) == "AMBER"
    assert mon.rag(0.30, 0.1, 0.25) == "RED"


def test_exceptions_need_materiality_and_risk():
    conc = pd.DataFrame({
        "seg": ["big_risky", "small_risky", "big_safe"],
        "accounts": [10, 10, 10], "exposure": [1, 1, 1],
        "bad_rate": [0.5, 0.5, 0.1],
        "exposure_share": [0.30, 0.01, 0.30],
        "bad_rate_ratio": [1.6, 1.6, 0.5],
    })
    out = mon.exceptions(conc, "seg")
    assert list(out["segment"]) == ["big_risky"]
    assert out["status"].iloc[0] == "RED"


def test_concentration_shares_sum_to_one():
    df = pd.DataFrame({"s": list("aabbc"), "amount": [10, 20, 30, 40, 50], "default": [0, 1, 0, 1, 1]})
    c = mon.concentration(df, "s")
    assert c["exposure_share"].sum() == pytest.approx(1)
    assert c.set_index("s").loc["c", "bad_rate"] == 1


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert model.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1)
    assert model.ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0)


def test_protected_fields_not_in_model():
    df = data.load()
    cols = data.model_columns(df)
    for c in data.PROTECTED + data.MONITOR_ONLY + ["default"]:
        assert c not in cols


def test_split_is_stratified_and_deterministic():
    df = data.load()
    a, b = data.split(df)
    a2, _ = data.split(df)
    assert a.index.equals(a2.index)
    assert abs(a["default"].mean() - b["default"].mean()) < 0.01
