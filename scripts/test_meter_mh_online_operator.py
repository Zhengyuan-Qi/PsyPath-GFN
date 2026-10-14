#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.meter_mh_online_operator import (
    BoundedSequentialIncrement,
    FrozenLinearPipeline,
    FrozenStandardPCA,
    OnlineSequentialPsychometricModel,
    summarize,
)


def test_standard_pca_equivalence() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 17))
    held = rng.normal(size=(13, 17))
    scaler = StandardScaler().fit(train)
    reducer = PCA(n_components=9, whiten=True, random_state=7).fit(scaler.transform(train))
    expected = reducer.transform(scaler.transform(held))
    actual = FrozenStandardPCA(scaler, reducer)(torch.from_numpy(held)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_linear_pipeline_equivalence() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(90, 19))
    labels = (rng.random(90) > 0.55).astype(int)
    scores = rng.normal(7.0, 2.0, size=90)
    held = rng.normal(size=(11, 19))
    scaler = StandardScaler().fit(train)
    reducer = PCA(n_components=10, whiten=True, random_state=17).fit(scaler.transform(train))
    reduced = reducer.transform(scaler.transform(train))

    classifier = LogisticRegression(solver="liblinear", random_state=17).fit(reduced, labels)
    torch_classifier = FrozenLinearPipeline(scaler, reducer, classifier, logistic=True)
    np.testing.assert_allclose(
        torch_classifier(torch.from_numpy(held)).detach().numpy(),
        classifier.predict_proba(reducer.transform(scaler.transform(held)))[:, 1],
        atol=1e-10,
        rtol=1e-10,
    )

    regressor = Ridge(alpha=10.0).fit(reduced, scores)
    torch_regressor = FrozenLinearPipeline(scaler, reducer, regressor, logistic=False)
    np.testing.assert_allclose(
        torch_regressor(torch.from_numpy(held)).detach().numpy(),
        regressor.predict(reducer.transform(scaler.transform(held))),
        atol=1e-10,
        rtol=1e-10,
    )


def test_summary_equivalence() -> None:
    rng = np.random.default_rng(27)
    value = rng.normal(size=(23, 8)).astype(np.float64)
    position = np.linspace(-1.0, 1.0, len(value), dtype=np.float64)
    delta = np.diff(value, axis=0)
    expected = np.concatenate(
        (
            value.mean(axis=0),
            value.std(axis=0),
            np.quantile(value, 0.1, axis=0),
            np.quantile(value, 0.9, axis=0),
            (position[:, None] * value).sum(axis=0) / np.square(position).sum(),
            value[-1] - value[0],
            np.mean(np.abs(delta), axis=0),
            np.sqrt(np.mean(np.square(delta), axis=0)),
        )
    )
    actual = summarize(torch.from_numpy(value)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_answer_dynamic_controls() -> None:
    torch.manual_seed(37)
    event_dim = 5
    increment = BoundedSequentialIncrement(event_dim=event_dim, hidden_dim=7).to(dtype=torch.float64)
    projected = {
        name: torch.randn(9, event_dim, dtype=torch.float64)
        for name in ("clinical", "answer", "context", "visual")
    }
    default = increment(projected)
    explicit = increment(projected, answer_dynamic=True)
    for key in ("risk_delta", "severity_delta", "attention", "expected_answer", "answer_deviation"):
        torch.testing.assert_close(default[key], explicit[key], atol=0.0, rtol=0.0)

    removed = increment(projected, answer_dynamic=False)
    torch.testing.assert_close(removed["answer_deviation"], torch.zeros_like(projected["answer"]))
    expected_mean = projected["answer"].mean(dim=0, keepdim=True).expand_as(projected["answer"])
    torch.testing.assert_close(removed["answer_target"], expected_mean)
    torch.testing.assert_close(
        removed["answer_target"].std(dim=0),
        torch.zeros(event_dim, dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )

    order = torch.tensor((8, 0, 7, 1, 6, 2, 5, 3, 4), dtype=torch.long)
    shuffled = increment(projected, order=order)
    torch.testing.assert_close(shuffled["answer_target"], projected["answer"][order])
    torch.testing.assert_close(
        shuffled["answer_deviation"],
        projected["answer"][order] - shuffled["expected_answer"],
    )


def test_counterfactual_keep_masks() -> None:
    selected = torch.tensor((1, 4), dtype=torch.long)
    projected = {"answer": torch.ones(6, 3)}
    keep, random_indices = OnlineSequentialPsychometricModel._counterfactual_keep(
        projected, selected, "selected_delete", 11
    )
    assert random_indices is None
    assert keep.tolist() == [True, False, True, True, False, True]

    keep, random_indices = OnlineSequentialPsychometricModel._counterfactual_keep(
        projected, selected, "selected_only", 11
    )
    assert random_indices is None
    assert keep.tolist() == [False, True, False, False, True, False]

    keep_a, random_a = OnlineSequentialPsychometricModel._counterfactual_keep(
        projected, selected, "random_delete", 23
    )
    keep_b, random_b = OnlineSequentialPsychometricModel._counterfactual_keep(
        projected, selected, "random_delete", 23
    )
    torch.testing.assert_close(random_a, random_b)
    torch.testing.assert_close(keep_a, keep_b)
    assert int((~keep_a).sum()) == len(selected)


if __name__ == "__main__":
    test_standard_pca_equivalence()
    test_linear_pipeline_equivalence()
    test_summary_equivalence()
    test_answer_dynamic_controls()
    test_counterfactual_keep_masks()
