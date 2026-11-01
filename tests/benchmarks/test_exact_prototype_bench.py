"""`--bench --impl prototype|module` of the standalone exact-LM prototype."""

import numpy as np
import pytest

import benchmarks.prototypes.exact_accumulated_lm as lm


def test_bench_impls_agree_on_cost():
    proto = lm.main(["--bench", "exp_decay", "20000", "--repeats", "1"])
    mod = lm.main(
        ["--bench", "exp_decay", "20000", "--repeats", "1", "--impl", "module"]
    )
    assert (proto["impl"], mod["impl"]) == ("prototype", "module")
    assert np.isfinite(proto["cost"]) and np.isfinite(mod["cost"])
    assert mod["cost"] == pytest.approx(proto["cost"], rel=1e-9)
    assert mod["nfev"] == proto["nfev"]


def test_bench_prints_which_impl_ran(capsys):
    lm.main(["--bench", "exp_decay", "20000", "--repeats", "1", "--impl", "module"])
    assert "impl=module" in capsys.readouterr().out


def test_bench_rejects_unknown_impl(capsys):
    with pytest.raises(SystemExit) as e:
        lm.main(["--bench", "exp_decay", "20000", "--impl", "bogus"])
    assert e.value.code == 2
    assert "bogus" in capsys.readouterr().err
