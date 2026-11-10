import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization import GlobalOptimizationConfig, TournamentSelector
from nlsq.utils.safe_serialize import safe_dumps, safe_loads


def _f(x, a, b):
    return a * jnp.exp(-b * x)


def test_resume_matches_uninterrupted_run():
    rng = np.random.default_rng(0)
    cands = rng.uniform([0.5, 0.1], [4, 2], size=(16, 2))
    cfg = GlobalOptimizationConfig(
        n_starts=16, elimination_rounds=3, elimination_fraction=0.5, batches_per_round=2
    )

    def batches(k):
        for _ in range(k):
            x = rng.uniform(0, 4, 200)
            yield x, 2 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(200)

    full = TournamentSelector(cands, cfg)
    full.run_tournament(batches(100), _f, top_m=1)

    part = TournamentSelector(cands, cfg)
    part.run_tournament(batches(2), _f, top_m=1)  # data runs out in round 1
    ck = safe_loads(safe_dumps(part.to_checkpoint()))
    res = TournamentSelector.from_checkpoint(ck, cfg)
    res.run_tournament(batches(100), _f, top_m=1)

    assert res.current_round == full.current_round
    assert res.n_survivors == full.n_survivors
    rounds = [h["round"] for h in res.round_history]
    assert rounds == sorted(set(rounds))
