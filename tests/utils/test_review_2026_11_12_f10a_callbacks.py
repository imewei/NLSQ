"""functools.wraps decorators around callbacks must keep receiving positional args."""

import functools

import numpy as np

from nlsq.callbacks import call_user_callback


def cb(iteration, cost, params, info):
    return (iteration, cost, params, info)


def every(n):
    def deco(f):
        @functools.wraps(f)
        def w(*args, **kwargs):
            if args[0] % n == 0:  # reads the iteration positionally
                return f(*args, **kwargs)

        return w

    return deco


def forward_args_only(f):
    @functools.wraps(f)
    def w(*args):
        return f(*args)

    return w


def forward_args_only_kw(f):
    # (*args, **kwargs) wrapper that only forwards *args
    @functools.wraps(f)
    def w(*args, **kwargs):
        return f(*args)

    return w


P = np.array([1.0])


def test_decorator_reading_args0_works():
    assert call_user_callback(every(2)(cb), 4, 1.5, P, {}) == (4, 1.5, P, {})
    assert call_user_callback(every(2)(cb), 3, 1.5, P, {}) is None


def test_decorator_forwarding_only_args_works():
    assert call_user_callback(forward_args_only_kw(cb), 4, 1.5, P, {}) == (
        4,
        1.5,
        P,
        {},
    )
    assert call_user_callback(forward_args_only(cb), 4, 1.5, P, {}) == (4, 1.5, P, {})


def test_stacked_decorators_on_canonical_names():
    wrapped = every(2)(forward_args_only_kw(cb))
    assert call_user_callback(wrapped, 2, 0.5, P, {"k": 1}) == (2, 0.5, P, {"k": 1})


def test_wrapper_over_keyword_only_callback_still_uses_keywords():
    def kwonly(*, iteration, cost, params, info):
        return (iteration, cost)

    @functools.wraps(kwonly)
    def w(*args, **kwargs):
        return kwonly(*args, **kwargs)

    assert call_user_callback(w, 1, 2.0, P, {}) == (1, 2.0)
