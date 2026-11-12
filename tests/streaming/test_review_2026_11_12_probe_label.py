"""check_model_contract names the underlying error instead of hiding a model bug."""

import numpy as np
import pytest

from nlsq.streaming.exact_lm import check_model_contract


def test_model_bug_is_labelled_with_its_exception_type():
    def wrong_arity(x, a):  # called with two parameters
        return a * x

    with pytest.raises(ValueError, match=r"^model is not pointwise") as e:
        check_model_contract(wrong_arity, [1.0, 2.0], np.linspace(0, 1, 8))
    assert "TypeError" in str(e.value)
    assert "fix it first" in str(e.value)
