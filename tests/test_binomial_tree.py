import math

import pytest

from binomial_tree import fun, black_scholes

# The example option used throughout the project
SO, K, T, r, sigma = 75.0, 65.0, 1, 0.05, 0.30
N = 365


def test_matches_original_call_price():
    # The European call should be unchanged from the original version of the code
    assert fun(SO, K, T, r, N, sigma) == pytest.approx(16.355043, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_european_close_to_black_scholes(option_type, q):
    tree = fun(SO, K, T, r, N, sigma, option_type, q)
    bs = black_scholes(SO, K, T, r, sigma, option_type, q)
    assert abs(tree - bs) < 0.01


@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity_holds_exactly(q):
    call = fun(SO, K, T, r, N, sigma, "call", q)
    put = fun(SO, K, T, r, N, sigma, "put", q)
    parity = SO * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - put == pytest.approx(parity, abs=1e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_worth_at_least_european(option_type):
    european = fun(SO, K, T, r, N, sigma, option_type)
    american = fun(SO, K, T, r, N, sigma, option_type, american=True)
    assert american >= european - 1e-12


def test_american_call_equals_european_without_dividends():
    european = fun(SO, K, T, r, N, sigma, "call")
    american = fun(SO, K, T, r, N, sigma, "call", american=True)
    assert american == pytest.approx(european, abs=1e-12)


def test_american_put_has_early_exercise_premium():
    european = fun(SO, K, T, r, N, sigma, "put")
    american = fun(SO, K, T, r, N, sigma, "put", american=True)
    assert american - european > 0.1


def test_american_bounds_without_dividends():
    # S0 - K <= C_A - P_A <= S0 - K e^{-rT}
    call = fun(SO, K, T, r, N, sigma, "call", american=True)
    put = fun(SO, K, T, r, N, sigma, "put", american=True)
    assert SO - K <= call - put <= SO - K * math.exp(-r * T)


def test_converges_toward_black_scholes():
    bs = black_scholes(SO, K, T, r, sigma, "call")
    assert abs(fun(SO, K, T, r, 400, sigma) - bs) < abs(fun(SO, K, T, r, 10, sigma) - bs)


def test_rejects_arbitrage_tree():
    # sigma = 1% with one step gives p > 1
    with pytest.raises(ValueError):
        fun(75, 65, 1, 0.05, 1, 0.01)


def test_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        fun(SO, K, T, r, N, sigma, option_type="straddle")
