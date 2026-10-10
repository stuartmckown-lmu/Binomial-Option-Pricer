import numpy as np
from math import log, sqrt, exp, erf


def fun(SO, K, T, r, N, sigma, option_type="call", q=0.0, american=False):
    """
    Price an option with a Cox-Ross-Rubinstein binomial tree.

    SO          stock price today
    K           exercise (strike) price
    T           years until exercise date
    r           risk-free rate (continuously compounded)
    N           number of steps in the tree
    sigma       standard deviation of log returns (annual volatility)
    option_type "call" or "put"
    q           continuous dividend yield (0 means no dividends)
    american    True if the option can be exercised early, False for European

    Example: fun(75.0, 65.0, 1, 0.05, 365, 0.30) prices a European call.
    """
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put"')
    if SO <= 0 or K <= 0 or T <= 0 or sigma <= 0 or N < 1:
        raise ValueError("SO, K, T and sigma must be positive and N at least 1")

    def payoff(s):  # value of exercising right now when the stock price is s
        if option_type == "call":
            return max(0, s - K)
        return max(0, K - s)

    DT = T / N                                  # length of one step in years
    U = np.exp(sigma * np.sqrt(DT))             # price change if stock goes up
    D = np.exp(-sigma * np.sqrt(DT))            # price change if stock goes down
    p = (np.exp((r - q) * DT) - D) / (U - D)    # risk-neutral probability of an up move
    disc = np.exp(-r * DT)                      # discount factor for one step

    # p has to be a real probability, otherwise the tree allows arbitrage
    if not 0 < p < 1:
        raise ValueError(f"p = {p:.4f} is not between 0 and 1; use more steps (larger N)")

    # Stock price tree: S[t, j] = price at step t after j up moves
    S = np.zeros((N + 1, N + 1))                # (N+1) by (N+1) matrix of 0's
    for t in range(N + 1):
        for j in range(t + 1):
            S[t, j] = SO * (U ** j) * (D ** (t - j))

    # Option value tree: start with the payoff at expiration (row N)
    C = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        C[N, j] = payoff(S[N, j])

    # Work backward through rows N-1, N-2, ..., 0
    for t in range(N - 1, -1, -1):
        for j in range(t + 1):                  # every node in that row
            hold = disc * (p * C[t + 1, j + 1] + (1 - p) * C[t + 1, j])  # value of waiting
            if american:
                C[t, j] = max(hold, payoff(S[t, j]))  # exercise early if that's worth more
            else:
                C[t, j] = hold

    return C[0, 0]


def black_scholes(SO, K, T, r, sigma, option_type="call", q=0.0):
    """Closed-form European price, used to check the tree."""
    def Ncdf(x):  # standard normal CDF
        return 0.5 * (1 + erf(x / sqrt(2)))

    d1 = (log(SO / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return SO * exp(-q * T) * Ncdf(d1) - K * exp(-r * T) * Ncdf(d2)
    return K * exp(-r * T) * Ncdf(-d2) - SO * exp(-q * T) * Ncdf(-d1)


if __name__ == "__main__":
    SO = 75.0     # stock price today
    K = 65.0      # exercise price
    T = 1         # years until exercise date
    r = 0.05      # risk-free rate
    sigma = 0.30  # SD of log returns
    N = 365       # number of steps

    call = fun(SO, K, T, r, N, sigma, "call")
    put = fun(SO, K, T, r, N, sigma, "put")
    am_call = fun(SO, K, T, r, N, sigma, "call", american=True)
    am_put = fun(SO, K, T, r, N, sigma, "put", american=True)

    print("Checks against Black-Scholes")
    print(f"  European call: tree {call:.4f}   Black-Scholes {black_scholes(SO, K, T, r, sigma, 'call'):.4f}")
    print(f"  European put:  tree {put:.4f}   Black-Scholes {black_scholes(SO, K, T, r, sigma, 'put'):.4f}")
    print(f"  Put-call parity gap (should be ~0): {call - put - (SO - K * np.exp(-r * T)):.1e}")
    print()
    print("American options")
    print(f"  American put:  {am_put:.4f}  (early exercise premium {am_put - put:.4f})")
    print(f"  American call: {am_call:.4f}  (same as European when there are no dividends)")
