import numpy as np
import math
import matplotlib.pyplot as plt

def monte_carlo_call_price(S0=100, K=100, r=0.02, sigma=0.2, T=1.0, n_sims=100_000):
    """Monte Carlo price of a European call option."""
    z = np.random.randn(n_sims)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    payoffs = np.maximum(S_T - K, 0)
    price_mc = math.exp(-r * T) * payoffs.mean()
    return price_mc

def bs_call_price(S0=100, K=100, r=0.02, sigma=0.2, T=1.0):
    """Black–Scholes price of a European call option (closed form)."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # standard normal CDF
    return S0 * phi(d1) - K * math.exp(-r * T) * phi(d2)

def simulate_paths(S0=100, r=0.02, sigma=0.2, T=1.0, n_steps=252, n_paths=20):
    """Simulate GBM stock-price paths for visualisation."""
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
    return paths

if __name__ == "__main__":
    # Parameters
    S0 = 100   # initial stock price
    K = 100    # strike
    r = 0.02   # risk-free rate
    sigma = 0.2  # volatility
    T = 1.0    # time to maturity in years

    # Monte Carlo vs Black–Scholes
    price_mc = monte_carlo_call_price(S0, K, r, sigma, T, n_sims=100_000)
    price_bs = bs_call_price(S0, K, r, sigma, T)

    print(f"Monte Carlo call price: {price_mc:.4f}")
    print(f"Black–Scholes call price: {price_bs:.4f}")

    # Plot a few simulated price paths
    paths = simulate_paths(S0, r, sigma, T, n_steps=252, n_paths=10)

    plt.figure(figsize=(8, 4))
    for i in range(paths.shape[0]):
        plt.plot(paths[i], alpha=0.7)
    plt.title("Simulated stock-price paths (GBM)")
    plt.xlabel("Time step");
    plt.ylabel("Price");
    plt.tight_layout()
    plt.show()
