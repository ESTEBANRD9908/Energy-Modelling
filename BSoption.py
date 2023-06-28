### Black&Scholes
import numpy as np
from scipy.stats import  norm

def black_scholes(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
    put_price = K * np.exp(-r * T) * (1 - N_d2) - S * (1 - N_d1)
    return call_price, put_price
