### Simulations 
import numpy as np

def simulate_paths(S0, r, sigma, T, n, m):
    """Simulates m paths of n steps for a geometric Brownian motion"""
    dt = T/n
    S = np.zeros((n+1, m))
    S[0] = S0
    for i in range(1, n+1):
        dW = np.random.normal(0, 1, m)
        S[i] = S[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)
    return S

def Monte_Carlo(S0,r,sigma,T,n,m,K):
    
    S = simulate_paths(S0,r,sigma,T,n,m)
    
    final_prices = S[-1,:]

    call_value = np.array(final_prices - K)
    put_value = np.array(K- final_prices)

    call_value[call_value < 0] = 0
    put_value[put_value < 0] = 0

    pv_call = call_value * np.exp(-r * T)
    pv_put = put_value * np.exp(-r * T)

    call = np.mean(pv_call)
    put = np.mean(pv_put)

    return call,put