"""
Options Greeks Calculator with Numba Optimization

High-performance options Greeks calculations using Black-Scholes model
and other advanced options pricing models, optimized with Numba JIT.
"""

import numpy as np
import numba
from numba import jit
import math
from typing import Tuple, Dict, Optional
from datetime import datetime, date
import sys
import os

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger

logger = get_logger(__name__)

# Numba optimization settings
NUMBA_CONFIG = {
    'nopython': True,
    'nogil': True,
    'cache': True,
    'fastmath': True,
    'boundscheck': False,
    'wraparound': False,
}

class OptionsGreeksCalculator:
    """High-performance options Greeks calculator using Numba"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def norm_pdf(x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        return d1, d2
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes call option price"""
        
        if T <= 0:
            return max(S - K, 0.0)
        
        if sigma <= 0:
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        call_price = (S * math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(d1) - 
                     K * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(d2))
        
        return max(call_price, 0.0)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes put option price"""
        
        if T <= 0:
            return max(K - S, 0.0)
        
        if sigma <= 0:
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        put_price = (K * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(-d2) - 
                    S * math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(-d1))
        
        return max(put_price, 0.0)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def delta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call option delta"""
        
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        if sigma <= 0:
            return 1.0 if S > K else 0.0
        
        d1, _ = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        return math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(d1)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def delta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put option delta"""
        
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        if sigma <= 0:
            return -1.0 if S < K else 0.0
        
        d1, _ = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        return -math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(-d1)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Option gamma (same for calls and puts)"""
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, _ = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        gamma_val = (math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1)) / (S * sigma * math.sqrt(T))
        
        return gamma_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def theta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call option theta (time decay)"""
        
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return 0.0
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        term1 = -(S * math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        term2 = -r * K * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(d2)
        term3 = q * S * math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(d1)
        
        theta_val = (term1 + term2 + term3) / 365.0  # Per day
        
        return theta_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def theta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put option theta (time decay)"""
        
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return 0.0
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        term1 = -(S * math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        term2 = r * K * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(-d2)
        term3 = -q * S * math.exp(-q * T) * OptionsGreeksCalculator.norm_cdf(-d1)
        
        theta_val = (term1 + term2 + term3) / 365.0  # Per day
        
        return theta_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Option vega (volatility sensitivity)"""
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, _ = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        vega_val = S * math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1) * math.sqrt(T) / 100.0  # Per 1% vol change
        
        return vega_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def rho_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call option rho (interest rate sensitivity)"""
        
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return K * T * math.exp(-r * T) if S > K else 0.0
        
        _, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        rho_val = K * T * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(d2) / 100.0  # Per 1% rate change
        
        return rho_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def rho_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put option rho (interest rate sensitivity)"""
        
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return -K * T * math.exp(-r * T) if S < K else 0.0
        
        _, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        rho_val = -K * T * math.exp(-r * T) * OptionsGreeksCalculator.norm_cdf(-d2) / 100.0  # Per 1% rate change
        
        return rho_val
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def lambda_greek(option_price: float, delta: float, S: float) -> float:
        """Lambda (leverage) - percentage change in option price per percentage change in underlying"""
        
        if option_price <= 0:
            return 0.0
        
        return (delta * S) / option_price
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def intrinsic_value(S: float, K: float, is_call: bool) -> float:
        """Calculate intrinsic value"""
        
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def time_value(option_price: float, intrinsic_val: float) -> float:
        """Calculate time value"""
        return max(option_price - intrinsic_val, 0.0)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def moneyness(S: float, K: float, is_call: bool) -> float:
        """Calculate moneyness ratio"""
        
        if K <= 0:
            return 1.0
        
        if is_call:
            return S / K
        else:
            return K / S
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def probability_itm(S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Probability of finishing in-the-money"""
        
        if T <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return 1.0 if S < K else 0.0
        
        if sigma <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return 1.0 if S < K else 0.0
        
        # Use risk-neutral probability
        d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if is_call:
            return OptionsGreeksCalculator.norm_cdf(d2)
        else:
            return OptionsGreeksCalculator.norm_cdf(-d2)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def calculate_all_greeks(
        S: float, K: float, T: float, r: float, sigma: float, 
        is_call: bool, q: float = 0.0
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """
        Calculate all Greeks and metrics in one optimized call
        
        Returns:
            (option_price, delta, gamma, theta, vega, rho, lambda_greek, 
             intrinsic_value, time_value, moneyness, probability_itm)
        """
        
        # Calculate option price
        if is_call:
            option_price = OptionsGreeksCalculator.black_scholes_call(S, K, T, r, sigma, q)
            delta = OptionsGreeksCalculator.delta_call(S, K, T, r, sigma, q)
            theta = OptionsGreeksCalculator.theta_call(S, K, T, r, sigma, q)
            rho = OptionsGreeksCalculator.rho_call(S, K, T, r, sigma, q)
        else:
            option_price = OptionsGreeksCalculator.black_scholes_put(S, K, T, r, sigma, q)
            delta = OptionsGreeksCalculator.delta_put(S, K, T, r, sigma, q)
            theta = OptionsGreeksCalculator.theta_put(S, K, T, r, sigma, q)
            rho = OptionsGreeksCalculator.rho_put(S, K, T, r, sigma, q)
        
        # Calculate Greeks that are same for calls and puts
        gamma = OptionsGreeksCalculator.gamma(S, K, T, r, sigma, q)
        vega = OptionsGreeksCalculator.vega(S, K, T, r, sigma, q)
        
        # Calculate derived metrics
        lambda_val = OptionsGreeksCalculator.lambda_greek(option_price, delta, S)
        intrinsic_val = OptionsGreeksCalculator.intrinsic_value(S, K, is_call)
        time_val = OptionsGreeksCalculator.time_value(option_price, intrinsic_val)
        moneyness_val = OptionsGreeksCalculator.moneyness(S, K, is_call)
        prob_itm = OptionsGreeksCalculator.probability_itm(S, K, T, r, sigma, is_call, q)
        
        return (option_price, delta, gamma, theta, vega, rho, lambda_val,
                intrinsic_val, time_val, moneyness_val, prob_itm)

class ImpliedVolatilityCalculator:
    """Calculate implied volatility using numerical methods"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def vega_for_iv(S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Vega calculation for IV solver"""
        return OptionsGreeksCalculator.vega(S, K, T, r, sigma, q)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def option_price_for_iv(S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Option price calculation for IV solver"""
        if is_call:
            return OptionsGreeksCalculator.black_scholes_call(S, K, T, r, sigma, q)
        else:
            return OptionsGreeksCalculator.black_scholes_put(S, K, T, r, sigma, q)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def implied_volatility_newton_raphson(
        market_price: float, S: float, K: float, T: float, r: float, 
        is_call: bool, q: float = 0.0, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        
        if T <= 0 or market_price <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.2  # 20% volatility as starting point
        
        for i in range(max_iterations):
            # Calculate option price and vega
            calculated_price = ImpliedVolatilityCalculator.option_price_for_iv(S, K, T, r, sigma, is_call, q)
            vega_val = ImpliedVolatilityCalculator.vega_for_iv(S, K, T, r, sigma, is_call, q)
            
            # Price difference
            price_diff = calculated_price - market_price
            
            # Check convergence
            if abs(price_diff) < tolerance:
                return sigma
            
            # Check if vega is too small (avoid division by zero)
            if abs(vega_val) < 1e-10:
                break
            
            # Newton-Raphson update
            sigma_new = sigma - price_diff / (vega_val * 100.0)  # vega is per 1% change
            
            # Ensure sigma stays positive and reasonable
            sigma_new = max(0.001, min(sigma_new, 5.0))  # Between 0.1% and 500%
            
            # Check for convergence in sigma
            if abs(sigma_new - sigma) < tolerance:
                return sigma_new
            
            sigma = sigma_new
        
        return sigma  # Return last calculated value if not converged
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def implied_volatility_bisection(
        market_price: float, S: float, K: float, T: float, r: float,
        is_call: bool, q: float = 0.0, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> float:
        """
        Calculate implied volatility using bisection method (more robust)
        """
        
        if T <= 0 or market_price <= 0:
            return 0.0
        
        # Set bounds
        sigma_low = 0.001   # 0.1%
        sigma_high = 5.0    # 500%
        
        # Check if market price is within possible bounds
        price_low = ImpliedVolatilityCalculator.option_price_for_iv(S, K, T, r, sigma_low, is_call, q)
        price_high = ImpliedVolatilityCalculator.option_price_for_iv(S, K, T, r, sigma_high, is_call, q)
        
        if market_price < price_low or market_price > price_high:
            # Market price is outside theoretical bounds
            if market_price < price_low:
                return sigma_low
            else:
                return sigma_high
        
        # Bisection method
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2.0
            price_mid = ImpliedVolatilityCalculator.option_price_for_iv(S, K, T, r, sigma_mid, is_call, q)
            
            if abs(price_mid - market_price) < tolerance:
                return sigma_mid
            
            if price_mid < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            
            if abs(sigma_high - sigma_low) < tolerance:
                return (sigma_low + sigma_high) / 2.0
        
        return (sigma_low + sigma_high) / 2.0

class AdvancedGreeks:
    """Advanced Greeks calculations (second and third order)"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def charm(S: float, K: float, T: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
        """Charm - rate of change of delta with respect to time"""
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        if is_call:
            charm_val = -math.exp(-q * T) * (
                OptionsGreeksCalculator.norm_pdf(d1) * (2.0 * (r - q) * T - d2 * sigma * math.sqrt(T)) / 
                (2.0 * T * sigma * math.sqrt(T)) + 
                q * OptionsGreeksCalculator.norm_cdf(d1)
            )
        else:
            charm_val = -math.exp(-q * T) * (
                OptionsGreeksCalculator.norm_pdf(d1) * (2.0 * (r - q) * T - d2 * sigma * math.sqrt(T)) / 
                (2.0 * T * sigma * math.sqrt(T)) - 
                q * OptionsGreeksCalculator.norm_cdf(-d1)
            )
        
        return charm_val / 365.0  # Per day
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Vanna - rate of change of delta with respect to volatility"""
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        vanna_val = -math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1) * d2 / sigma
        
        return vanna_val / 100.0  # Per 1% vol change
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def volga(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Volga (Vomma) - rate of change of vega with respect to volatility"""
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, d2 = OptionsGreeksCalculator.d1_d2(S, K, T, r, sigma)
        
        volga_val = (S * math.exp(-q * T) * OptionsGreeksCalculator.norm_pdf(d1) * 
                    math.sqrt(T) * d1 * d2 / sigma)
        
        return volga_val / 10000.0  # Per 1% vol change squared

class OptionsAnalytics:
    """Advanced options analytics and risk metrics"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def break_even_call(K: float, premium: float) -> float:
        """Break-even price for call option"""
        return K + premium
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def break_even_put(K: float, premium: float) -> float:
        """Break-even price for put option"""
        return K - premium
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def max_profit_call(premium: float) -> float:
        """Maximum profit for call option (unlimited)"""
        return float('inf')  # Unlimited profit potential
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def max_profit_put(K: float, premium: float) -> float:
        """Maximum profit for put option"""
        return K - premium
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def max_loss_long_option(premium: float) -> float:
        """Maximum loss for long option position"""
        return premium
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def probability_of_profit(S: float, K: float, T: float, r: float, sigma: float, 
                            premium: float, is_call: bool, q: float = 0.0) -> float:
        """Probability of profit at expiration"""
        
        if T <= 0:
            if is_call:
                return 1.0 if S > K + premium else 0.0
            else:
                return 1.0 if S < K - premium else 0.0
        
        if sigma <= 0:
            if is_call:
                return 1.0 if S > K + premium else 0.0
            else:
                return 1.0 if S < K - premium else 0.0
        
        # Calculate break-even point
        if is_call:
            break_even = K + premium
            # Probability that S > break_even
            d = (math.log(S / break_even) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            return OptionsGreeksCalculator.norm_cdf(d)
        else:
            break_even = K - premium
            # Probability that S < break_even
            d = (math.log(S / break_even) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            return OptionsGreeksCalculator.norm_cdf(-d)
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def iv_rank(current_iv: float, iv_history: np.ndarray) -> float:
        """Calculate IV rank (0-100 scale)"""
        
        if len(iv_history) == 0:
            return 50.0
        
        min_iv = np.min(iv_history)
        max_iv = np.max(iv_history)
        
        if max_iv == min_iv:
            return 50.0
        
        rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100.0
        
        return max(0.0, min(100.0, rank))
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def iv_percentile(current_iv: float, iv_history: np.ndarray) -> float:
        """Calculate IV percentile"""
        
        if len(iv_history) == 0:
            return 50.0
        
        count_below = 0
        for iv in iv_history:
            if iv < current_iv:
                count_below += 1
        
        percentile = (count_below / len(iv_history)) * 100.0
        
        return percentile

# Batch Greeks calculation for multiple options
@jit(**NUMBA_CONFIG)
def calculate_options_chain_greeks(
    spot_prices: np.ndarray,
    strikes: np.ndarray,
    time_to_expiry: np.ndarray,
    risk_free_rates: np.ndarray,
    implied_vols: np.ndarray,
    option_types: np.ndarray,  # 1 for calls, 0 for puts
    dividend_yields: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Greeks for entire options chain in batch
    
    Returns:
        Tuple of (deltas, gammas, thetas, vegas, rhos, lambdas)
    """
    
    n = len(strikes)
    deltas = np.empty(n)
    gammas = np.empty(n)
    thetas = np.empty(n)
    vegas = np.empty(n)
    rhos = np.empty(n)
    lambdas = np.empty(n)
    
    for i in range(n):
        S = spot_prices[i] if len(spot_prices) > 1 else spot_prices[0]
        K = strikes[i]
        T = time_to_expiry[i] if len(time_to_expiry) > 1 else time_to_expiry[0]
        r = risk_free_rates[i] if len(risk_free_rates) > 1 else risk_free_rates[0]
        sigma = implied_vols[i]
        is_call = option_types[i] == 1
        q = dividend_yields[i] if len(dividend_yields) > 1 else dividend_yields[0]
        
        # Calculate all Greeks
        (option_price, delta, gamma, theta, vega, rho, lambda_val,
         intrinsic_val, time_val, moneyness_val, prob_itm) = OptionsGreeksCalculator.calculate_all_greeks(
            S, K, T, r, sigma, is_call, q
        )
        
        deltas[i] = delta
        gammas[i] = gamma
        thetas[i] = theta
        vegas[i] = vega
        rhos[i] = rho
        lambdas[i] = lambda_val
    
    return deltas, gammas, thetas, vegas, rhos, lambdas
