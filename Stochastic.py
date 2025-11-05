import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.integrate import quad
from scipy.stats import norm

@dataclass
class GeometricBrownianMotion():
    """
    A Geometric Brownian Motion stochastic process

    Parameters:
        mu (float): drift of log-returns
        sigma (float): yearly volatility
    """

    mu: float
    sigma: float

    def simulate(self, S0: float, T: float, n: int, m: int) -> np.ndarray: 
        """
        Simulate GBM paths

        Parameters:
            S0 (float): initial stock price
            T (float): time duration
            n (int): number of simulated paths
            m (int): number of discretisation points

        Returns:
            numpy array of simulated GBM path
        """
        
        dt = T / m
        noise = np.random.normal(0, 1, size = (n, m))
        increments_log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * noise
        log_returns = np.cumsum(increments_log_returns, axis = 1)
        
        paths = S0 * np.exp(log_returns)
        paths = np.insert(paths, 0, S0, axis = 1)
        
        return paths 

    def AmericanPutPrice(self, S0: float, K: float, t: float) -> float:

        """
        Computes the Black-Scholes price of an American put option using LSM
    
        Parameters:
            S0 (float): Current asset price
            K (float): Strike price
            t (float): Time to expiration (in years)
    
        Returns:
            float: Put option price
        """
    
        paths = self.simulate(S0=S0, T=t, n=5000, m=252)
        cashflow = np.maximum(K - paths[:, -1], 0)
        dt = t/252
    
        for i in range(paths.shape[1]-2, 0, -1):
        
            # Discounted cashflows from next period
            cashflow = np.exp(-self.mu * dt) * cashflow
            
            # Price of the stock at current period
            x = paths[:, i]
            
            # Exercise value at time t[i]
            exercise = np.maximum(K-x, 0)
            
            # Index for In-the-Money paths
            itm = exercise > 0

            # If we dont have enough in-the-money paths, skip regression
            if np.sum(itm) < 4:  
                continue
            
            # Fit polynomial to estimate Continuation Value at time t[i]
            fitted = np.polynomial.Polynomial.fit(x[itm], cashflow[itm], 3)
            
            # Estimate Continuation Value at time t[i]
            continuation = fitted(x)
            
            # Index where Exercise is Beneficial
            ex_idx = itm & (exercise > continuation)
            
            # Update cashflows with early exercises
            cashflow[ex_idx] = exercise[ex_idx]
        
        price = np.mean(cashflow)*np.exp(-self.mu * dt)   
        return price

    def EuropeanCallPrice(self, S0: float, K: float, t: float) -> float:
        
        """
        Computes the Black-Scholes price of a European call option.
    
        Parameters:
            S0 (float): Current asset price
            K (float): Strike price
            t (float): Time to expiration (in years)
    
        Returns:
            float: Call option price
        """
        
        d1 = (np.log(S0/K) + (self.mu +.5* self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        d2 = d1 - self.sigma*np.sqrt(t)
        
        call_price = S0*norm.cdf(d1) - K*np.exp(-self.mu * t)*norm.cdf(d2)
        return call_price

    def EuropeanPutPrice(self, S0: float, K: float, t: float) -> float:
        
        """
        Computes the Black-Scholes value of a European put option.
        
        Parameters:
            S0: Current asset price
            K: Strike price
            t: Time to expiration (in years)
        
        Returns:
            Put option price
        """
        
        d1 = (np.log(S0/K) + (self.mu + .5*self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        d2 = d1 - self.sigma*np.sqrt(t)
        
        put_price = -S0*norm.cdf(-d1) + K*np.exp(-self.mu*t)*norm.cdf(-d2)
        return put_price

@dataclass
class Heston():
    """
    A Heston Stochastic Process
    
    Parameters:
        mu (float):    Drift of log-returns
        kappa (float): Rate of mean reversion of variance
        theta (float): Long-run variance
        xi (float):    Volatility of volatility
        rho (float):   Correlation between Brownian motions
    """

    mu: float           # drift of log-returns
    kappa: float        # Rate of mean reversion of variance
    theta: float        # Long-run variance
    xi: float           # Volatility of volatility
    rho: float          # Correlation between Brownian motions

    def simulate(self, S0, v0, r, t, n_steps, n_sims=1, return_vol=False):
        """
        Simulate Heston paths

        Parameters:
            S0 (float): initial stock price
            v0 (float): initial velocity
            r (float): risk-free rate
            t (float): time duration
            n_steps (int): number of discretisation points 
            n_sims (int): number of simulated paths
            return_vol (bool): set to True to return the paths and the volatilities

        Returns:
            numpy array of simulated Heston paths (and optionally: volatilities)
        """

        dt = t / n_steps
        N1 = np.random.normal(0, 1, size=(n_sims, n_steps))
        N2 = np.random.normal(0, 1, size=(n_sims, n_steps))
        Y = self.rho * N1 + np.sqrt(1 - self.rho**2) * N2
    
        paths = np.zeros((n_sims, n_steps + 1))
        vols = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        vols[:, 0] = v0
    
        for i in range(0, n_steps):
            vols[:, i + 1] = np.abs(vols[:, i] + self.kappa * (self.theta - vols[:, i]) * dt + self.xi * np.sqrt(np.maximum(vols[:, i], 0) * dt) * N1[:, i])
            paths[:, i + 1] = paths[:, i] * np.exp((self.mu + r - 0.5 * vols[:, i]) * dt + np.sqrt(np.maximum(vols[:, i], 0) * dt) * Y[:, i])
        
        return (paths, vols) if return_vol else paths

    def EuropeanCallPrice(self, S0, K, v0, r, t):
        """
        Price of a European call option under the Heston model
        
        Parameters:
        - S0 (float): Initial stock price
        - K (float): Strike Price
        - v0 (float): Initial variance
        - r (float): Risk-free interest rate
        - t (float): Time-to-expiration (in years)
    
        Returns:
        - call_price (float): Call option price
        """
    
        def integrand(phi, Pnum):
            i = complex(0, 1)
            u = 0.5 if Pnum == 1 else -0.5
            b = self.kappa - self.rho * self.xi if Pnum == 1 else self.kappa
            a = self.kappa * self.theta
            d = np.sqrt((self.rho * self.xi * phi * i - b)**2 - self.xi**2 * (2 * u * phi * i - phi**2))
            g = (b - self.rho * self.xi * phi * i + d) / (b - self.rho * self.xi * phi * i - d)
            
            exp1 = np.exp(i * phi * np.log(S0 / K))
            C = r * phi * i * t + a / self.xi**2 * ((b - self.rho * self.xi * phi * i + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
            D = (b - self.rho * self.xi * phi * i + d) / self.xi**2 * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
            f = exp1 * np.exp(C + D * v0)
            return np.real(f / (phi * i))
    
        P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
        P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]
        call_price = S0 * P1 - K * np.exp(-r * t) * P2
        return call_price

    def EuropeanPutPrice(self, S0: float, K: float, v0: float, r: float, t: float):
        """
        Price of a European put option under the Heston model using put-call parity.
    
        Parameters:
        - S0 (float): Initial stock price
        - K (float): Strike Price
        - v0 (float): Initial variance
        - r (float): Risk-free interest rate
        - t (float): Time-to-expiration (in years)
    
        Returns:
        - put_price (float): Put option price
        """
    
        def integrand(phi, Pnum):
            i = complex(0, 1)
            u = 0.5 if Pnum == 1 else -0.5
            b = self.kappa - self.rho * self.xi if Pnum == 1 else self.kappa
            a = self.kappa * self.theta
            d = np.sqrt((self.rho * self.xi * phi * i - b)**2 - self.xi**2 * (2 * u * phi * i - phi**2))
            g = (b - self.rho * self.xi * phi * i + d) / (b - self.rho * self.xi * phi * i - d)
    
            exp1 = np.exp(i * phi * np.log(S0 / K))
            C = r * phi * i * t + a / self.xi**2 * ((b - self.rho * self.xi * phi * i + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
            D = (b - self.rho * self.xi * phi * i + d) / self.xi**2 * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
            f = exp1 * np.exp(C + D * v0)
            return np.real(f / (phi * i))
    
        P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
        P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]
        
        call_price = S0 * P1 - K * np.exp(-r * t) * P2
        put_price = call_price - S0 + K * np.exp(-r * t)  # via put-call parity
    
        return put_price

    
    # Naive LSM with no regression on the U_i's at each time-step

    def AmericanCallPrice_worse(self, S0: float, K: float, v0: float, r: float, t: float) -> float:
        
        def ExerciseValueCall(x):
            return np.maximum(x - K, 0)

        return self.AmericanPrice_worse(S0, K, v0, r, t, ExerciseValueCall)

    def AmericanPutPrice_worse(self, S0: float, K: float, v0: float, r: float, t: float) -> float:
        
        def ExerciseValuePut(x):
            return np.maximum(K - x, 0)

        return self.AmericanPrice_worse(S0, K, v0, r, t, ExerciseValuePut)

    def AmericanPrice_worse(self, S0: float, K: float, v0: float, r: float, t: float, ExerciseValue) -> float:
    
        heston_paths = self.simulate(S0=S0, v0=v0, r=r, t=t, n_steps=252, n_sims=5_000)
    
        cashflow = ExerciseValue(heston_paths[:, -1])

        dt = t/252
    
        for i in range(heston_paths.shape[1]-2, 0, -1):
        
            # Discounted cashflows from next period
            cashflow = np.exp(-r * dt) * cashflow
            
            # Price of the stock at current period
            x = heston_paths[:, i]
            
            # Exercise value at time t[i]
            exercise = ExerciseValue(x)
            
            # Index for In-the-Money paths
            itm = exercise > 0
    
            if np.sum(itm) < 4:  
                continue
            
            # Fit polynomial to estimate Continuation Value at time t[i]
            fitted = np.polynomial.Polynomial.fit(x[itm], cashflow[itm], 3)
            
            # Estimate Continuation Value at time t[i]
            continuation = fitted(x)
            
            # Index where Exercise is Beneficial
            ex_idx = itm & (exercise > continuation)
            
            # Update cashflows with early exercises
            cashflow[ex_idx] = exercise[ex_idx]
        
        price = np.mean(cashflow)* np.exp(-r * dt)
    
        return price


    # Augmented LSM:

    def AmericanPutPrice(self, S0: float, K: float, v0: float, r: float, t: float) -> float:
        def ExerciseValuePut(x):
            return np.maximum(K - x, 0)

        return self.AmericanPrice(S0, K, v0, r, t, ExerciseValuePut)

    def AmericanCallPrice(self, S0: float, K: float, v0: float, r: float, t: float) -> float:
        def ExerciseValueCall(x):
            return np.maximum(x - K, 0)

        return self.AmericanPrice(S0, K, v0, r, t, ExerciseValueCall)

    def AmericanPrice(self, S0: float, K: float, v0: float, r: float, t: float, ExerciseValue) -> float:
    
        heston_paths, vols = self.simulate(S0=S0, v0=v0, r=r, t=t, n_steps=252, n_sims=5_000, return_vol=True)
        dt = t / 252  
    
        cashflow = ExerciseValue(heston_paths[:, -1])
    
        for i in range(heston_paths.shape[1]-2, 0, -1):
        
            # Discounted cashflows from next period
            cashflow = np.exp(-r * dt) * cashflow
            
            # State variables at current period
            x = heston_paths[:, i]  # Stock price
            u = vols[:, i]          # Volatility
            
            # Exercise value at time t[i]
            exercise = ExerciseValue(x)
            
            # Index for In-the-Money paths
            itm = exercise > 0
    
            if np.sum(itm) < 10:  # Increased minimum for 2D regression
                continue
            
            # Extract ITM paths data
            x_itm = x[itm]
            u_itm = u[itm]
            cashflow_itm = cashflow[itm]
            
            # Create 2D basis functions for regression using polynomial terms: 1, x, u, x², u², x*u
            X = np.column_stack([
                np.ones_like(x_itm),  # constant
                x_itm,               # stock price
                u_itm,               # volatility  
                x_itm**2,            # price squared
                u_itm**2,            # vol squared
                x_itm * u_itm        # interaction term
            ])
            
            # Perform linear regression
            try:
                coefficients = np.linalg.lstsq(X, cashflow_itm, rcond=None)[0]
                
                # Estimate continuation value for ALL ITM paths
                X_full = np.column_stack([
                    np.ones_like(x_itm),
                    x_itm,
                    u_itm, 
                    x_itm**2,
                    u_itm**2,
                    x_itm * u_itm
                ])
                continuation_itm = X_full @ coefficients
                
                # Create continuation array for all paths
                continuation = np.zeros_like(x)
                continuation[itm] = continuation_itm
                
            except np.linalg.LinAlgError:
                # If regression fails, fall back to price-only regression
                fitted = np.polynomial.Polynomial.fit(x_itm, cashflow_itm, 3)
                continuation = fitted(x)
            
            # Index where Exercise is Beneficial
            ex_idx = itm & (exercise > continuation)
            
            # Update cashflows with early exercises
            cashflow[ex_idx] = exercise[ex_idx]
        
        price = np.mean(cashflow) * np.exp(-r * dt)
        return price