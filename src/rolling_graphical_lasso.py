import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler

class RollingMarketStructure:
    def __init__(self, window=60, step=5, alpha=0.01):
        self.window = window
        self.step = step
        self.alpha = alpha
        
        self.precision_matrices = []
        self.covariance_matrices = []
        self.dates = []

    def compute_returns(self, prices: pd.DataFrame):
        return np.log(prices / prices.shift(1)).dropna()

    def fit(self, prices: pd.DataFrame):
        returns = self.compute_returns(prices)
        dates = returns.index

        for start in range(0, len(returns) - self.window, self.step):
            end = start + self.window
            
            window_data = returns.iloc[start:end]

            # Standardise
            scaler = StandardScaler()
            X = scaler.fit_transform(window_data)

            # Graphical Lasso
            model = GraphicalLasso(alpha=self.alpha)
            model.fit(X)

            self.precision_matrices.append(model.precision_)
            self.covariance_matrices.append(model.covariance_)
            self.dates.append(dates[end])

        return self