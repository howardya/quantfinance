import numpy as np

__all__ = ['get_sharpe_ratio']

def get_sharpe_ratio(s):
    return (s.mean()*260) / (s.std()*np.sqrt(260))
