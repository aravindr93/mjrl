import numpy as np
import copy

class LinearBaseline:
    def __init__(self, env_spec, reg_coeff=1e-5):
        n = env_spec.observation_dim       # number of states
        self._reg_coeff = reg_coeff
        self._coeffs = None

    def _features(self, path):
        # compute regression features for the path
        o = np.clip(path["observations"], -10, 10)
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 1000.0
        feat = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)
        return feat

    def fit(self, paths, return_errors=False):

        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if return_errors:
            predictions = featmat.dot(self._coeffs) if self._coeffs is not None else np.zeros(returns.shape)
            errors = returns - predictions
            error_before = np.sum(errors**2)/np.sum(returns**2)

        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = returns - predictions
            error_after = np.sum(errors**2)/np.sum(returns**2)
            return error_before, error_after

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)
