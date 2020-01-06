
import numpy as np

class Baseline:
    def __init__(self):
        pass

    def get_weights(self, obs, actions, times):
        raise NotImplementedError("Baseline is an abstract class. Implement get_weights in a subclass!")


# TODO: make current baselines subclasses of this
class PathBaseline(Baseline):
    def __init__(self):
        pass 

    def fit(self, paths):
        raise NotImplementedError("PathBaseline is an abstract class. Implement fit in a subclass!")

    def predict(self, path):
        raise NotImplementedError("PathBaseline is an abstract class. Implement predict in a subclass!")

    def get_weights(self, obs, actions, times):
        path = dict(
            observations=np.array(obs),
            actions=np.array(actions),
            time=np.array(times)
        )

        return self.predict(path)

class ReplayBufferBaseline(Baseline):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def update(self):
        raise NotImplementedError("ReplayBufferBaseline is an abstract class. Implement update in a subclass!")

