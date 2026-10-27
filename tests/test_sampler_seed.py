import unittest
from unittest.mock import patch

from mjrl.samplers import core


class _ImmediateResult(object):
    def get(self, timeout):
        return []


class _RecordingPool(object):
    instances = []

    def __init__(self, processes, maxtasksperchild=None):
        self.calls = []
        self.instances.append(self)

    def apply_async(self, func, kwds):
        self.calls.append(kwds)
        return _ImmediateResult()

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class SamplePathsSeedTest(unittest.TestCase):
    def setUp(self):
        _RecordingPool.instances = []

    @patch("mjrl.samplers.core.mp.Pool", _RecordingPool)
    def test_multiprocess_sampling_accepts_default_none_seed(self):
        paths = core.sample_paths(
            num_traj=1,
            env=object(),
            policy=object(),
            base_seed=None,
            num_cpu=2,
        )

        self.assertEqual(paths, [])
        self.assertEqual(
            [call["base_seed"] for call in _RecordingPool.instances[-1].calls],
            [None, None],
        )

    @patch("mjrl.samplers.core.mp.Pool", _RecordingPool)
    def test_multiprocess_sampling_partitions_explicit_seed(self):
        core.sample_paths(
            num_traj=1,
            env=object(),
            policy=object(),
            base_seed=37,
            num_cpu=2,
        )

        self.assertEqual(
            [call["base_seed"] for call in _RecordingPool.instances[-1].calls],
            [37, 38],
        )


if __name__ == "__main__":
    unittest.main()
