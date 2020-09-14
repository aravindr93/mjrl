import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._current_size = 0
        self._buffer_head = 0
        self._buffer_s = None
        self._buffer_a = None
        self._buffer_r = None
        self._buffer_sp = None
        self._buffer_terminated = None
        return

    def clear(self):
        self._buffer_head = 0
        return

    def store(self, s, a, r, sp, terminated):
        n = s.shape[0]
        assert(a.shape[0] == n)
        assert(r.shape[0] == n)
        assert(sp.shape[0] == n)
        assert(terminated.shape[0] == n)

        if (self._buffer_s is None):
            self._build_buffers(s, a, r, sp, terminated)

        store_idx = np.arange(self._buffer_head, self._buffer_head + n)
        store_idx = np.mod(store_idx, self._buffer_size)
        self._buffer_s[store_idx] = s
        self._buffer_a[store_idx] = a
        self._buffer_r[store_idx] = r
        self._buffer_sp[store_idx] = sp
        self._buffer_terminated[store_idx] = terminated

        self._buffer_head = (self._buffer_head + n) % self._buffer_size
        self._current_size = min(self.current_size() + n, self._buffer_size)

        return

    def current_size(self):
        return self._current_size

    def sample(self, n):
        curr_size = self.current_size()
        idx = np.array([], np.int32)
        while (idx.shape[0] < n):
            remainder = n - idx.shape[0]
            num_samples = min(remainder, curr_size)
            curr_idx = np.random.choice(curr_size, num_samples, replace=False)
            idx = np.concatenate([idx, curr_idx], axis=0)

        s = self._buffer_s[idx]
        a = self._buffer_a[idx]
        r = self._buffer_r[idx]
        sp = self._buffer_sp[idx]
        terminated = self._buffer_terminated[idx]
        return s, a, r, sp, terminated

    def _build_buffers(self, s, a, r, sp, terminated):
        self._buffer_s = np.empty([self._buffer_size, s.shape[-1]], dtype=s.dtype)
        self._buffer_a = np.empty([self._buffer_size, a.shape[-1]], dtype=a.dtype)
        self._buffer_r = np.empty([self._buffer_size], dtype=r.dtype)
        self._buffer_sp = np.empty([self._buffer_size, sp.shape[-1]], dtype=sp.dtype)
        self._buffer_terminated = np.empty([self._buffer_size], dtype=terminated.dtype)
        return