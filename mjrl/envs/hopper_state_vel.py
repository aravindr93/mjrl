import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperStateVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_current_state=None):
        if use_current_state is None:
            raise ValueError("must set use current state")
        self.use_current_state = use_current_state
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        if not self.use_current_state:
            self.do_simulation(a, self.frame_skip)
        
        height, ang = self.sim.data.qpos[1:3]
        alive_bonus = 1.0
        reward = self.sim.data.qvel[0]
        reward += alive_bonus

        if self.use_current_state:
            self.do_simulation(a, self.frame_skip)

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20