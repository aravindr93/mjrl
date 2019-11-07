import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os


class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        # trajopt specific attributes
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/sawyer.xml', 2)
        utils.EzPickle.__init__(self)
        self.observation_dim = 26
        self.action_dim = 7

        self.hand_sid = self.model.site_name2id("finger")
        self.target_sid = self.model.site_name2id("target")

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.data.site_xpos[self.target_sid]
        l1_dist = np.sum(np.abs(hand_pos - target_pos))
        l2_dist = np.linalg.norm(hand_pos-target_pos)
        reward = - l1_dist - 5.0 * l2_dist
        ob = self.get_obs()
        self.env_timestep += 1   # keep track of env timestep for timed events
        self.trigger_timed_events()
        return ob, reward, False, self.get_env_infos()

    def get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.data.site_xpos[self.hand_sid],
            self.data.site_xpos[self.hand_sid] - self.data.site_xpos[self.target_sid],
        ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
        target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
        target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        self.env_timestep = 0
        return self.get_obs()

    def trigger_timed_events(self):
        # will be used in the continual version
        pass

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    qa=self.data.qacc.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        qa = state['qa'].copy()
        target_pos = state['target_pos']
        self.env_timestep = state['timestep']
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.data.qacc[:] = qa
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 2.0


class ContinualReacher7DOFEnv(Reacher7DOFEnv):

    def trigger_timed_events(self):
        if self.env_timestep % 50 == 0 and self.env_timestep > 0 and self.real_step is True:
            self.target_reset()
