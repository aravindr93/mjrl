import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent_bid = 0
        self.target_sid = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'point_mass.xml', 5)
        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        dist = np.linalg.norm(agent_pos-target_pos)
        reward = -0.01*dist
        if dist < 0.1:
            reward += 1.0 # bonus for being very close
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        return np.concatenate([agent_pos[:2], self.data.qvel.ravel(), target_pos[:2]])

    def reset_model(self):
        # randomize the agent and goal
        agent_x = self.np_random.uniform(low=-1.0, high=1.0)
        agent_y = self.np_random.uniform(low=-1.0, high=1.0)
        goal_x  = self.np_random.uniform(low=-1.0, high=1.0)
        goal_y  = self.np_random.uniform(low=-1.0, high=1.0)
        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid][0] = goal_x
        self.model.site_pos[self.target_sid][1] = goal_y
        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()