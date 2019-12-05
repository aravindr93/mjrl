import numpy as np
from gym import utils
from mjrl.envs import mujoco_env


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class Humanoid3Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_3.xml', 5)
        utils.EzPickle.__init__(self)
        
        self.NU = self.env.env.model.nu // 3

        self.default_motor_gear = np.array([2.,2,2,2,2,6,4,1,1,2,2,6,4,1,1,1,1,2,1,1,2]) * 100

    def get_obs(self):
        data = self.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self.get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self.get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()

    def set_gain(self, index, gain):
        self.env.env.model.actuator_gainprm[index][0] = gain
    
    def set_gear(self, index, gear):
        self.env.env.model.actuator_gear[index][0] = gear

    def set_motor_gain(self, gain):
        self.env.env.model.actuator_gainprm[:self.NU] = gain
        
    def set_velocity_gain(self, gain):
        self.env.env.model.actuator_gainprm[self.NU+1:self.NU*2+1] = gain
        
    def set_position_gain(self, gain):
        self.env.env.model.actuator_gainprm[-self.NU:] = gain



