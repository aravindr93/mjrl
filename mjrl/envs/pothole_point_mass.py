
from mjrl.envs.point_mass import PointMassEnv
from mjrl.envs import mujoco_env

import numpy as np

class PotholePointMassEnv(PointMassEnv):

    def __init__(self):
        # (x,y,w,h) where (x,y) is the bottom left coordinate
        self.rectangles = [
            (-1, -1.2, 0.5, 2),
            (0.5, 0, 0.7, 0.3)
        ]

        super().__init__()
        mujoco_env.MujocoEnv.__init__(self, 'pothole_point_mass.xml', 5)

    def is_in_rectangle(self, x, y):
        for rectangle in self.rectangles:
            is_in_x = x > rectangle[0] and x < rectangle[0] + rectangle[2]
            is_in_y = y > rectangle[1] and y < rectangle[1] + rectangle[3]
            if is_in_x and is_in_y:
                return True
        return False

    def get_reward(self, obs, act=None):
        agent_pos = obs[:2]
        target_pos = obs[-2:]
        l2_dist = -np.linalg.norm(agent_pos - target_pos)
        in_rect_penalty = 0.0
        close_bonus = 0.0
        if self.is_in_rectangle(agent_pos[0], agent_pos[1]):
            in_rect_penalty -= 100

        if l2_dist <= 0.01:
            close_bonus += 10

        reward = l2_dist + in_rect_penalty + close_bonus
        return reward
    
    def reset_model(self):
        agent_x = self.np_random.uniform(low=-1.3, high=1.3)
        agent_y = self.np_random.uniform(low=-1.3, high=1.3)
        while self.is_in_rectangle(agent_x, agent_y):
            agent_x = self.np_random.uniform(low=-1.3, high=1.3)
            agent_y = self.np_random.uniform(low=-1.3, high=1.3)
        
        goal_x  = 1.0
        goal_y  = 1.0
        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid][0] = goal_x
        self.model.site_pos[self.target_sid][1] = goal_y
        self.sim.forward()
        return self.get_obs()

if __name__ == '__main__':
    env = PotholePointMassEnv()