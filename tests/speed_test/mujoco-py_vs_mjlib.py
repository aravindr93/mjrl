"""
Compare the speeds between mujoco-py and mjlib.
As of now, mjlib doesn't work with multiprocessing in python.
So we will compare single core performance of both wrappers.

NOTE: Change the path directory to point to where you have mujoco
"""

XML_PATH = "/home/aravind/.mujoco/mjpro150/model/humanoid.xml"

import numpy as np
import time
from mujoco_py import load_model_from_path, MjSim
from dm_control.mujoco import Physics
SEED = 123
N_TRAJ = 100
N_STEPS = 250

mujoco_py_sim = MjSim(load_model_from_path(XML_PATH))
mjlib_sim = Physics.from_xml_path(XML_PATH)
act_bounds = mjlib_sim.model.actuator_ctrlrange

# mujoco-py
t_start = time.time()
for tau in range(N_TRAJ):
	np.random.seed(SEED + tau)
	mujoco_py_sim.reset()
	for t in range(N_STEPS):
		ctrl = np.random.uniform(low=act_bounds[:,0], high=act_bounds[:,1])
		mujoco_py_sim.data.ctrl[:] = ctrl[:]
		mujoco_py_sim.step()
t_end = time.time()
print("mujoco-py time = %f" % (t_end-t_start))

# mjlib
t_start = time.time()
for tau in range(N_TRAJ):
	np.random.seed(SEED + tau)
	mjlib_sim.reset()
	for t in range(N_STEPS):
		ctrl = np.random.uniform(low=act_bounds[:,0], high=act_bounds[:,1])
		mjlib_sim.data.ctrl[:] = ctrl[:]
		mjlib_sim.step()
t_end = time.time()
print("mjlib time = %f" % (t_end-t_start))
