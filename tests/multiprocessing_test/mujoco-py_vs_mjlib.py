"""
Perform rollouts in parallel using multiprocessing.
MjLib seems to throw error with multiprocessing. Require help from DM to fix.

NOTE: Change the path directory to point to where you have mujoco
"""

XML_PATH = "/home/aravind/.mujoco/mjpro150/model/humanoid.xml"

import numpy as np
import multiprocessing as mp
import time
from mujoco_py import load_model_from_path, MjSim
from dm_control.mujoco import Physics
SEED = 123
N_PROCESS = 4
N_STEPS = 10000

mujoco_py_sim = MjSim(load_model_from_path(XML_PATH))
mjlib_sim = Physics.from_xml_path(XML_PATH)
act_bounds = mjlib_sim.model.actuator_ctrlrange

# ------------------------
def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts, function):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=5)
    parallel_runs = [pool.apply_async(function,
                    args=(args_list[i],)) for i in range(num_cpu)]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts-1, function)

    pool.close()
    pool.terminate()
    pool.join()
    return results
# ---------------------


def rollout_mujoco_py(seed, horizon, thread_id):
    mujoco_py_sim = MjSim(load_model_from_path(XML_PATH)) # create sim inside the process
    act_bounds = mujoco_py_sim.model.actuator_ctrlrange
    np.random.seed(seed)
    print("Started mujoco-py thread : %s at time %f" % (thread_id, time.time()))
    for t in range(N_STEPS):
        ctrl = np.random.uniform(low=act_bounds[:,0], high=act_bounds[:,1])
        mujoco_py_sim.data.ctrl[:] = ctrl[:]
        mujoco_py_sim.step()
    print("End of  mujoco-py thread : %s at time %f" % (thread_id, time.time()))
    return True

def rollout_mujoco_py_star(args_list):
    return rollout_mujoco_py(*args_list)


def rollout_mjlib(seed, horizon, thread_id):
    mjlib_sim = Physics.from_xml_path(XML_PATH) # create sim inside the process
    act_bounds = mjlib_sim.model.actuator_ctrlrange
    np.random.seed(seed)
    print("Started mjlib thread : %s at time %f" % (thread_id, time.time()))
    for t in range(N_STEPS):
        ctrl = np.random.uniform(low=act_bounds[:,0], high=act_bounds[:,1])
        mjlib_sim.data.ctrl[:] = ctrl[:]
        mjlib_sim.step()
    print("End of  mjlib thread : %s at time %f" % (thread_id, time.time()))
    return True

def rollout_mjlib_star(args_list):
    return rollout_mjlib(*args_list)


# single threaded
print("***************")
print("mujoco-py single threaded execution")
for i in range(N_PROCESS):
    rollout_mujoco_py(i, N_STEPS, i)

# multiprocessing
print("mujoco-py multiprocessing execution")
args_list = []
for i in range(N_PROCESS):
    args_list_process = [i, N_STEPS, i]
    args_list.append(args_list_process)
results = _try_multiprocess(args_list, N_PROCESS, 100, 1, rollout_mujoco_py_star)

# single threaded (mjlib)
print("***************")
print("mjlib single threaded execution")
for i in range(N_PROCESS):
    rollout_mjlib(i, N_STEPS, i)

# multiprocessing
print("mjlib multiprocessing execution")
args_list = []
for i in range(N_PROCESS):
    args_list_process = [i, N_STEPS, i]
    args_list.append(args_list_process)
results = _try_multiprocess(args_list, N_PROCESS, 100, 1, rollout_mjlib_star)
