"""
Perform rollouts in parallel using multiprocessing.
MjLib seems to throw error with multiprocessing. Require help from DM to fix.

NOTE: Change the path directory to point to where you have mujoco
"""

XML_PATH = "/home/aravind/.mujoco/mjpro150/model/humanoid.xml"

import numpy as np
import multiprocessing
import time
from mujoco_py import load_model_from_path, MjSim
SEED = 123
N_PROCESS = 4
N_STEPS = 10000

# ------------------------
def run_parallel(target_function, args_list):
	"""
	Runs the target_function on multiple processes in parallel

    Args:
        target_function: function to be executed on multiple processes
        args_list: a list with each element being arguments to be used for target_function

    Returns:
    	Output of each target_function as a list
    """

    num_runs = len(args_list)
    num_cpu  = multiprocessing.cpu_count()
    assert num_runs <= num_cpu

    workers = []
    for i in range(num_runs):
    	proc = multiprocessing.Process(
    			target=target_function, args=(*args_list[i],))
    	proc.start()
    	workers.append(proc)

	results = [worker.join() for worker in workers]
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


def rollout_mjlib(seed, horizon, thread_id):
    # Must import DM Control here for fork-safety.
    import dm_control.mujoco as dmj
    mjlib_sim = dmj.Physics.from_xml_path(XML_PATH) # create sim inside the process
    act_bounds = mjlib_sim.model.actuator_ctrlrange
    np.random.seed(seed)
    print("Started mjlib thread : %s at time %f" % (thread_id, time.time()))
    for t in range(N_STEPS):
        ctrl = np.random.uniform(low=act_bounds[:,0], high=act_bounds[:,1])
        mjlib_sim.data.ctrl[:] = ctrl[:]
        mjlib_sim.step()
    print("End of  mjlib thread : %s at time %f" % (thread_id, time.time()))
    return True



if __name__ == '__main__':

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
	results = run_parallel(rollout_mujoco_py, args_list)

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
	results = run_parallel(rollout_mjlib, args_list)