import logging
logging.disable(logging.CRITICAL)

import numpy as np
import copy
import multiprocessing as mp
import time as timer
import mjrl.samplers.base_sampler as base_sampler
import mjrl.samplers.evaluation_sampler as eval_sampler

def sample_paths(N, policy, T=1e6, env=None, env_name=None, pegasus_seed=None, mode='sample'):
    if mode == 'sample':
        return base_sampler.do_rollout(N, policy, T, env, env_name, pegasus_seed)
    elif mode == 'evaluation':
        return eval_sampler.do_evaluation_rollout(N, policy, env, env_name, pegasus_seed)
    else:
        print("Mode has to be either 'sample' for training time or 'evaluation' for test time performance")

def sample_paths_parallel(N,
    policy,
    T=1e6,
    env_name=None,
    pegasus_seed=None,
    num_cpu='max',
    max_process_time=300,
    max_timeouts=4,
    suppress_print=False,
    mode='sample'):

    if num_cpu == None or num_cpu == 'max':
        num_cpu = mp.cpu_count()
    elif num_cpu == 1:
        return base_sampler.do_rollout(N, policy, T, None, env_name, pegasus_seed)
    else:
        num_cpu = min(mp.cpu_count(), num_cpu)

    paths_per_cpu = int(np.ceil(N/num_cpu))
    args_list = []
    for i in range(num_cpu):
        if pegasus_seed is None:
            args_list_cpu = [paths_per_cpu, policy, T, None, env_name, pegasus_seed]
        else:
            args_list_cpu = [paths_per_cpu, policy, T,
                            None, env_name, pegasus_seed+i*paths_per_cpu]           
        args_list.append(args_list_cpu)

    # Do multiprocessing
    if suppress_print == False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(args_list, num_cpu,
                                max_process_time, max_timeouts, mode)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print == False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )
    
    return paths

def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts, mode):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    if mode == 'sample':
        parallel_runs = [pool.apply_async(base_sampler.do_rollout_star,
                        args=(args_list[i],)) for i in range(num_cpu)]
    elif mode == 'evaluation':
        parallel_runs = [pool.apply_async(eval_sampler.do_evaluation_rollout_star,
                        args=(args_list[i],)) for i in range(num_cpu)]
    else:
        print("Mode has to be either 'sample' for training time or 'evaluation' for test time performance")
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()        
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts-1, mode)

    pool.close()
    pool.terminate()
    pool.join()  
    return results
