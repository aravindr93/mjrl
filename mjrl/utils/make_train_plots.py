import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from mjrl.utils.logger import DataLog

def make_train_plots(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None,
                     sample_key = 'num_samples'):
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            plt.figure(figsize=(10,6))
            try: 
                cum_samples = [np.sum(log[sample_key][:i]) for i in range(len(log[sample_key]))]
                plt.plot(cum_samples, log[key])
                plt.xlabel('samples')
            except:
                plt.plot(log[key])
                plt.xlabel('iterations')
            plt.title(key)
            plt.savefig(save_loc+'/'+key+'.png', dpi=100)
            plt.close()
