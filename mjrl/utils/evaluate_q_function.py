
from mjrl.utils.process_samples import discount_sum
import numpy as np

    
def get_ith(path, i):
    new_path = {} 
    for k in path: 
        try: 
            new_path[k] = path[k][i:i+1] 
        except: 
            pass 
    return new_path 

def get_first(path):
    return get_ith(path, 0)

def get_last(path): 
    last_idx = len(path['observations']) - 1
    return get_ith(path, last_idx)

def evaluate(path, gamma, baseline):
    T = len(path['actions'])
    p0 = get_first(path)
    pl = get_last(path)
    
    s0 = p0['observations']
    a0 = p0['actions']
    t0 = p0['time']

    sl = pl['observations']
    al = pl['actions']
    tl = pl['time']
    pred = baseline.predict(s0, a0, t0)
    last = baseline.predict(sl, al, tl)

    mc = discount_sum(path['rewards'], gamma)
    return pred, mc[0] + + gamma**T * last

def evaluate_idx(path, start_idx, end_idx, gamma, baseline):
    if start_idx >= end_idx:
        raise IndexError('start_idx should be < than end_idx')
    
    p0 = get_ith(path, start_idx)
    pl = get_ith(path, end_idx)
    
    s0 = p0['observations']
    a0 = p0['actions']
    t0 = p0['time']

    sl = pl['observations']
    al = pl['actions']
    tl = pl['time']
    pred = baseline.predict(s0, a0, t0)
    last = baseline.predict(sl, al, tl)

    mc = discount_sum(path['rewards'][start_idx:end_idx], gamma)

    return pred, mc[0] + + gamma**(end_idx - start_idx) * last

def evaluate_start_end(gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        pred, mc_term = evaluate(path, gamma, baseline)
        preds.append(pred[0])
        mc_terms.append(mc_term[0])
    return preds, mc_terms

def evaluate_n_step(n, gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        T = len(path['observations'])
        for t in range(T-n):
            pred, mc_term = evaluate_idx(path, t, t+n, gamma, baseline)
            preds.append(pred[0])
            mc_terms.append(mc_term[0])
    return preds, mc_terms

def mse(pred, mc):
    pred = np.array(pred)
    mc = np.array(mc)
    n = len(mc)
    return np.sum((pred - mc)**2) / n


if __name__ == '__main__':
    pass
