import numpy as np
import math

def SAWS_offline(batches, thres, winds, loss, solver):

    ''' Stability-based adaptive window selection (SAWS) in an individual time period
    Args:
        batches: list of batches of samples; if current time is n, then batches[-i] is a list that stores samples at time n-i+1
        thres: list of threshold functions for non-stationarity tests
        winds: list of candidate window sizes
        loss: loss function ell(theta, z)
        solver: empirical risk minimization solver

    Returns:
        w_opt: selected window size
        th_opt: selected empirical minimizer
    '''

    m = len(winds)   # number of candidate windows
    w_opt = None   # selected window
    th_opt = None   # selected minimizer
    stationary = True   # stationarity indicator
    thmins = []   # thmins[s] stores the approximate minimizer of f_{n,k_s}
    fmins = []   # fmins[s] stores the approximate minimum of f_{n,k_s}
    
    j = 0

    # pairwise comparisons of window sizes
    while stationary and j <= m-1:
        w = winds[j]
        
        thmin = solver(batches[-w:])
        fmin = np.mean([np.mean([loss(thmin, sample) for sample in batch]) for batch in batches[-w:]])
        
        # comparison with window sizes k_s, s = 0,1,...,j-1
        for s in range(j):
            if np.mean([np.mean([loss(thmin, sample) for sample in batch]) for batch in batches[-winds[s]:]]) - fmins[s] > thres[s]:
                stationary = False
                w_opt = winds[j-1]
                th_opt = thmins[-1]
                break
           
        # record minimizer and minimum of f_{n,k_j}
        thmins.append(thmin)
        fmins.append(fmin)
        j += 1
    
    # if all tests are passed, then pick the largest window
    if stationary:
        w_opt = winds[-1]
        th_opt = thmins[-1]
    
    return w_opt, th_opt



def SAWS_online(env, loss, param, init_dcsn, solver):

    ''' Implements SAWS for multiple periods
    Args:
        env: problem environment
        loss: loss function associated with the problem
        param: a dictionary of hyperparameters for SAWS
        init_dcsn: initial decision in the first period
        solver: empirical risk minimization solver associated with the problem

    Returns:
        dcsn_list: list of decisions made, indexed by time
        wind_list: list of selected window sizes, indexed by time
        reg_list: list of cumulative regrets, indexed by time
    '''
    
    d = env.d   # dimension
    N = env.N   # time horizon
    B = param['B']   # batch size in every period
    tau = param['tau']   # threshold parameter
    alpha = param['alpha']   # probability of exception


    # record decisions, batches, regrets
    dcsn_list = []   # list of decisions
    wind_list = []   # list of selected windows
    batch_cache = []   # list of stored batches, excluding discarded ones
    reg = 0   # cumulative regret
    reg_list = []   # list of cumulative regrets


    # first period
    dcsn_list.append(init_dcsn)   # initial decision
    wind_list.append(0)   # window size 0
    env.update()   # update environment
    batch = env.get_batch(B)   # get sample
    batch_cache.append(batch)
    
    # compute excess loss
    if param['compute_reg']:
        excess = env.get_excess_loss(init_dcsn)
        reg += excess
        reg_list.append(reg)
    

    # main loop
    for n in range(2, N+1):

        # candidate window sizes
        K = len(batch_cache)
        if param['geo_wind']:   # geometric candidate window sequence
            wind_candi = [2**j for j in range(math.ceil(math.log2(K)))] + [K]
        else:
            wind_candi = [j for j in range(1, K+1)]
        
        # choose thresholds according to function class
        if env.fclass == 'scv':
            thres_list = [tau * d * math.log(1/alpha + B + n) / (B*k) for k in wind_candi]
        elif env.fclass == 'lip':
            thres_list = [tau * math.sqrt(d * math.log(1/alpha + B + n) / (B*k)) for k in wind_candi]
        else:
            print('ERROR: Function class should be \'scv\' or \'lip\'!')
            return
        
        # make decision
        w, dcsn = SAWS_offline(batch_cache, thres_list, wind_candi, loss, solver)

        dcsn_list.append(dcsn)
        wind_list.append(w)
        
        # discard unused samples
        if not param['reuse']:
            batch_cache = batch_cache[-w:]

        # update environment
        env.update()

        # get sample from environment
        batch = env.get_batch(B)
        batch_cache.append(batch)

        # compute excess loss and regret
        if param['compute_reg']:
            excess = env.get_excess_loss(dcsn)
            reg += excess
            reg_list.append(reg)
        
    return dcsn_list, wind_list, reg_list



def MA(env, param, init_dcsn, solver):

    ''' Implements moving average (MA) with a fixed window size for multiple periods
    Args:
        env: problem environment
        param: a dictionary of hyperparameters for MA
        init_dcsn: initial decision in the first period
        solver: empirical risk minimization solver associated with the problem

    Returns:
        dcsn_list: list of decisions made, indexed by time
        reg_list: list of cumulative regrets, indexed by time
    '''

    N = env.N   # time horizon
    B = param['B']   # batch size in every period
    wind = param['wind']   # fixed window size

    # record decisions, batches, regrets
    dcsn_list = []   # list of decisions
    batch_cache = []   # list of stored batches, excluding discarded ones
    reg = 0   # cumulative regret
    reg_list = []   # list of cumulative regrets


    # first period
    dcsn_list.append(init_dcsn)   # initial decision
    env.update()   # update environment
    batch = env.get_batch(B)   # get sample
    batch_cache.append(batch)

    # compute excess loss and regret
    if param['compute_reg']:
        excess = env.get_excess_loss(init_dcsn)
        reg += excess
        reg_list.append(reg)
    
    
    # main loop
    for n in range(2, N+1):
        
        # discard unused samples
        batch_cache = batch_cache[-min(wind,n) : ]

        # make decision
        dcsn = solver(batch_cache)
        dcsn_list.append(dcsn)

        # update environment
        env.update()

        # get sample from environment
        batch = env.get_batch(B)
        batch_cache.append(batch)

        # compute excess loss and regret
        if param['compute_reg']:
            excess = env.get_excess_loss(dcsn)
            reg += excess
            reg_list.append(reg)

    return dcsn_list, reg_list