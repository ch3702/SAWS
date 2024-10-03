import numpy as np
import math

def sample_sphere(n, d, ord):

    ''' Randomly sample n points over a unit-norm sphere in R^d
    Args:
        n: number of samples
        d: dimension
        ord: order of the norm
    
    Returns:
        v: n-by-d numpy array, with each sampled point as a row
    '''
    if ord == 2:
        v = np.random.randn(n, d)
        v /= np.matrix(np.linalg.norm(v, axis=1)).T
    elif ord == 1:
        u = np.random.dirichlet([1]*d, n)
        signs = 2 * np.random.randint(0, 2, size=(n, d)) - 1
        v = np.multiply(u, signs)
    elif ord == 'Inf':
        # for a generic sample, half of the entries are uniform over [-1,1], and the other half are uniform over {-1,1}
        u1 = np.random.random_sample((n,math.floor(d/2)))
        u2 = np.ones((n,math.ceil(d/2)))
        u = np.concatenate((u1,u2), axis=1)
        signs = 2 * np.random.randint(2, size=(n,d)) - 1
        v = np.multiply(signs, u)
        np.apply_along_axis(np.random.shuffle, 1, v) 
    return v



def piecewise(init_pt, N, jump_scale, jump_times, ord):

    ''' Generate piecewise stationary sequence
    Args:
        init_pt: initial point
        N: number of periods
        jump_scale: scale of the jump between stationary pieces
        jump_times: time periods of the jumps
        ord: order of the norm

    Returns: 
        pts_list: generated list of points, each point as a list
    '''
    
    d = len(init_pt)   # dimension
    pts = [np.array(init_pt)] * N
    num_jumps = min(len(jump_times), N)   # number of jumps
    V = sample_sphere(num_jumps, d, ord)
    jump_count = 0
    
    for time in range(1, N):

        # make a jump at every jump time
        if time in jump_times:
            pts[time] = jump_scale * V[jump_count]
            jump_count += 1
        else:
            pts[time] = pts[time-1]

    pt_list = [pt.tolist() for pt in pts]
        
    return pt_list
    


# strongly convex instance: linear regression

def scv_ins(d, N):

    ''' Generate a piecewise stationary sequence with varying segment lengths
    Args:
        d: dimension
        N: number of periods

    Returns: 
        th_list: generated list of points, each point as a list
    '''

    jump_time = 1
    jump_times = [1]
    
    for k in range(N):
        if k % 2 == 0:
            jump_time += 5 * math.ceil(N**(1/3))
        else:
            jump_time += 5 * math.ceil(N**(1/6))
        jump_times.append(jump_time)

        jump_time += 5 * math.ceil(N**(1/2))
        jump_times.append(jump_time)

        if jump_time >= N:
          break

    jump_scale = 3
    init_pt = [0] * d
    
    M = 2 * jump_scale

    th_list = piecewise(init_pt, N, jump_scale, jump_times, ord=2)

    return th_list, M



# Lipschitz instance: prediction from expert advice

def lip_ins(d, N):

    ''' Generate a piecewise stationary sequence with varying segment lengths
    Args:
        d: dimension
        N: number of periods

    Returns:
        th_list: generated list of points, each point as a list
    '''

    jump_time = 1
    jump_times = [1]

    for k in range(N):
        
        if k % 2 == 0:
            jump_time += math.ceil(N**(1/2))
        else:
            jump_time += math.ceil(N**(1/6))
        jump_times.append(jump_time)

        jump_time += math.ceil(N**(1/3))
        jump_times.append(jump_time)

        if jump_time >= N:
            break

    jump_scale = 5
    init_pt = [0] * d
    z_list = piecewise(init_pt, N, jump_scale, jump_times, ord='Inf')

    return z_list