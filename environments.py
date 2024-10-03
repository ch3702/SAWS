import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
import math
import cvxpy as cp

# sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))



# Gaussian mean estimation
class Gauss_env:
    
    def __init__(self, th_list, sigma):
        self.n = 0   # current time
        self.N = len(th_list)   # total number of time periods
        self.d = len(th_list[0])   # dimension
        self.th_list = th_list   # list of Gaussian means
        self.sigma = sigma   # standard deviation
        self.th = None   # current Gaussian mean
        self.batch = None   # current sample
        self.fclass = 'scv'   # strongly convex function class
        
    def get_batch(self, B):
        # get a batch of B iid samples, each as a row
        self.th = self.th_list[self.n-1]
        cov = (self.sigma**2) * np.identity(self.d)
        samp_matrix = np.random.multivariate_normal(self.th, cov, B)
        self.batch = [np.array(samp) for samp in samp_matrix.tolist()]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def Gauss_loss(v, sample):
            return 0.5 * np.linalg.norm(v - sample)**2
        return Gauss_loss
    
    def get_grad(self):
        # get gradient of empirical loss
        def Gauss_grad(v, sample):
            return v - sample
        return Gauss_grad
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def Gauss_avg(batches):
            all_samples = [sample for batch in batches for sample in batch]
            return np.mean(all_samples, axis=0)
        return Gauss_avg
    
    def get_excess_loss(self, v):
        return 0.5 * np.linalg.norm(v - self.th)**2
    
    def update(self):
        self.n += 1



# linear regression
class lin_reg_env:
    
    def __init__(self, th_list, x_list, sigma_x, sigma_eps, M):
        self.n = 0   # current time
        self.N = len(th_list)   # total number of time periods
        self.d = len(th_list[0])   # dimension
        self.th_list = th_list   # list of true thetas
        self.x_list = x_list   # list of covariate means
        self.x = None   # current covariate mean
        self.sigma_x = sigma_x   # standard deviation of covariate
        self.sigma_eps = sigma_eps   # standard deviation of noise
        self.th = None   # current true theta
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'scv'   # strongly convex function class
        self.M = M   # radius of decision space
        
    def get_batch(self, B):
        # get a batch of B iid samples, each as a row
        self.th = self.th_list[self.n-1]
        self.x = self.x_list[self.n-1]
        
        cov_x = (self.sigma_x**2) * np.identity(self.d)
        samp_x_matrix = np.random.multivariate_normal(self.x, cov_x, B)
        samp_x_list = [np.array(samp) for samp in samp_x_matrix.tolist()]
        
        samp_eps_array = np.random.normal(0, self.sigma_eps, B)
        samp_eps_list = [np.array(samp) for samp in samp_eps_array.tolist()]
        
        samp_y_list = [np.dot(samp_x_list[i], self.th) + samp_eps_list[i] for i in range(B)]
        
        self.batch = [[samp_x_list[i], samp_y_list[i]] for i in range(B)]
        
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def lin_reg_loss(v, sample):
            return 0.5 * (np.dot(sample[0], v) - sample[1])**2
        return lin_reg_loss
    
    def get_grad(self):
        # get gradient of empirical loss
        def lin_reg_grad(v, sample):
            return (np.dot(sample[0], v) - sample[1]) * sample[0]
        return lin_reg_grad
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def solve_lin_reg(batches):
            all_x = [sample[0] for batch in batches for sample in batch]
            all_y = [sample[1] for batch in batches for sample in batch]
            X = np.stack(all_x, axis=0)
            Y = np.stack(all_y, axis=0)

            # linear regression
            reg = LinearRegression(fit_intercept=False).fit(X, Y)
            th = reg.coef_

            # use CVX instead if solution is too large
            if np.linalg.norm(th) > self.M:
                var = cp.Variable(self.d)
                obj = cp.Minimize(cp.sum_squares(X @ var - Y))
                con = [cp.norm(var) <= self.M]
                prob = cp.Problem(obj, con)
                prob.solve()
                th = var.value
            return th
        return solve_lin_reg
    
    def get_excess_loss(self, v):
        return 0.5 * (self.sigma_x * np.linalg.norm(self.th - v))**2
    
    def update(self):
        self.n += 1



# logistic regression
class logit_reg_env:
    
    def __init__(self, th_list, x_list, sigma_x, M):
        self.n = 0   # current time
        self.N = len(th_list)   # total number of time periods
        self.d = len(th_list[0])   # dimension
        self.th_list = th_list   # list of true thetas
        self.x_list = x_list   # list of covariate means
        self.x = None   # current covariate mean
        self.sigma_x = sigma_x   # standard deviation of covariate
        self.th = None   # current true theta
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'scv'   # strongly convex function class
        self.M = M   # radius of decision space
        
    def get_batch(self, B, update=True):
        # get a batch of B iid samples, each as a row
        self.th = self.th_list[self.n-1]
        self.x = self.x_list[self.n-1]
        
        cov_x = (self.sigma_x**2) * np.identity(self.d)
        samp_x_matrix = np.random.multivariate_normal(self.x, cov_x, B)
        samp_x_list = [np.array(samp) for samp in samp_x_matrix.tolist()]
        
        samp_y_list = [np.random.binomial(1, sigmoid(np.dot(self.th, samp_x_list[i]))) for i in range(B)]
        
        samp = [[samp_x_list[i], samp_y_list[i]] for i in range(B)]

        if update:   # whether to update self.sample
            self.batch = samp
        
        return samp
    
    def get_loss(self):
        # get empirical loss function
        def logit_reg_loss(v, sample):
            prod = np.dot(v, sample[0])
            return math.log(1 + math.exp(prod)) - sample[1] * prod
        return logit_reg_loss
    
    def get_grad(self):
        # get gradient of empirical loss
        def logit_reg_grad(v, sample):
            return (sigmoid(np.dot(v, sample[0])) - sample[1]) * sample[0]
        return logit_reg_grad
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def solve_logit_reg(batches):
            all_x = [sample[0] for batch in batches for sample in batch]
            all_y = [sample[1] for batch in batches for sample in batch]
            X = np.stack(all_x, axis=0)
            Y = np.stack(all_y, axis=0)

            # logistic regression (with regularization)
            reg = LogisticRegression(fit_intercept=False).fit(X, Y)
            th = reg.coef_

            # use CVX instead if solution is too large
            if np.linalg.norm(th) > self.M:
                var = cp.Variable(self.d)
                obj = cp.Minimize(cp.sum(cp.logistic(X @ var) - cp.multiply(Y, X @ var)))
                con = [cp.norm(var) <= self.M]
                prob = cp.Problem(obj, con)
                prob.solve()
                th = var.value
            return th
        return solve_logit_reg
    
    def get_excess_loss(self, v):
        # approximate excess loss by simulation
        sim_samp = self.get_batch(1000, update=False)
        logit_loss = self.get_loss()
        return np.mean([logit_loss(v, samp) for samp in sim_samp])
    
    def update(self):
        self.n += 1



# prediction with expert advice
class experts_env:
    
    def __init__(self, z_list, sigma_z):
        self.n = 0   # current time
        self.N = len(z_list)   # total number of time periods
        self.d = len(z_list[0])   # dimension
        self.z_list = z_list   # list of expert mean
        self.z = None   # current expert mean
        self.sigma_z = sigma_z   # standard deviation of experts
        self.batch = None    # batch of sample(s) at current time
        self.fclass = 'lip'   # Lipschitz function class
        
    def get_batch(self, B):
        # get a batch of B iid samples
        self.z = self.z_list[self.n-1]
        cov_z = (self.sigma_z**2) * np.identity(self.d)
        samp_matrix = np.random.multivariate_normal(self.z, cov_z, B)
        self.batch = [np.array(samp) for samp in samp_matrix.tolist()]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def experts_loss(v, sample):
            return np.dot(v, sample)
        return experts_loss
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def min_expert(batches):
            best = np.argmin(sum([sample for batch in batches for sample in batch]))
            th = np.zeros(self.d)
            th[best] = 1
            return th
        return min_expert
    
    def get_excess_loss(self, v):
        return np.dot(v, self.z) - np.min(self.z)
    
    def update(self):
        self.n += 1



# Gaussian mean estimation with real data
class real_Gauss_env:
    
    def __init__(self, d, N, batch_list):
        self.n = 0   # current time
        self.N = N   # total number of time periods
        self.d = d   # dimension
        self.batch_list = batch_list   # all batches of samples
        self.th = None   # current Gaussian mean
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'scv'   # strongly convex function class
        
    def get_batch(self, B=1):
        self.batch = self.batch_list[self.n-1][:B]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def Gauss_loss(v, sample):
            return 0.5 * np.linalg.norm(v - sample)**2
        return Gauss_loss
    
    def get_grad(self):
        # get gradient of empirical loss
        def Gauss_grad(v, sample):
            return v - sample
        return Gauss_grad
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def Gauss_avg(batches):
            all_samples = [sample for batch in batches for sample in batch]
            return np.mean(all_samples, axis=0)
        return Gauss_avg
    
    def get_excess_loss(self, v):
        return 0
    
    def update(self):
        self.n += 1



#  linear regression with real data
class real_linear_reg_env:
    
    def __init__(self, d, N, batch_list, M=np.inf):
        self.n = 0   # current time
        self.N = N   # total number of time periods
        self.d = d   # dimension
        self.batch_list = batch_list   # all batches of samples
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'scv'   # strongly convex function class
        self.M = M   # radius of decision space
        
    def get_batch(self, B=1):
        self.batch = self.batch_list[self.n-1][:B]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def lin_reg_loss(v, sample):
            return 0.5 * (np.dot(sample[0], v) - sample[1])**2
        return lin_reg_loss
    
    def get_grad(self):
        # get gradient of empirical loss
        def lin_reg_grad(v, sample):
            return (np.dot(sample[0], v) - sample[1]) * sample[0]
        return lin_reg_grad
    
    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def solve_lin_reg(batches):
            all_x = [sample[0] for batch in batches for sample in batch]
            all_y = [sample[1] for batch in batches for sample in batch]
            X = np.stack(all_x, axis=0)
            Y = np.stack(all_y, axis=0)

            # linear regression
            reg = LinearRegression(fit_intercept=False).fit(X, Y)
            th = reg.coef_

            # use CVX instead if solution is too large
            if np.linalg.norm(th) > self.M:
                var = cp.Variable(self.d)
                obj = cp.Minimize(cp.sum_squares(X @ var - Y))
                con = [cp.norm(var) <= self.M]
                prob = cp.Problem(obj, con)
                prob.solve()
                th = var.value
            return th
        return solve_lin_reg
    
    def get_excess_loss(self, v):
        return 0
    
    def update(self):
        self.n += 1



#  newsvendor problem with real data
class real_newsvendor_env:
    
    def __init__(self, N, batch_list, r, M=np.inf):
        self.n = 0   # current time
        self.N = N   # total number of time periods
        self.d = 1   # dimension
        self.batch_list = batch_list   # all batches of samples
        self.r = r   # critical ratio
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'lip'   # Lipschitz function class
        self.M = M   # radius of decision space
        
    def get_batch(self, B=1):
        self.batch = self.batch_list[self.n-1][:B]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def newsvendor_loss(v, sample, r=self.r):
            return r * max(sample-v, 0) + (1-r) * max(v-sample, 0)
        return newsvendor_loss

    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def solve_newsvendor(batches, r=self.r):
            all_y = [sample for batch in batches for sample in batch]
            return np.quantile(all_y, r, method='closest_observation')
        
        return solve_newsvendor
    
    def get_excess_loss(self, v):
        return 0
    
    def update(self):
        self.n += 1



#  quantile regression with real data
class real_quantile_env:
    
    def __init__(self, N, d, batch_list, r, M=np.inf):
        self.n = 0   # current time
        self.N = N   # total number of time periods
        self.d = d   # dimension
        self.batch_list = batch_list   # all batches of samples
        self.batch = None   # batch of sample(s) at current time
        self.fclass = 'lip'   # Lipschitz function class
        self.r = r   # critical ratio
        self.M = M   # radius of decision space
        
    def get_batch(self, B=1):
        self.batch = self.batch_list[self.n-1][:B]
        return self.batch
    
    def get_loss(self):
        # get empirical loss function
        def quantile_loss(v, sample, r=self.r):
            return r * max(sample[1]-np.dot(sample[0],v), 0) + (1-r) * max(np.dot(sample[0],v)-sample[1], 0)
        return quantile_loss

    def get_solver(self):
        # solver for empirical minimization from a list of batches
        def solve_lin_reg(batches, r=self.r):
            all_x = [sample[0] for batch in batches for sample in batch]
            all_y = [sample[1] for batch in batches for sample in batch]
            X = np.stack(all_x, axis=0)
            Y = np.stack(all_y, axis=0)

            # quantile regression
            reg = QuantileRegressor(quantile=r, fit_intercept=False, alpha=0, solver='highs').fit(X, Y)
            th = reg.coef_

            # use CVX instead if solution is too large
            if np.linalg.norm(th) > self.M:
                var = cp.Variable(self.d)
                obj = cp.Minimize(cp.sum(r * cp.pos(Y-X@var) + (1-r) * cp.neg(Y-X@var)))
                con = [cp.norm(var) <= self.M]
                prob = cp.Problem(obj, con)
                prob.solve()
                th = var.value
            return th
        return solve_lin_reg
    
    def get_excess_loss(self, v):
        return 0
    
    def update(self):
        self.n += 1