import numpy as np

def rbf_sample(gamma_t, gamma_x, gamma_y, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is defined as 
                K = exp(-x^2*gamma_x) * exp(-y^2*gamma_y) * exp(-t^2*gamma_t).
        The Inverse Fourier transform of K: 
                P(w) = C*exp(-0.5*X^T*\Sigma^-1*X),
        where 
                \Sigma = diag([2*gamma_x, 2*gamma_y, 2*gamma_t]).

        Therefore, we sample from the probability distribution P(w)
        '''
        cov = np.diag([2*gamma_t, 2*gamma_x, 2*gamma_y])
        return np.random.multivariate_normal(np.array([0,0,0]), cov, N)

def exp_sample(a_t, a_x, a_y, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is defined as 
                K = exp(-abs(x)*gamma_x) * exp(-abs(y)*gamma_y) * exp(-abs(t)*gamma_t).
        The Inverse Fourier transform of each separate k: 
                P(w) = 2*gamma/(gamma^2+w^2).

        We sample from P(w) using rejection sampling method.
        '''
        samples = np.zeros((N*2, 3))
        i = 0

        while i < N:
                t,x,y = np.random.uniform(-400, 400, (3, N))
                p = np.random.uniform(0, 8./(a_t*a_x*a_y), N)
                u = 8*a_t*a_x*a_y/((a_t**2+t**2) * (a_x**2+x**2) * (a_y**2+y**2))

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                t[mask].reshape((-1,1)), 
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]
