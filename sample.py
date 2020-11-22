import numpy as np

def rff_sample(gamma_t, gamma_x, gamma_y, N):
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
