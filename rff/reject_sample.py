import numpy as np

# 3-dimensional isotropic rejection sampling
# 
def reject_sample(R, F, N):
        samples = np.zeros((N*2, 3))
        i = 0

        while i < N:
                x, y, z = np.random.uniform(-R[-1], R[-1], (3, N))
                p = np.random.uniform(0, F[0], N)
                u = np.interp((x*x+y*y+z*z)**0.5, R, F, right=0)

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1)), 
                                z[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]
        
def joint_reject_sample(Rt, Ft, Rx, Fx, Ry, Fy, N):
        '''
        Importance sampling from multi-variate distribution. The dstribution must be 
        independent and RBF w.r.t. each variable, i.e., depends only on length.
        R:  radius
        F:  probability density, each of the same size as R
        '''
        samples = np.zeros((N*2, 3))
        i = 0
        while i < N:
                t = np.random.uniform(-Rt[-1], Rt[-1], (1, N))
                x = np.random.uniform(-Rx[-1], Rx[-1], (1, N))
                y = np.random.uniform(-Ry[-1], Ry[-1], (1, N))
                p = np.random.uniform(0, Fx[0]*Fy[0]*Ft[0], N)
                u = np.interp(np.abs(t), Rt, Ft, right=0) * np.interp(np.abs(x), Rx, Fx, right=0) * np.interp(np.abs(y), Ry, Fy, right=0)
                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                t[mask].reshape((-1,1)),
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]

def rff_sample(gamma_t, gamma_x, gamma_y, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is defined as 
                K = exp(-\frac{x^2}{gamma_x}) * exp(-\frac{y^2}{gamma_y}) * exp(-\frac{t^2}{gamma_t}).
        The Fourier transform of K: 
                F(K) = F(w_x,w_y,w_t) = F(w_x)*F(w_y)*F(w_t),
        where 
                F(w_x) = exp(-\frac{w_x^2}{4*gamma_x^-1}).
        Therefore, we sample from a probability distribution (non-normalized) 
        p(w) = F(w_x)*F(w_y)*F(w_z).

        '''
        # find maximum frequency so that its coefficient is less than 1e-2
        Rt = np.linspace(0, (2*np.log(10)*4/gamma_t)**0.5, 1000)
        Rx = np.linspace(0, (2*np.log(10)*4/gamma_x)**0.5, 1000)
        Ry = np.linspace(0, (2*np.log(10)*4/gamma_y)**0.5, 1000)
        # the Fourier transform of each separate kernel
        Ft = np.exp(-Rt**2*gamma_t/4)
        Fx = np.exp(-Rx**2*gamma_x/4)
        Fy = np.exp(-Ry**2*gamma_y/4)
        return joint_reject_sample(Rt, Ft, Rx, Fx, Ry, Fy, N)
