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
        
# 6-dimensional joint rejection sampling
def joint_reject_sample(R_p, F_p, R_d, F_d, N):
        samples = np.zeros((N*2, 6))
        i = 0

        while i < N:
                x,y,z = np.random.uniform(-R_p[-1], R_p[-1], (3, N))
                dx,dy,dz = np.random.uniform(-R_d[-1], R_d[-1], (3, N))
                p = np.random.uniform(0, F_p[0]*F_d[0], N)
                u = np.interp((x*x+y*y+z*z)**0.5, R_p, F_p, right=0) * np.interp((dx*dx+dy*dy+dz*dz)**0.5, R_d, F_d, right=0)

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1)), 
                                z[mask].reshape((-1,1)), 
                                dx[mask].reshape((-1,1)), 
                                dy[mask].reshape((-1,1)), 
                                dz[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]