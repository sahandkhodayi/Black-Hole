import numpy as np




def RK4(rhs,phi,state,step_size):

    state = np.asarray(state, dtype=float)
    if not np.isfinite(step_size) or step_size <=0:
        raise ValueError(" step size in negetive")
    
    k1 = np.asarray(rhs(phi,state),dtype=float)

    k2=np.asarray(rhs(phi+step_size/2,state+(k1*step_size/2)),dtype=float)

    k3=np.asarray(rhs(phi+step_size/2,state+(k2*step_size/2)),dtype=float)

    k4=np.asarray(rhs(phi+step_size,state+(k3*step_size)),dtype=float)

    return state + (step_size/6)*(k1+2*k2+2*k3+k4)