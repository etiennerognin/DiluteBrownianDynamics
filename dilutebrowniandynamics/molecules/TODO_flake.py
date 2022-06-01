# This is a library to simulate dilute solution of flakes
import numpy as np
from multiprocessing import Pool
import tqdm
from scipy import interpolate
from scipy.linalg.lapack import dptsv





# About the data structure:
# One flake is a triangle made of three Brownian beads linked together
# with rigid rods. Therefore a flake is represented by a (3,3) ndarray
# Ignoring rotation in the plane of the flake, a flake is also captured
# by the normal vector to the flake.



_maxiter = 1000
_tol = 1e-6
_verbose = False
_debug = False

# Adaptive time stepping levels (main dt divided by 2**level):
_maxlevel = 20

def simulate_flakes(flakes, gradU, Nrec, dt, avg=1, nproc=4):
    """Simulate an ensemble of chains under some velocity gradient conditions.
    
    Parameters
    ----------
    chains : list of (N,3) ndarray
        The list of chains to simulate.
    gradU : {callable, (9,) ndarray}
        Velocity gradient (row-wise) for the simulation. If `gradU` is callable,
        then it will be evaluated at each time step.
    Nrec : int
        Number of points to record.
    dt : float
        Dimensionless time step.
    avg : int, default 1
        Number of time step to average for one recording.
    nproc : int, default 4
        Number of processor cores to use.
        
    Returns
    -------
    A : (Nrec,6) ndarray
        Time series of the estimator of the conformation tensor (covariance of 
        the end-to-end vectors).
    stress: (Nrec,6) ndarray
        Time series of the estimator of the stress.
    flakes_out: list of (N, 3) ndarray
        List of chains after the last time step. 
    
    """
    
    import warnings
    warnings.simplefilter("ignore")
    
    Nensemble = len(flakes)  
    print("Physical time to compute:",Nrec*avg*dt)
    A = np.zeros((Nrec,6))
    stress = np.zeros((Nrec,6))
    flakes_out = []
    
    # Preparing arguments for parallelisation
    args = list((flake, gradU, Nrec, dt, avg, False) for flake in flakes)
    
    # The following loop is parallelised
    with Pool(nproc) as p:
        print("Calculation started on {} cores.".format(nproc))
        results = list(tqdm.tqdm(p.imap(flake_trajectory, args),total=Nensemble))
        
    for elemA, elemstress, flake in results:
        A += 1./Nensemble*elemA
        stress += 1./Nensemble*elemstress
        flakes_out.append(flake)
    
    return A, stress, flakes_out


def flake_trajectory(args):
    """Compute trajectory (in the modelcular dynamics sense) of a flake.
    
    Parameters
    ----------
    Packed in a tuple args.
    flake : (3, 3) ndarray
        Flake made of 3 rods.
    gradU : {callable, (9,) ndarray}
        Velocity gradient (row-wise) for the simulation. If `gradU` is callable,
        then it will be evaluated at each time step.
    Nrec : int
        Number of points to record.
    dt : float
        Dimensionless time step.
    avg : int
        Number of time step to average for one recording.
    full_trajectory : bool
        If True returns trajectory as a list of flakes at each time step.
        
    Returns
    -------
    elemA : (Nrec,6) ndarray
        Time series of Elementary conformation RR, where R is a vector normal
        to the flake.
    elemstress: (Nrec,6) ndarray
        Elementary stress.
    flake_out: (3, 3) ndarray or list of (3, 3) ndarray
        Flake after the last time step, or full list at each time step.
    
    Notes
    -----
    The random force is correlated with neighbour rods.
    
    """
    
    
    # Unpack
    flake_out, gradU, Nrec, dt, avg, full_trajectory = args

    # For parallel compatibility:
    np.random.seed()
    
    # Trajectories.
    elemA = np.zeros((Nrec,6))
    elemstress = np.zeros((Nrec,6))
    if full_trajectory:
        trajectory = []
    
    # Time step subdivision level
    level = 0
    dt_local = dt
    
    
    
    for i in range(Nrec):
        if _verbose: print(f"In flake_trajectory, iteration {i}; level {level}")
        
        # Part of the time step job done
        subit = 0
        while subit < avg*dt:
            # Random force 
            dW = np.sqrt(dt_local)*randn_central_limit((3,3))/np.sqrt(2)
            
            # -- To reduce brownian noise in stress, the following keeps
            # -- the part that is creating rotation only.
            # -- Note: the improvement is only minor.
            

            
            # Remove component not along normal
            R = np.cross(flake_out[0],flake_out[1])*2/np.sqrt(3) 
            dW = np.reshape(np.sum(dW*R, axis=1), (3,1))*R
            
            # Add diffusion aroud normal axis (not needed in practice)
            # Bead 0 is opposite to Q1, so Q1 can be used to support a pure
            # rotation force applied on bead 0, and so on.
            #dWN = (np.sqrt(dt_local)*np.random.standard_normal(1)/np.sqrt(2)*
            #       np.roll(flake_out, -1, axis=0)    )          
            #dW += dWN
               
            #(uncorrelated on beads, so correlated on rods!):      
            dW = np.roll(dW, -1, axis=0) -dW
            
            # Keep only rotating element
            #for i in range(3):
            #    dW[i] = dW[i] - np.sum(dW[i]*flake_out[i])*flake_out[i]
            
            # Evaluate velocity gradient
            gradUt = gradU(i*avg*dt+subit) if callable(gradU) else gradU
            
            # Evolve one time step
            # For now, only rigid model
            try:
                flake_out, tensions = evolve(flake_out, gradU, dt_local, dW)

                subit += dt_local
                # Record
                # -- Record
                # Normal vector
                R = np.cross(flake_out[0],flake_out[1])*2/np.sqrt(3)                
                elemA[i] = outer_sym(R, R) /(avg*2**level)
                              
                elemstress[i] += np.sum(outer_sym(tensions.reshape((3,1))*flake_out, flake_out), axis=0)/(avg*2**level)
            except:
                # evolve will raise exceptions if confergence fails. In that case
                # we subdivide the time step to increase stability
                if level < _maxlevel:
                    level += 1
                    dt_local = dt/2**level
                    
                else: 
                    raise RuntimeError("Convergence failed and maximum level of time step subdivision reached.")
        
        if full_trajectory: 
            trajectory.append(flake_out)
        if  level > 0:            
            level += -1
            dt_local = dt/2**level
    if full_trajectory:
        return elemA, elemstress, trajectory
    else:    
        return elemA, elemstress, flake_out


def outer_sym(a, b):
    if a.ndim==1 and b.ndim==1:
        return np.array([a[0]*b[0], a[0]*b[1], a[0]*b[2],
                                    a[1]*b[1], a[1]*b[2],
                                               a[2]*b[2]])
    if a.ndim==2 and b.ndim==2:
        out=np.empty((a.shape[0],6))
        out[:,0] = a[:,0]*b[:,0]
        out[:,1] = a[:,0]*b[:,1]
        out[:,2] = a[:,0]*b[:,2]
        out[:,3] = a[:,1]*b[:,1]
        out[:,4] = a[:,1]*b[:,2]
        out[:,5] = a[:,2]*b[:,2]
        return out

def randn_central_limit(size):
    return np.sum(np.random.random((size[0],size[1],12)), axis=2)-6

def evolve(flake, gradU, dt, dW):
    """Evolve one flake by a time step dt. Itô calculus convention.
    
    Parameters
    ----------
    flake : (N,3) ndarray
        flake of 3 rods. Each line is a rod vector Q.
    gradU : (9,) ndarray
        Velocity gradient, row-wise.
    dt : float
        Time step.
    dW : (3,3) ndarray
        Random forces
        
    Returns
    -------
    flake_out : (3,3) ndarray
        flake after time step
    tensions : (3,) ndarray
        Tension in rods during time step 
    """
    
    
    # Right hand side
    # Q_gradU_Q
    Q_gradU = np.empty_like(flake)
    Q_gradU[:,0] = gradU[0]*flake[:,0] + gradU[3]*flake[:,1] + gradU[6]*flake[:,2]
    Q_gradU[:,1] = gradU[1]*flake[:,0] + gradU[4]*flake[:,1] + gradU[7]*flake[:,2]
    Q_gradU[:,2] = gradU[2]*flake[:,0] + gradU[5]*flake[:,1] + gradU[8]*flake[:,2]
    
    Q_gradU_Q = (Q_gradU[:,0]*flake[:,0] + Q_gradU[:,1]*flake[:,1] + Q_gradU[:,2]*flake[:,2])
    dW_dot_Q = dW[:,0]*flake[:,0] + dW[:,1]*flake[:,1] + dW[:,2]*flake[:,2]
    
    # Steaky part of the RHS
    RHS0 = Q_gradU_Q + dW_dot_Q/dt
    
    RHS = RHS0.copy()
    
    # Matrix to invert the problem
    b = np.array([[ 5., -1., -1.],
                  [-1.,  5., -1.],
                  [-1., -1.,  5.]])/9

    dQ = np.zeros_like(flake)
    tensions = np.zeros(3)
    
    for i in range(_maxiter):
        if _verbose: print(f"From evolve: iteration {i}")
        # Solve the system
        for i in range(3):
            tensions[i] = b[i,0]*RHS[0] + b[i,1]*RHS[1] + b[i,2]*RHS[2]
     
        # Compute dQ
        for i in range(3):
            dQ[i] = dt*(Q_gradU[i] + tensions[i-1]*flake[i-1] 
                    -2*tensions[i]*flake[i] + tensions[(i+1)%3]*flake[(i+1)%3]) + dW[i]

        # Compute new flake
        flake_out = flake + dQ
        
        # Compute rod square lengths
        l = np.sum(flake_out**2, axis=1)
        err = np.max(np.abs(l-1))
        if _verbose: print(f"error = {err}")
        if _debug: print(f"error at rod {np.argmax(np.abs(l-1))}")
        if err < _tol:
            # Re normalise and exit loop
            # Take normalised first vector
            Q0 = flake_out[0]/np.sqrt(np.sum(flake_out[0]**2))
            # Make a basis using second vector:
            B2 = flake_out[1] - np.sum(Q0*flake_out[1])*Q0
            B2 = B2/np.sqrt(np.sum(B2**2))
            # Build a corrected second vector:
            Q1 = -0.5*Q0 + np.sqrt(3.)/2*B2
            # And the third
            Q2 = -(Q0 + Q1)
            flake_out = np.vstack((Q0,Q1,Q2))
            #flake_out[0]=Q0
            #flake_out[1]=Q1
            #flake_out[2]=Q2
            
            
            break
        if i==_maxiter-1:
            raise RuntimeError(f"Could not converge in {_maxiter} iterations.")
        if np.isnan(dQ).any():
            raise RuntimeError(f"Convergence failed at iteration {i}: NaN detected.")
        # Otherwise update the right-hand-side and solve again
        RHS = RHS0 + 0.5/dt*np.sum(dQ**2, axis=1)

    return flake_out, tensions

def make_flakes(Nensemble):
    """Helper function to make an ensemble of randomly oriented flakes.
    
    Paramters
    ---------
    Nensemble : int
        Number of flakes to initiate
    
    Returns
    -------
    list of (3,3) ndarray
        The list of flakes
    """
    # Initial vectors
    init = np.random.standard_normal((Nensemble,3))
    init = init/np.reshape(np.sqrt(np.sum(init**2, axis=1)), (Nensemble,1))
    out = []
    for Q0 in init:
        B1 = np.random.standard_normal(3)
        B2 = B1 - np.sum(Q0*B1)*Q0
        B2 = B2/np.sqrt(np.sum(B2**2))
        # Build a corrected second vector:
        Q1 = -0.5*Q0 + np.sqrt(3.)/2*B2
        # And the third
        Q2 = -(Q0 + Q1)
        out.append(np.vstack((Q0,Q1,Q2)))
    return out 

