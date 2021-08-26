import numpy as np
import lib.fiber_solvers as solvers
from scipy.integrate import solve_ivp

# Input functions
def S_repressor(x,n):
    return 1/(1+x**(n))

def S_activator(x,n):
    return 1 - 1/(1+x**(n))

def T_repressor(x,y,n):
    return 1/(1+(x+y)**(n))

def S_prime1(x,y,n):
    z = x+y
    return -(2*z)/((1+z**2)**2)

def S_prime2(x,y,n):
    z = x+y
    return -2*(1-3*z**2)/((1+z**2)**3)

def S_prime3(x,y,n):
    z = x+y
    return (24*z*(1-z**2))/((1+z**2)**4)

def T_activator(x,y,n):
    return 1 - T_repressor(x,y,n)

def calculate_Io_fff(params, fiber="UNSAT"):
    if fiber=="UNSAT":
        return params[1]*((3/8)*((1-params[4])/(1+params[4])) - 0.75)
    else:
        return -params[1]*((3/8)*((1-params[4])/(1+params[4])) + 0.25)
    
def calculate_Io_fibo(params, g="repressor"):
    if g=="repressor":
        par_ratio = (params[1]*params[3])/(params[0]*params[2])
        return -params[1]*(par_ratio*(1/np.sqrt(3)) + 0.75)
    if g=="activator":
        pass

def run_fff_test(params, initial, regul_comb, to=0, tf=20, 
                 I_delta=2.0, npoints_I=50, I_factor=2, I_sample=1.0):
    '''
        Quick test for given parameters and defined regulation combination.
        
        Args:
            params:
                [delta, gamma, alpha, beta, sigma]
            initial:
            input_range:
            regulation_comb:
                2-dimensional tuple x signalling the type of regulation within the 
                fiber and for the external regulations. x[0] for \tilde{f} and x[1]
                for \tilde{g}.
    '''
    # Set parameters for the FFF solver. 
    fff = solvers.FFF_solver(params)
    fff.set_initial(initial)
    fff.set_regulations(regul_comb[0], regul_comb[1])
    
    # Sample solution in case we desire to plot an example of the dynamics.
    sample_solution = fff.solve_eq(to,tf,I_sample,dense=True)
    
    mode = "UNSAT" if regul_comb[0]==1 else "SAT"
    Io = calculate_Io_fff(params, fiber=mode)
    I_min, I_max = Io - I_factor*I_delta, Io + I_factor*I_delta
    input_range = np.linspace(I_min, I_max, npoints_I)
    
    x1p = []
    x2p = []
    x3p = []
    for index, inp in enumerate(input_range):
        sol = fff.solve_eq(to,tf,inp,dense=True)
        x1p.append(sol.y[1][-1])
        x2p.append(sol.y[3][-1])
        x3p.append(sol.y[5][-1])
        
    return (sample_solution, input_range, (np.array(x1p), np.array(x2p), np.array(x3p)), Io)

def run_2fibo_test(params, initial, regul_comb, to=0, tf=20, I_delta=2.0, 
                   npoints_I=50, I_factor=2, I_sample=1.0, input_node=[1.0,0.0,0.0,0.0]):
    '''
        Quick test for given parameters and defined regulation combination.
        
        Args:
            params:
                [delta, gamma, alpha, beta]
            initial:
            input_range:
            regulation_comb:
                2-dimensional tuple x signalling the type of regulation within the 
                fiber and for the external regulations. x[0] for \tilde{f} and x[1]
                for \tilde{g}.
            to:
            tf:
            I_delta:
            npoints_I:
            I_factor:
            I_sample:
            input_node:
    '''
    # Set parameters for the FIBO solver. 
    fibo = solvers.FIBO2_solver(params)
    fibo.set_initial(initial)
    fibo.set_regulations(regul_comb[0], regul_comb[1], regul_comb[2])
    
    # Sample solution in case we desire to plot an example of the dynamics.
    sample_solution = fibo.solve_eq(to,tf,I_sample,input_node=input_node,dense=True)
    
    Io = I_sample
    I_min, I_max = Io - I_factor*I_delta, Io + I_factor*I_delta
    input_range = np.linspace(I_min, I_max, npoints_I)
    
    x1p = []
    x2p = []
    x3p = []
    x4p = []
    for index, inp in enumerate(input_range):
        sol = fibo.solve_eq(to,tf,inp,input_node=input_node,dense=True)
        x1p.append(sol.y[1][-1])
        x2p.append(sol.y[3][-1])
        x3p.append(sol.y[5][-1])
        x4p.append(sol.y[7][-1])
    
    protein_conc = (np.array(x1p), np.array(x2p), np.array(x3p), np.array(x4p))
    return (sample_solution, input_range, protein_conc, Io)

    