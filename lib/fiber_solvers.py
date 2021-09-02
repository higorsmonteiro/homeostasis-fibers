import numpy as np
from scipy.integrate import solve_ivp

class FFF_solver:
    def __init__(self, params):
        self.delta = params[0]
        self.gamma = params[1]
        self.alpha = params[2]
        self.beta = params[3]
        self.sigma = params[4]
        
    def set_initial(self, x0):
        '''
            Set the initial values to the coordinates' vector through 'x0'.
            
            Args:
                x0:
                    vector of coordinates following the pattern 
                    x0 = [x_1R, x_1P, x_2R, x_2P, ...]
        '''
        self.x0 = np.array(x0).copy()
        
    def set_regulations(self, f=1, g=1):
        '''
            Set the type of regulations within the fiber and the type of
            the external regulations coming out the fiber.
            
            Args:
                f:
                    {0,1} - If f = 0, then we have activator regulations, while
                    if f = 1, then the regulations are repressors.
                g:
                    {0,1,2,3} - doc TO DO.
        '''
        if f==0:
            self.f = S_activator
        else:
            self.f = S_repressor
            
        # for now
        if g==0:
            self.g = T_activator
        elif g==1:
            self.g = T_repressor
        elif g==2:
            self.g = lambda x,y,n: S_activator(x,n)*S_repressor(y,n)
        else:
            self.g = lambda x,y,n: S_repressor(x,n)*S_activator(y,n)
        #self.g = S_repressor
        
    def verify_conditions(self, Io):
        '''
        
        '''
        pass
            
            
        
    def solve_eq(self, t0, tf, Inp, hill_exp=2, dense=False):
        '''
            Solve the FFF circuit according to the parsed parameters and regulations
            using the 4th-order Runge-Kutta method.
            
            Args:
                t0:
                tf:
                stepsize:
                Inp:
                hill_exp:
                dense:
        '''
        func = lambda t,x: np.array([-self.delta*x[0]+self.gamma*self.f(x[1],hill_exp) + Inp,
                                 -self.alpha*x[1] + self.beta*x[0],
                                 -self.delta*x[2]+self.gamma*self.f(x[1],hill_exp) + self.sigma*Inp,
                                 -self.alpha*x[3] + self.beta*x[2],
                                 -self.delta*x[4] + self.gamma*self.g(x[1],x[3],hill_exp),
                                 -self.alpha*x[5] + self.beta*x[4]])
        
        return solve_ivp(func, [t0,tf], self.x0, method="RK45", dense_output=dense)
    
class FFF_modified_solver:
    def __init__(self, params):
        self.delta = params[0]
        self.gamma = params[1]
        self.alpha = params[2]
        self.beta = params[3]
        self.sigma = params[4]
        
    def set_initial(self, x0):
        '''
        
        '''
        self.x0 = np.array(x0).copy()
        
    def set_regulations(self, f=1, g=1):
        '''
            Set the type of regulations within the fiber and the type of
            the external regulations coming out the fiber.
            
            Args:
                f:
                    {0,1} - If f = 0, then we have activator regulations, while
                    if f = 1, then the regulations are repressors.
                g:
                    {0,1,2,3} - doc TO DO.
        '''
        if f==0:
            self.f = T_activator
        else:
            self.f = T_repressor
            
        # for now
        if g==0:
            self.g = T_activator
        elif g==1:
            self.g = T_repressor
        elif g==2:
            self.g = lambda x,y,n: S_activator(x,n)*S_repressor(y,n)
        else:
            self.g = lambda x,y,n: S_repressor(x,n)*S_activator(y,n)
    
    def solve_eq(self, t0, tf, Inp, input_node=[1.0, 0.0, 0.0, 0.0], hill_exp=2, dense=False):
        '''
            Solve the FFF circuit according to the parsed parameters and regulations
            using the 4th-order Runge-Kutta method.
            
            Args:
                t0:
                tf:
                stepsize:
                Inp:
                hill_exp:
                dense:
        '''
        func = lambda t,x: np.array([-self.delta*x[0],
                                     -self.alpha*x[1] + self.beta*x[0] + Inp*input_node[0],
                                     -self.delta*x[2] + self.gamma*self.f(x[1], x[3], hill_exp),
                                     -self.alpha*x[3] + self.beta*x[2] + Inp*input_node[1],
                                     -self.delta*x[4] + self.sigma*self.gamma*self.f(x[1], x[3], hill_exp),
                                     -self.alpha*x[5] + self.beta*x[4] + Inp*input_node[2],
                                     -self.delta*x[6] + self.gamma*self.g(x[3], x[5], hill_exp),
                                     -self.alpha*x[7] + self.beta*x[6] + Inp*input_node[3]])
        
        return solve_ivp(func, [t0,tf], self.x0, method="RK45", dense_output=dense)
    
class FIBO2_solver:
    def __init__(self, params):
        self.delta = params[0]
        self.gamma = params[1]
        self.alpha = params[2]
        self.beta = params[3]
        
    def set_initial(self, x0):
        '''
            Set the initial values to the coordinates' vector through 'x0'.
            
            Args:
                x0:
                    vector of coordinates following the pattern 
                    x0 = [x_1R, x_1P, x_2R, x_2P, ...]
        '''
        self.x0 = np.array(x0).copy()
        
    def set_regulations(self, f=0, g=0, h=0):
        '''
            Set the type of regulations within the fiber and the type of
            the external regulations coming out the fiber.
            
            Args:
                f:
                    {0,1} - If f = 0, then we have activator regulations, while
                    if f = 1, then the regulations are repressors.
                g:
                    {0,1} - doc TO DO.
                h:
                    {0,1} - doc TO DO.
        '''
        if f==0:
            self.f = S_activator
        else:
            self.f = S_repressor
            
        if g==0:
            self.g = T_activator
        elif g==1:
            self.g = T_repressor
            
        if h==0:
            self.h = T_activator
        elif h==1:
            self.h = T_repressor
            
    def solve_eq(self, t0, tf, Inp, input_node=[1.0,0.0,0.0,0.0], hill_exp=2, dense=False):
        '''
            Solve the 2-Fibonacci circuit according to the parsed parameters and 
            regulations using the 4th-order Runge-Kutta method.
            
            Args:
                t0:
                tf:
                Inp:
                input_node:
                hill_exp:
                dense:
        '''
        func = lambda t,x: np.array([-self.delta*x[0]+self.gamma*self.f(x[3],hill_exp) + 
                                     Inp*input_node[0],
                                     -self.alpha*x[1] + self.beta*x[0],
                                     -self.delta*x[2]+self.gamma*self.g(x[1], x[3], hill_exp) +
                                     Inp*input_node[1],
                                     -self.alpha*x[3] + self.beta*x[2],
                                     -self.delta*x[4]+self.gamma*self.g(x[1], x[3], hill_exp) +
                                     Inp*input_node[2],
                                     -self.alpha*x[5] + self.beta*x[4],
                                     -self.delta*x[6]+self.gamma*self.h(x[3], x[5], hill_exp) +
                                     Inp*input_node[3],
                                     -self.alpha*x[7]+self.beta*x[6]])
        
        return solve_ivp(func, [t0,tf], self.x0, method="RK45", dense_output=dense)
    
# Input functions
def S_repressor(x,n):
    return 1/(1+x**(n))

def S_activator(x,n):
    return 1 - 1/(1+x**(n))

def T_repressor(x,y,n):
    return 1/(1+(x+y)**(n))

def T_activator(x,y,n):
    return 1 - T_repressor(x,y,n)