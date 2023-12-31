'''
BrainPy Models written by Xiaoyu Chen(chenxy_sjtu@sjtu.edu.cn) and Yixiao Feng(newtonpula@sjtu.edu.cn) in 2023-11-03
'''


import brainpy as bp
import brainpy.math as bm
from jax import vmap
from typing import Union,Callable
from brainpy.types import ArrayType


def AbbottChance(inp, a=270, b=108, d=0.154, epsilon=1e-2):
    x=a*inp-b
    out=bm.ifelse( 
        bm.abs(x)<=epsilon, 
        operands=x, 
        branches=(lambda x: 0.5*x + 1/d,
                  lambda x: x/(1 - bm.exp(-d*x)), )
        )
    return out
    

class DecoModel(bp.DynamicalSystemNS):
    """
    reduced-wong-wang-deco model using BrainPy 

    tau_S, w and I should be float or (1,) or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    TrainVar_list should be a list such as ['tau_S','G','w','I'] to specify if this parameter is trainable
    
    S_init and H_init should be float or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    S(gating state of model) and H(firing rate) will be initialized by S_init and H_init into (batch_size,num) with broadcasting along batch axis  

    """
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641, # kinetic parameter
        J: float = 0.2609, # synaptic coupling
        tau_S: Union[float, ArrayType] = 0.1, # time constant
        G: float = 1.0, # global coupling weight
        w: Union[float, ArrayType] = 0.9, # recurrent weights
        I: Union[float, ArrayType] = 0.0, # background inputs (intercepts)
        TrainVar_list = ['G','w','I'],
        H_x_act: Union[str, Callable] = vmap(vmap(AbbottChance, out_axes=0, in_axes=0), out_axes=0, in_axes=0), # or other callable activation function or 'Softplus' or 'AbbottChance'
        S_init: Union[float, ArrayType] = None, # initial S
        H_init: Union[float, ArrayType] = None, # initial H (firing rate)
    ):
        

        super(DecoModel, self).__init__()
        

        #>>> fixed parameters:
        self.num = size # number of network size (# of node)
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J
        

        #>>> Trainable weights
        if 'tau_S' in TrainVar_list:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S = bm.TrainVar(tau_S) # (1,) or (num,)
        else:
            self.tau_S = tau_S # float or (1,) (num,) bm.array

        if 'G' in TrainVar_list:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        if 'w' in TrainVar_list:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w = bm.TrainVar(w) # (1,) or (num,)
        else:
            self.w = w # float or (1,) (num,) bm.array

        if 'I' in TrainVar_list:
            if isinstance(I, float):
                self.I = bm.TrainVar(I * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.I = bm.TrainVar(I) # (1,) or (num,)
        else:
            self.I = I # float or (1,) (num,) bm.array
        
            
        #>>> activation function
        if callable(H_x_act):
            self.H_x_act = H_x_act
        elif H_x_act == 'Softplus':
            self.H_x_act = lambda x: bp.dnn.Softplus(beta=0.154, threshold=1e12)(270*x-108)
        elif H_x_act == 'AbbottChance': # the original AbbottChance
            self.H_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )


        #>>> Variables:
        if S_init is None:
            self.S_init = bm.zeros(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init) # (num,)
        

        if H_init is None:
            self.H_init = bm.zeros(self.num)
        elif isinstance(H_init, float):
            self.H_init = bm.asarray(H_init * bm.ones(self.num))
        else:
            self.H_init = bm.asarray(H_init) # (num,)
        

        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init*bm.ones((batch_size,self.num))
        self.H.value = self.H_init*bm.ones((batch_size,self.num))
    

    def reset_init(self,):
        self.S_init.value = bm.mean(self.S,axis=0)
        self.H_init.value = bm.mean(self.H,axis=0)


    def update(self, inp = 0):
        # update S based on H and input, noise is integrated into the input
        self.S.value = self.S + ( - self.S / self.tau_S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # hard sigmoid of S
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # get x based on S
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S , self.struc_conn_matrix) + self.I
        
        # get firing rate H(x) based on its input x
        self.H.value = self.H_x_act(x) 
        
        return self.S


class outLinear(bp.DynamicalSystemNS):
    '''
    Output-linear-scalling-layer of S from RNN-layer DecoModel
    size = num, i.e., number of network size (# of node)
    a or b must be float or bm.array with (1,) or (num,) shape
    if a or b is float, then set (num,) paramters with same initialization such as [a,a,a,a,....]
    if a or b is None, then it will not be included in scalling
    '''
    def __init__(
        self,
        size: int,
        a: Union[float, ArrayType] = None, # trainable
        b: Union[float, ArrayType] = None, # trainable
    ):

        super(outLinear, self).__init__()

        if a is None:
            self.a = 1.0
        elif isinstance(a,float):
            self.a = bm.TrainVar(a*bm.ones((size))) # (num,) with same initialization
        else:
            self.a = bm.TrainVar(a) # (1,) or (num,)

        if b is None:
            self.b = 0.0
        elif isinstance(b,float):
            self.b = bm.TrainVar(b*bm.ones((size))) # (num,) with same initialization
        else:
            self.b = bm.TrainVar(b) # (1,) or (num,)

    def update(self, inp = 0):
        return bm.multiply(self.a, inp) + self.b


class outBalloon(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        batch_size: int = 1,
        rou: float = 0.34, # ρ = 0.34 is the resting oxygen extraction fraction, the p_constant
        tau: float = 0.98, # hemodynamic transit time τ = 0.98 s
        kappa: float = 0.65, # The kinetic parameters rate of signal decay κ
        gamma: float = 0.41, # rate of elimination
        alpha: float = 0.32, # Grubb’s exponent
        v_0: float = 0.02,# resting blood volume fraction
        B_0: float = 3.0, # B0, which is 3T in the HCP dataset
        r_0: float = 110, # the intravascular relaxation rate is r0 = 110 Hz
        e: float = 0.47, # the ratio between intravascular and extravascular MR signal is e = 0.47
        TE: float = 0.0331, # The echo time TE = 33.1 ms in the HCP dataset
        ):

        super(outBalloon, self).__init__()
        '''
        Output-nonlinear-scalling-layer (Balloon dynamics) of S from RNN-layer DecoModel, i.e., turning postsynaptic gating variable S to BOLD signal in an element-wise fashion.
        Balloon-Windkessel Hemodynamic model is used based on the codes from https://www.nature.com/articles/s41467-021-26704-y with equations and default parameters from https://www.science.org/doi/10.1126/sciadv.aat7854
        
        size = num, i.e., number of network size (# of node)
        '''

        self.rou = rou
        self.tau = tau
            
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
 
        self.v_0 = v_0

        eta_0 = 28.265 * B_0 # ϑ0 = 28.265B0 is the frequency offset at the outer surface of magnetized vessels and depends on the main magnetic field strength B0
        self.k_1 = 4.3 * eta_0 * rou * TE
        self.k_2 = e * r_0 * rou * TE
        self.k_3 = 1-e
            
        self.F_0 = bm.Variable(0*bm.ones((batch_size,size)), batch_axis = 0)
        self.F_1 = bm.Variable(1*bm.ones((batch_size,size)), batch_axis = 0) 
        self.F_2 = bm.Variable(1*bm.ones((batch_size,size)), batch_axis = 0) 
        self.F_3 = bm.Variable(1*bm.ones((batch_size,size)), batch_axis = 0) 


    def update(self,S=0):
        '''
        input   S:  batch_size*size matrix represents synaptic gating variable
                batch_size is the number of batch
                size is the number of nodes

        output  S_BOLD: same shape matrix for BOLD signal output (element-wise Hemodynamic transformation)
        '''
                
        # gating2bold_derivative：
        dF_0 = S - self.kappa * self.F_0 - self.gamma * (self.F_1 -1)
        dF_1 = self.F_0
        dF_2 = 1 / self.tau * (self.F_1 - self.F_2**(1/self.alpha))
        dF_3 = 1 / self.tau * (self.F_1 / self.rou * (1-(1-self.rou)**(1/self.F_1)) - self.F_3 * self.F_2 **(1/self.alpha - 1))

        # forward-eular
        self.F_0.value = self.F_0 + dF_0*bm.dt
        self.F_1.value = self.F_1 + dF_1*bm.dt
        self.F_2.value = self.F_2 + dF_2*bm.dt
        self.F_3.value = self.F_3 + dF_3*bm.dt
        
        # caculate and return BOLD signal
        v_t = self.F_2
        q_t = self.F_3
        return 100/self.rou*self.v_0 * (self.k_1 * (1 - q_t) + self.k_2 * (1 - q_t / v_t) + self.k_3 * (1 - v_t))

