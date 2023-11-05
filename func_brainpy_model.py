import brainpy as bp
import brainpy.math as bm
from typing import Union
from brainpy.types import ArrayType


class DecoModel(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641,
        J: float = 0.2609,
        tau_S: Union[float, ArrayType] = 0.1,
        tau_S_trainable: bool = False,
        G: float = 1.0, 
        G_trainable: bool = True,
        w: Union[float, ArrayType] = 0.9, 
        w_trainable: bool = True,
        I: Union[float, ArrayType] = 0.0, 
        I_trainable: bool = True,
        H_x_act = 'Softplus', 
        S_init: Union[float, ArrayType] = None,
        H_init: Union[float, ArrayType] = None,
    ):
        """
        reduced-wong-wang-deco model using BrainPy 
        written by Xiaoyu Chen(chenxy_sjtu@sjtu.edu.cn) and Yixiao Feng(newtonpula@sjtu.edu.cn) in 2023-11-03
        """

        super(DecoModel, self).__init__()

        #>>> fixed parameters:
        self.num = size # network size (# of node)
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J

        #>>> time constant of S
        if tau_S_trainable:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S = bm.TrainVar(tau_S) # (1,) or (num,)
        else:
            self.tau_S = tau_S # float or (1,) (num,) bm.array

        #>>> global coupling weight
        if G_trainable:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        #>>> recurrent weights
        if w_trainable:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w = bm.TrainVar(w) # (1,) or (num,)
        else:
            self.w = w # float or (1,) (num,) bm.array

        #>>> background input
        if I_trainable:
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
            self.H_x_act = lambda x: bp.dnn.Softplus(beta = 0.154, threshold = 1e12)(270 * x - 108)
        elif H_x_act == 'AbbottChance':
            self.H_x_act = lambda x: bm.nan_to_num((270 * x - 108) / (1 - bm.exp(-0.154 * (270 * x - 108))) , nan = 1 / 0.154 )

        #>>> initial values
        if S_init is None:
            self.S_init = bm.zeros(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init)
        
        if H_init is None:
            self.H_init = bm.zeros(self.num)
        elif isinstance(H_init, float):
            self.H_init = bm.asarray(H_init * bm.ones(self.num))
        else:
            self.H_init = bm.asarray(H_init)
        
        #>>> variables
        self.S = bm.Variable(self.S_init * bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init * bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init * bm.ones((batch_size,self.num))
        self.H.value = self.H_init * bm.ones((batch_size,self.num))

    def reset_init(self,):
        self.S_init.value = bm.mean(self.S,axis=0)
        self.H_init.value = bm.mean(self.H,axis=0)

    def update(self, inp = 0):
        # update S based on H and input, noise is integrated into the input
        self.S.value = self.S + ( - self.S / self.tau_S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # hard sigmoid of S
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # compute x based on S 
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S , self.struc_conn_matrix) + self.I
        
        # get firing rate H(x) based on its input x
        self.H.value = self.H_x_act(x) 
        
        return self.S

class DecoModel_v1(bp.DynamicalSystemNS):
    def __init__(
        self, 
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641,
        J: float = 0.2609,
        tau_S: Union[float, ArrayType] = 0.1,
        G: float = 1.0, 
        w: Union[float, ArrayType] = 0.9, 
        I: Union[float, ArrayType] = 0.0, 
        H_x_act = 'AbbottChance', 
        S_init: Union[float, ArrayType] = None,
        train: list = ['G', 'w', 'I'],
        method = 'euler',
    ):
        super(DecoModel_v1, self).__init__()

        #>>> fixed parameters:
        self.num = size # network size (# of node)
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J
        
        #>>> time constant of S
        if 'tau_S' in train:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S = bm.TrainVar(tau_S) # (1,) or (num,)
        else:
            self.tau_S = tau_S # float or (1,) (num,) bm.array

        #>>> global coupling weight
        if 'G' in train:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        #>>> recurrent weights
        if 'w' in train:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w = bm.TrainVar(w) # (1,) or (num,)
        else:
            self.w = w # float or (1,) (num,) bm.array

        #>>> background input
        if 'I' in train:
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
            self.H_x_act = lambda x: bp.dnn.Softplus(beta = 0.154, threshold = 1e12)(270 * x - 108)
        elif H_x_act == 'AbbottChance':
            self.H_x_act = lambda x: bm.nan_to_num((270 * x - 108) / (1 - bm.exp(-0.154 * (270 * x - 108))) , nan = 1 / 0.154 )
        
        #>>> initial values
        if S_init is None:
            self.S_init = bm.zeros(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init)

        #>>> variables
        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 

        #>>> functions
        self.int_S = bp.odeint(self.dS, method = method)

    def dS(self, S, t):
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S , self.struc_conn_matrix) + self.I
        dS = - self.S / self.tau_S + self.gamma * (1-self.S) * self.H_x_act(x)
        return dS

    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init * bm.ones((batch_size,self.num))

    def reset_init(self,):
        self.S_init.value = bm.mean(self.S,axis=0)   

    def update(self, inp = 0):
        t = bp.share['t']
        self.S.value = self.int_S(self.S, t) + inp
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        return self.S
        

class outLinear(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        a: Union[float, ArrayType] = None, # trainable
        b: Union[float, ArrayType] = None, # trainable
        ):

        '''
        Output-linear-scaling-layer of S from RNN-layer DecoModel
        size = num, i.e., number of network size (# of node)
        a or b must be float or bm.array with (1,) or (num,) shape
        if a or b is float, then set (num,) parameters with same initialization such as [a,a,a,a,....]
        if a or b is None, then it will not be included in scaling
        '''

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

