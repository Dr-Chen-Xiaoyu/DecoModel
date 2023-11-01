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
    G: float = 1.0, # trainable
    w: Union[float, ArrayType] = 0.9, # trainable
    I: Union[float, ArrayType] = None, # trainable
    out_scale_a: float = None, # trainable
    out_scale_b: float = None, # trainable
    H_x_act = 'Softplus', 
    out_act = 'linear', 
    S_init: Union[float, ArrayType] = None,
    H_init: Union[float, ArrayType] = None,
    ):
    '''
    reduced-wang-wong-deco model using BrainPy 
    written by Xiaoyu Chen(chenxy_sjtu@sjtu.edu.cn) and Yixiao Feng() 2023-11

    
    '''

        super(DecoModel, self).__init__()
        

        #>>> Model setted parameters:
        self.num = size
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix)
        self.gamma = gamma
        self.J = J
        self.tau_S = tau_S

        
        #>>> Trainable weights
        self.G = bm.TrainVar(G)
        

        if w is None:
            self.w = 0.0
        elif isinstance(w, float):
            self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,)
        else:
            self.w = bm.TrainVar(w) # (num,)


        if I is None:
            self.I = 0.0
        elif isinstance(I, float):
            self.I = bm.TrainVar(I * bm.ones(self.num)) # (num,)
        else:
            self.I = bm.TrainVar(I) # (num,)


        if out_scale_a is None:
            self.out_scale_a = 1.0
        else:
            self.out_scale_a = bm.TrainVar(out_scale_a)
        
        
        if out_scale_b is None:
            self.out_scale_b = 0.0
        else:
            self.out_scale_b = bm.TrainVar(out_scale_b)
        
        
        #>>> activation func
        if callable(H_x_act):
            self.H_x_act = H_x_act
        elif H_x_act == 'Softplus':
            self.H_x_act = lambda x: bp.dnn.Softplus(beta=0.154, threshold=1e12)(270*x-108)
        elif H_x_act == 'AbbottChance':
            self.H_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) )

        if out_act == 'linear':
            self.out_act = lambda x: bm.multiply(self.out_scale_a, x) + self.out_scale_b
            # if 1*x+0, then no scaling conversion
        elif out_act == 'Balloon':
            pass # 


        #>>> Variables:
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
        

        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init*bm.ones((batch_size,self.num))
        self.H.value = self.H_init*bm.ones((batch_size,self.num))
    
    def reset_init(self,):
        self.S_init.value = bm.mean(self.S,axis=0)
        self.H_init.value = bm.mean(self.H,axis=0)

    def update(self, inp = 0):
        # update S based on H and input
        self.S.value = self.S + ( -1 / self.tau_S * self.S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # hard sigmoid of S
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # summary x based on S
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S , self.struc_conn_matrix) + self.I
        
        # get firing rate H(x) based on its input x
        self.H.value = self.H_x_act(x) 
        
        # get output
        out = self.out_act(self.S)

        return out
