import brainpy as bp
import brainpy.math as bm
from typing import Union
from brainpy.types import ArrayType


class DecoModel(bp.DynamicalSystemNS):
    def __init__(
    self,
    size,
    tau_S: Union[float, ArrayType] = 0.1,
    gamma: float = 0.641,
    J: float = 0.2609,
    I_0: Union[float, ArrayType] = None, 
    G: float = 1, # trainable
    w: Union[float, ArrayType] = 0.9, 
    struc_conn_matrix: ArrayType = None,
    S_init = None,
    H_x_init = None,
    batch_size = 1,
    LFP_a = None,
    LFP_b = None,
    ):
    
        super(DecoModel, self).__init__()
        
        #>>> Model setted parameters:
        self.num = size
        self.tau_S = tau_S
        self.gamma = gamma
        self.J = J
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix)
        
        #>>> activation func
        self.H_x_act = bp.dnn.Softplus(beta=0.154, threshold=1e12 )
        
        #>>> Trainable weights
        self.G = bm.TrainVar(G)
        
        if isinstance(w, float):
            w = w * bm.ones(self.num)
            self.w = bm.TrainVar(w) # (num,)
        
        if I_0 is None:
            self.I_0 = 0.0
        else:
            if isinstance(I_0, float):
                I_0 = I_0 * bm.ones(self.num)
                self.I_0 = bm.TrainVar(I_0) # (num,)
        
        if LFP_a is None:
            self.LFP_a = 1
        else:
            self.LFP_a = bm.TrainVar(LFP_a)
        
        
        if LFP_b is None:
            self.LFP_b = 0
        else:
            self.LFP_b = bm.TrainVar(LFP_b)
        
        
        #>>> Variables:
        if S_init is not None:
            self.S_init = bm.asarray(S_init)
        else:
            self.S_init = 0.0
        
        if H_x_init is not None:
            self.H_x_init = bm.asarray(H_x_init)
        else:
            self.H_x_init = 0.0
        
        self.S   = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H_x = bm.Variable(self.H_x_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value   = self.S_init*bm.ones((batch_size,self.num))
        self.H_x.value = self.H_x_init*bm.ones((batch_size,self.num))
    
    def update(self, stim = 0):
        # update S based on H_x
        self.S.value = self.S + ( -1 / self.tau_S * self.S + self.gamma * (1-self.S) * self.H_x) * bm.dt + stim
        
        # hard sigmoid
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # input x based on S
        x = self.J * bm.multiply(self.w, self.S) + self.J * self.G * bm.matmul(self.S , self.struc_conn_matrix) + self.I_0 # + stim
        
        # firing rate H_x based on x
        self.H_x.value = self.H_x_act(270*x-108) 
        
        # firing rate to LFP
        LFP = bm.multiply(self.LFP_a, self.S) + self.LFP_b # if 1*x+0, then no LFP conversion
    
        return LFP
