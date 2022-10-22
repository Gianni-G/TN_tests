import numpy as np
import quimb.tensor as qtn
import util
import seaborn as sns; sns.set(
    rc = {'figure.figsize':(18,2)},
    font="monospace"
    )
    

class MPS():
    def __init__(
        self,
        tensor,
        elements = None,
        bond_size = 4,
        method = "truncate", # "svd", "full"
        compress = False, # only relevant if method == "full"
        ):

        if method == "from_mps":
            self.order = tensor.L
            self.e_len = tensor.ind_sizes()["k0"]
        else:
            self.order = tensor.ndim
            self.e_len = tensor.shape[0]

        if elements != None:
            assert len(elements) == self.e_len, "The number of elements does not agree with the tensor dimensions"
            self.elements = elements
        else:
            self.elements = [str(i) for i in range(self.e_len)]
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}
        

        self.method = method

        if method == "from_mps":
            self.TN = tensor

        if method == "truncate":
            self.TN = qtn.tensor_1d.MatrixProductState.from_dense(tensor,[self.e_len]*self.order, max_bond=bond_size)

        elif method == "full":
            self.TN = qtn.tensor_1d.MatrixProductState.from_dense(tensor,[self.e_len]*self.order)
            if compress:
                for i in range(self.order-1):
                    qtn.tensor_compress_bond(self.TN.tensors[i], self.TN.tensors[i+1], max_bond=bond_size, absorb='both')

        elif method == "svd":
            M_T = qtn.Tensor(M, inds=[f"k{i}" for i in range(self.order)],tags='M')
            self.TN = M_T.split(
                left_inds=["k0"],
                right_inds=[f"k{i}" for i in range(1,self.order)],
                max_bond=bond_size,
                ltags="I0",
                rtags="I1",
                bond_ind="0-1",
                )
            for i in range(1,self.order-1):
                self.TN.split_tensor(
                    tags= f"I{i}",
                    left_inds=([f"{i-1}-{i}",f"k{i}"]),
                    right_inds = [f"k{j}" for j in range(i+1,self.order)],
                    max_bond=bond_size,
                    ltags=f"I{i}",
                    rtags=f"I{i+1}",
                    bond_ind=f"{i}-{i+1}",
                    )

        self.TN.add_tag("MPS")
        self.loaded = False

    def from_mps(self,mps,elements=None):
        self.order = mps.L
        self.e_len = mps.ind_sizes()["k0"]

        if elements != None:
            assert len(elements) == self.e_len, "The number of elements does not agree with the tensor dimensions"
            self.elements = elements
        else:
            self.elements = [str(i) for i in range(self.e_len)]
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}
        
        self.TN = mps
        self.TN.add_tag('MPS')
        self.loaded = False


    def load(self,expression):

        if len(expression)> self.order:
            print(f"Warning: the expression '{expression}' is not of the same size as the tensor network. Trimming as necessary")

        self.expression = expression[:self.order]

        expression_i = [self.elements_dict[char] for char in self.expression]

        expression_input = np.eye(self.e_len)[expression_i]
        node_inputs = [qtn.Tensor(input_v, inds=[f"k{i}"],tags={f"{self.expression[i]}",f"i{i}"}) for i,input_v in enumerate(expression_input)]

        self.input = qtn.TensorNetwork(node_inputs)

        self.TN = qtn.TensorNetwork(self.TN["MPS"]+self.input.tensors)

        self.loaded = True
    
    def draw(self):
        self.TN.draw(
            figsize=(10, 10),
            initial_layout="shell",
            node_size = 4000,
            # show_tags= False,
            show_inds='bond-size',
            color = ["MPS"] + (list(self.input.tags) if self.loaded else []),
            fix = dict(
                [(f"I{i}",(i,0)) for i in range(self.order)]+
                [(f"i{i}",(i,-1)) for i in range(self.order)] + [(f"k{i}",(i,-1)) for i in range(self.order)]
                ),

            node_scale = 20,
            edge_scale = 1.5,
            font_size_inner = 12,
            )
    
    def contract(
        self,
        skip=None,
        draw=False,
        draw_vmax = .1,
        draw_len = 20,
        ):

        if skip == None:
            self.contraction = self.TN ^ ...
        else:
            self.contraction = self.TN ^ set(self.TN.tags)-{f"i{skip}",self.expression[skip]}
            self.contraction = self.contraction["MPS"]
        
        if draw:
            util.plot((-np.sort(-self.contraction.data))[:draw_len],[self.elements[i] for i in np.argsort(-self.contraction.data)][:draw_len],vmax = draw_vmax)


class MPS_G():
    def __init__(self, mps:MPS):
        self.original = mps
        self.elements = self.original.elements
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}
        self.e_len = self.original.e_len
        
    
    def load(self,expression):

        self.expression = expression

        expression_i = [self.elements_dict[char] for char in self.expression]

        expression_input = np.eye(self.e_len)[expression_i]
        node_inputs = [qtn.Tensor(input_v, inds=[f"k{i}"],tags={f"{self.expression[i]}",f"i{i}"}) for i,input_v in enumerate(expression_input)]

        self.input = qtn.TensorNetwork(node_inputs)


        self.order = len(expression)
        self.it = self.original.TN["I0"].copy()
        self.it.modify(inds=("k0","k0-1"), tags="C0")
        self.ft = self.original.TN[f"I{self.original.order-1}"].copy()
        self.ft.modify(inds=(f"k{self.order-2}-{self.order-1}",f"k{self.order-1}"),tags=f"C{self.order-1}")

        cut = int(self.original.order/2)
        MPS = qtn.TensorNetwork(self.original.TN["MPS"])
        MPS_H = MPS.H
        MPS_H.drop_tags("MPS")
        MPS_H.add_tag("MPSh")
        cut_tensor = MPS_H[f"I{cut}"]
        cut_tensor.reindex({ind:(ind if ind!= f"k{cut}" else "cut") for i,ind in enumerate(cut_tensor.inds)}, inplace=True)
        MPS_tn = MPS & MPS_H
        MPS_compress = MPS_tn^...

        self.ct = []
        for i in range(1,self.order-1):
            new_tensor = self.original.TN[f"I{int(self.original.order/2)}"].copy()
            new_tensor.modify(inds=(f'k{i-1}-{i}', f'k{i}', f'k{i}-{i+1}'),tags=f"C{i}")
            self.ct.append(new_tensor)
        self.TN=qtn.TensorNetwork(self.ct+[self.it,self.ft])
        self.TN.add_tag("MPS")

        self.TN = qtn.TensorNetwork(self.TN["MPS"]+self.input.tensors)

        self.loaded = True

    def draw(self):
        self.TN.draw(
            figsize=(13, 15),
            initial_layout="shell",
            node_size = 2000,
            show_tags= True,
            show_inds='bond-size',
            legend=False,
            color = ["MPS"] + list(self.input.tags),
            fix = dict(
                [(f"C{i}",(i,0)) for i in range(self.order)]+
                [(f"i{i}",(i,-2)) for i in range(self.order)] + [(f"k{i}",(i,-1)) for i in range(self.order)]
                ),

            node_scale = 1,
            edge_scale = 1,
            font_size_inner = 12,
            )

    def contract(
        self,
        skip=None,
        draw=False,
        draw_vmax = .1,
        draw_len = 20,
        ):

        if skip == None:
            self.contraction = self.TN ^ ...
        else:
            self.contraction = self.TN ^ set(self.TN.tags)-{f"i{skip}",self.expression[skip]}
            self.contraction = self.contraction["MPS"]
        
        if draw:
            util.plot((-np.sort(-self.contraction.data))[:draw_len],[self.elements[i] for i in np.argsort(-self.contraction.data)][:draw_len],vmax = draw_vmax)

