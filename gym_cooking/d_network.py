from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNetworkFC(nn.Module):
    def __init__(self, obs_dim, obs_u_dim, mid_dim1, mid_dim2, gnn_hdim1, gnn_hdim2, gnn_out_dim, num_acts, device):
        super(AgentNetworkFC, self).__init__()
        self.obs_dim = obs_dim
        self.obs_u_dim = obs_u_dim
        self.mid_dim1 = mid_dim1
        self.mid_dim2 = mid_dim2
        self.gnn_hdim1 = gnn_hdim1
        self.gnn_hdim2 = gnn_hdim2
        self.gnn_out_dim = gnn_out_dim
        self.num_acts = num_acts
        self.device = device

        self.linear_obs_rep = nn.Sequential(
            nn.Linear(obs_dim + obs_u_dim, mid_dim1),
            nn.ReLU(),
            nn.Linear(mid_dim1, mid_dim2),
            nn.ReLU(),
            nn.Linear(mid_dim2, mid_dim2)
        ).to(self.device)

        self.gnn = nn.Sequential(
            nn.Linear(mid_dim2, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, gnn_out_dim)
        )

        self.act_logits = nn.Linear(gnn_out_dim, num_acts).to(self.device)
        self.v_compute = nn.Linear(gnn_out_dim, 1).to(self.device)


    def forward(self, input):
        n_inp = self.linear_obs_rep(input.to(self.device))
        n_out = self.gnn(n_inp)
        act_logits = self.act_logits(n_out)

        return act_logits


def fc_network(layer_dims, init_ortho=True):
    """
    Builds a fully-connected NN with ReLU activation functions.

    """
    if init_ortho: 
        init = init_orthogonal
    else:
        init = lambda m: m

    network = nn.Sequential(
                *chain(
                    *((init(nn.Linear(layer_dims[i], layer_dims[i+1])),
                       nn.ReLU())
                      for i in range(len(layer_dims)-1))
                    ),
                )
    del network[-1]  # remove the final ReLU layer
    return network

def init_orthogonal(m):
    nn.init.orthogonal_(m.weight)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)
    return m
