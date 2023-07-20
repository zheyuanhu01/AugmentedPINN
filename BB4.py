import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import argparse
import scipy.io
import sympy as sy
from tqdm import tqdm
import h5py

parser = argparse.ArgumentParser(description='XPINN Project')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--PINN', type=bool, default=False) # Train or not?
parser.add_argument('--XPINN', type=bool, default=False)
parser.add_argument('--SXPINN', type=bool, default=False)
parser.add_argument('--SXPINN_Prior', type=str, default="XPINN") # XPINN, MPINN, PINN
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--lr', type=float, default=8e-4)

parser.add_argument('--PINN_h', type=int, default=20)
parser.add_argument('--PINN_L', type=int, default=10)
parser.add_argument('--XPINN_h', type=int, default=20)
parser.add_argument('--XPINN_L', type=int, default=4)

parser.add_argument('--SXPINN_G_h', type=int, default=20)
parser.add_argument('--SXPINN_G_L', type=int, default=2)
parser.add_argument('--SXPINN_E_h', type=int, default=18)
parser.add_argument('--SXPINN_E_L', type=int, default=3)
parser.add_argument('--SXPINN_h_h', type=int, default=20)
parser.add_argument('--SXPINN_h_L', type=int, default=4)
parser.add_argument('--SXPINN_gate_trainable', type=bool, default=True)

parser.add_argument('--PINN_weights', type=list, default=[1, 20])
parser.add_argument('--XPINN_weights', type=list, default=[1, 0, 20, 20, 1]) # [1, 0, 20, 20, 1]; [1, 1, 20, 20, 0]
## R; R-interface; B; B-interface; R-additional-interface

parser.add_argument('--plot_error', type=bool, default=False)
parser.add_argument('--plot_loss', type=bool, default=False)
parser.add_argument('--save_loss', type=bool, default=False)
parser.add_argument('--save_error', type=bool, default=False)
parser.add_argument('--save_gate', type=bool, default=False)

args = parser.parse_args()
print(args)

torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
device = torch.device(args.device)

def load_data_BB(x0=-10, x1=15, t0=-3, t1=2, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    p = 1; q = 1; beta = 1; p1 = 2; q1 = p1*(2*q+p1*p+2*p**2)/(2*p);
    x, t = sy.symbols('x, t'); w = p * x + q * t + 0.5 * sy.log(1 + sy.exp(p1 * x + q1 * t));
    u1 = w.diff(x) / 2; u0 = (2 * w.diff(t) - w.diff(x).diff(x)) / (4 * w.diff(x));
    v2 = (beta / 2 - 1) * w.diff(x) * w.diff(x); v1 = (1 - beta / 2) * w.diff(x).diff(x);
    v0 = - (beta - 2) * \
        (2 * (w.diff(x))**4 - w.diff(x) * w.diff(x).diff(x).diff(x) + 2 * w.diff(x) * w.diff(x).diff(t) + (w.diff(x).diff(x))**2 - 2 * w.diff(x).diff(x) * w.diff(t)) / \
            (4 * w.diff(x) * w.diff(x))
    u = u0 + u1 * sy.tanh(w); v = v0 + v1 * sy.tanh(w) + v2 * (sy.tanh(w))**2;
    func_u = sy.lambdify([x, t], u,'numpy')
    func_v = sy.lambdify([x, t], v,'numpy')

    x, t = np.linspace(x0, x1, 201), np.linspace(t0, t1, 201); x, t = np.meshgrid(x, t); x, t = x.reshape(-1), t.reshape(-1);
    u, v = func_u(x, t), func_v(x, t)

    xb = np.concatenate([np.linspace(x0, x1, 201), np.linspace(x0, x1, 201), np.zeros(201) + x0, np.zeros(201) + x1], axis=0)
    tb = np.concatenate([np.zeros(201) + t0, np.zeros(201) + t1, np.linspace(t0, t1, 201), np.linspace(t0, t1, 201)], axis=0)
    ub, vb = func_u(xb, tb), func_v(xb, tb)

    threshold1, threshold2, threshold3 = t0 + (t1 - t0) / 4, 2 * (t1 + t0) / 4, t0 + 3 * (t1 - t0) / 4

    xi1 = np.linspace(x0, x1, 400); ti1 = np.zeros((400)) + threshold1;
    xi2 = np.linspace(x0, x1, 400); ti2 = np.zeros((400)) + threshold2;
    xi3 = np.linspace(x0, x1, 400); ti3 = np.zeros((400)) + threshold3;

    index = np.where(t < threshold1)[0]; x1 = x[index]; t1 = t[index]; u1 = u[index]; v1 = v[index];
    index = np.where((t < threshold2) & (t >= threshold1))[0]; x2 = x[index]; t2 = t[index]; u2 = u[index]; v2 = v[index];
    index = np.where((t < threshold3) & (t >= threshold2))[0]; x3 = x[index]; t3 = t[index]; u3 = u[index]; v3 = v[index];
    index = np.where(t >= threshold3)[0]; x4 = x[index]; t4 = t[index]; u4 = u[index]; v4 = v[index];
    index = np.where(tb < threshold1)[0]; xb1 = xb[index]; tb1 = tb[index]; ub1 = ub[index]; vb1 = vb[index];
    index = np.where((tb < threshold2) & (tb >= threshold1))[0]; xb2 = xb[index]; tb2 = tb[index]; ub2 = ub[index]; vb2 = vb[index];
    index = np.where((tb < threshold3) & (tb >= threshold2))[0]; xb3 = xb[index]; tb3 = tb[index]; ub3 = ub[index]; vb3 = vb[index];
    index = np.where((tb >= threshold3))[0]; xb4 = xb[index]; tb4 = tb[index]; ub4 = ub[index]; vb4 = vb[index];

    N_u = 400 # Number of boundary points
    N_f = 10000 # Number of collocation points

    N_u_i = 100 # Number of boundary points for each sub-domain
    N_f_i = 2500 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    xf = x[idx]; tf = t[idx];

    # Collocation XPINN selection
    idx = np.random.choice(u1.shape[0], N_f_i, replace=False)
    xf1 = x1[idx]; tf1 = t1[idx];
    idx = np.random.choice(u2.shape[0], N_f_i, replace=False)
    xf2 = x2[idx]; tf2 = t2[idx];
    idx = np.random.choice(u3.shape[0], N_f_i, replace=False)
    xf3 = x3[idx]; tf3 = t3[idx];
    idx = np.random.choice(u4.shape[0], N_f_i, replace=False)
    xf4 = x4[idx]; tf4 = t4[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; vb = vb[idx]; xb = xb[idx]; tb = tb[idx];

    # Boundary XPINN selection
    idx = np.random.choice(ub1.shape[0], N_u_i, replace=False)
    ub1 = ub1[idx]; vb1 = vb1[idx]; xb1 = xb1[idx]; tb1 = tb1[idx];
    idx = np.random.choice(ub2.shape[0], N_u_i, replace=False)
    ub2 = ub2[idx]; vb2 = vb2[idx]; xb2 = xb2[idx]; tb2 = tb2[idx];
    idx = np.random.choice(ub3.shape[0], N_u_i, replace=False)
    ub3 = ub3[idx]; vb3 = vb3[idx]; xb3 = xb3[idx]; tb3 = tb3[idx];
    idx = np.random.choice(ub4.shape[0], N_u_i, replace=False)
    ub4 = ub4[idx]; vb4 = vb4[idx]; xb4 = xb4[idx]; tb4 = tb4[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,tf1,s=3,c='b')
        plt.scatter(xf2,tf2,s=3,c='r')
        plt.scatter(xf3,tf3,s=3,c='b')
        plt.scatter(xf4,tf4,s=3,c='r')
        plt.scatter(xi1,ti1,s=3,c='g')
        plt.scatter(xi2,ti2,s=3,c='g')
        plt.scatter(xi3,ti3,s=3,c='g')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Boussinesq-Burger XPINN4 Points')
        plt.scatter(xb1,tb1,c='y',s=3)
        plt.scatter(xb2,tb2,c='k',s=3)
        plt.scatter(xb3,tb3,c='y',s=3)
        plt.scatter(xb4,tb4,c='k',s=3)
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(x, t)
        plt.tricontourf(triang_coarse, u, 100 ,cmap='jet', extend="both")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Boussinesq-Burger u(x,t)')
        plt.colorbar()
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(x, t)
        plt.tricontourf(triang_coarse, v, 100 ,cmap='jet', extend="both")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Boussinesq-Burger v(x,t)')
        plt.colorbar()
        plt.show()
    return x, t, u, v, x1, t1, u1, v1, x2, t2, u2, v2, x3, t3, u3, v3, x4, t4, u4, v4,\
        xb, tb, ub, vb, xb1, tb1, ub1, vb1, xb2, tb2, ub2, vb2, xb3, tb3, ub3, vb3, xb4, tb4, ub4, vb4,\
            xf, tf, xf1, tf1, xf2, tf2, xf3, tf3, xf4, tf4,\
                xi1, ti1, xi2, ti2, xi3, ti3

x, t, u, v, x1, t1, u1, v1, x2, t2, u2, v2, x3, t3, u3, v3, x4, t4, u4, v4,\
        xb, tb, ub, vb, xb1, tb1, ub1, vb1, xb2, tb2, ub2, vb2, xb3, tb3, ub3, vb3, xb4, tb4, ub4, vb4,\
            xf, tf, xf1, tf1, xf2, tf2, xf3, tf3, xf4, tf4,\
                xi1, ti1, xi2, ti2, xi3, ti3 = load_data_BB(save_fig=1)

class Net(nn.Module):
    def __init__(self, layers, act=torch.tanh):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x

class Gate_Net(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super(Gate_Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        x = nn.Softmax(dim=-1)(x)
        return x

class SXPINN_Net(nn.Module):
    def __init__(self, E_layers, G_layers, h_layers, act=nn.Tanh()):
        super(SXPINN_Net, self).__init__()
        self.u1_net, self.u2_net, self.v1_net, self.v2_net = Net(E_layers, act), Net(E_layers, act), Net(E_layers, act), Net(E_layers, act)
        self.u3_net, self.u4_net, self.v3_net, self.v4_net = Net(E_layers, act), Net(E_layers, act), Net(E_layers, act), Net(E_layers, act)
        self.w_gate = Gate_Net(G_layers, act)
        self.h = Net(h_layers, act)
        self.act = act

    def u_net(self, x):
        w = self.w_gate(x)
        w1, w2, w3, w4 = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
        w1, w2, w3, w4 =  w1.reshape(-1, 1), w2.reshape(-1, 1), w3.reshape(-1, 1), w4.reshape(-1, 1)
        return w1 * self.u1_net(self.act(self.h(x))) + w2 * self.u2_net(self.act(self.h(x))) + \
            w3 * self.u3_net(self.act(self.h(x))) + w4 * self.u4_net(self.act(self.h(x)))
    
    def v_net(self, x):
        w = self.w_gate(x)
        w1, w2, w3, w4 = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
        w1, w2, w3, w4 =  w1.reshape(-1, 1), w2.reshape(-1, 1), w3.reshape(-1, 1), w4.reshape(-1, 1)
        return w1 * self.v1_net(self.act(self.h(x))) + w2 * self.v2_net(self.act(self.h(x))) + \
            w3 * self.v3_net(self.act(self.h(x))) + w4 * self.v4_net(self.act(self.h(x)))
        
    def forward(self, x):
        u_out = self.w_gate(x) * self.u1_net(self.act(self.h(x))) + (1 - self.w_gate(x)) * self.u2_net(self.act(self.h(x)))
        v_out = self.w_gate(x) * self.v1_net(self.act(self.h(x))) + (1 - self.w_gate(x)) * self.v2_net(self.act(self.h(x)))
        return u_out, v_out
    
    def u1(self, x):
        return self.w_gate(x) * self.u1_net(self.act(self.h(x)))
    
    def u2(self, x):
        return (1 - self.w_gate(x)) * self.u2_net(self.act(self.h(x)))

class PINN_XPINN:
    def __init__(self):
        self.epoch = args.epochs
        self.verbose = 1
        self.adam_lr = args.lr
        # PINN weights
        self.pinn_w_rc = args.PINN_weights[0] # Residual Collocation
        self.pinn_w_b = args.PINN_weights[1] # Boundary
        # XPINN weights
        self.xpinn_w_rc = args.XPINN_weights[0] # Residual Collocation
        self.xpinn_w_ri = args.XPINN_weights[1] # Residual interface
        self.xpinn_w_b = args.XPINN_weights[2] # Boundary
        self.xpinn_w_i = args.XPINN_weights[3] # Interface
        self.xpinn_w_ra = args.XPINN_weights[4] # additional residual interface
        
        # boundary points
        # PINN
        self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb = torch.unsqueeze(torch.tensor(tb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        # XPINN
        self.xb1 = torch.unsqueeze(torch.tensor(xb1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb1 = torch.unsqueeze(torch.tensor(tb1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub1 = torch.unsqueeze(torch.tensor(ub1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb1 = torch.unsqueeze(torch.tensor(vb1, dtype=torch.float32, requires_grad=True),-1).to(device)

        self.xb2 = torch.unsqueeze(torch.tensor(xb2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb2 = torch.unsqueeze(torch.tensor(tb2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub2 = torch.unsqueeze(torch.tensor(ub2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb2 = torch.unsqueeze(torch.tensor(vb2, dtype=torch.float32, requires_grad=True),-1).to(device)

        self.xb3 = torch.unsqueeze(torch.tensor(xb3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb3 = torch.unsqueeze(torch.tensor(tb3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub3 = torch.unsqueeze(torch.tensor(ub3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb3 = torch.unsqueeze(torch.tensor(vb3, dtype=torch.float32, requires_grad=True),-1).to(device)

        self.xb4 = torch.unsqueeze(torch.tensor(xb4, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb4 = torch.unsqueeze(torch.tensor(tb4, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub4 = torch.unsqueeze(torch.tensor(ub4, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb4 = torch.unsqueeze(torch.tensor(vb4, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        # collocation points
        # PINN
        self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf = torch.unsqueeze(torch.tensor(tf, dtype=torch.float32, requires_grad=True),-1).to(device)
        # XPINN
        self.xf1 = torch.unsqueeze(torch.tensor(xf1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf1 = torch.unsqueeze(torch.tensor(tf1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xf2 = torch.unsqueeze(torch.tensor(xf2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf2 = torch.unsqueeze(torch.tensor(tf2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xf3 = torch.unsqueeze(torch.tensor(xf3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf3 = torch.unsqueeze(torch.tensor(tf3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xf4 = torch.unsqueeze(torch.tensor(xf4, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf4 = torch.unsqueeze(torch.tensor(tf4, dtype=torch.float32, requires_grad=True),-1).to(device)

        # interface point
        self.xi1 = torch.unsqueeze(torch.tensor(xi1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ti1 = torch.unsqueeze(torch.tensor(ti1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xi2 = torch.unsqueeze(torch.tensor(xi2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ti2 = torch.unsqueeze(torch.tensor(ti2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xi3 = torch.unsqueeze(torch.tensor(xi3, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ti3 = torch.unsqueeze(torch.tensor(ti3, dtype=torch.float32, requires_grad=True),-1).to(device)

        # Initalize Neural Networks
        layers = [2] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        sub_layers = [2] + [args.XPINN_h] * (args.XPINN_L - 1) + [args.output_dim]

        self.u_net, self.v_net = Net(layers).to(device), Net(layers).to(device)
        self.u1_net, self.v1_net = Net(sub_layers).to(device), Net(sub_layers).to(device)
        self.u2_net, self.v2_net = Net(sub_layers).to(device), Net(sub_layers).to(device)
        self.u3_net, self.v3_net = Net(sub_layers).to(device), Net(sub_layers).to(device)
        self.u4_net, self.v4_net = Net(sub_layers).to(device), Net(sub_layers).to(device)

        self.net_params_pinn = list(self.u_net.parameters()) + list(self.v_net.parameters())
        self.net_params_xpinn = list(self.u1_net.parameters())+list(self.u2_net.parameters())+list(self.v1_net.parameters())+list(self.v2_net.parameters())+ \
            list(self.u3_net.parameters())+list(self.u4_net.parameters())+list(self.v3_net.parameters())+list(self.v4_net.parameters())

        self.loss_pinn = []
        self.loss_xpinn = []
        self.loss = 10

        self.best_l2_error = [1, 1]
        self.best_l2_abs_error = 0
        
    def PDE1(self, u, u_t, u_x, v_x): 
        return u_t - 2 * u * u_x - 0.5 * v_x

    def PDE2(self, u, v, v_t, u_x, v_x, u_xxx): 
        return v_t - 0.5 * u_xxx - 2 * u * v_x - 2 * u_x * v

    def num_params(self):
        num_pinn, num_xpinn = 0, 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        for p in self.net_params_xpinn:
            num_xpinn += len(p.reshape(-1))
        return num_pinn, num_xpinn

    def get_loss_pinn(self):
        # Boundry loss
        mse_b = (self.ub - self.u_net(torch.cat((self.xb, self.tb), 1))).square().mean() + (self.vb - self.v_net(torch.cat((self.xb, self.tb), 1))).square().mean()
        
        # Residual/collocation loss
        u = self.u_net(torch.cat((self.xf, self.tf), 1)); u_sum = u.sum();
        u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
        u_t = torch.autograd.grad(u_sum, self.tf, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx.sum(), self.xf, create_graph=True)[0]
        v = self.v_net(torch.cat((self.xf, self.tf), 1)); v_sum = v.sum();
        v_x = torch.autograd.grad(v_sum, self.xf, create_graph=True)[0]
        v_t = torch.autograd.grad(v_sum, self.tf, create_graph=True)[0]
        f1 = self.PDE1(u, u_t, u_x, v_x)
        f2 = self.PDE2(u, v, v_t, u_x, v_x, u_xxx) 

        mse_f = f1.square().mean() + f2.square().mean()
    
        # Sum losses
        loss = self.pinn_w_b*mse_b + self.pinn_w_rc*mse_f
        
        return loss
    
    def _residual(self, u_net_, v_net_, xf_, tf_):
        u_ = u_net_(torch.cat((xf_, tf_), 1))
        u_sum = u_.sum()
        u_x = torch.autograd.grad(u_sum, xf_, create_graph=True)[0]
        u_t = torch.autograd.grad(u_sum, tf_, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), xf_, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx.sum(), xf_, create_graph=True)[0]
        v_ = v_net_(torch.cat((xf_, tf_), 1))
        v_sum = v_.sum()
        v_x = torch.autograd.grad(v_sum, xf_, create_graph=True)[0]
        v_t = torch.autograd.grad(v_sum, tf_, create_graph=True)[0]
        f1 = self.PDE1(u_, u_t, u_x, v_x)
        f2 = self.PDE2(u_, v_, v_t, u_x, v_x, u_xxx) 
        return f1, f2
    
    def _interface(self, u1_net, v1_net, u2_net, v2_net, xi1, ti1):
        # Sub-net 1, Interface 1
        u1i1 = u1_net(torch.cat((xi1, ti1), 1))
        u1i1_sum = u1i1.sum()
        v1i1 = v1_net(torch.cat((xi1, ti1), 1))
        v1i1_sum = v1i1.sum()
        # Sub-net 2, Interface 1
        u2i1 = u2_net(torch.cat((xi1, ti1), 1))
        u2i1_sum = u2i1.sum()
        v2i1 = v2_net(torch.cat((xi1, ti1), 1))
        v2i1_sum = v2i1.sum()
        u1i1_x = torch.autograd.grad(u1i1_sum, xi1, create_graph=True)[0]
        u1i1_t = torch.autograd.grad(u1i1_sum, ti1, create_graph=True)[0]
        u2i1_x = torch.autograd.grad(u2i1_sum, xi1, create_graph=True)[0]
        u2i1_t = torch.autograd.grad(u2i1_sum, ti1, create_graph=True)[0]
        mse_fai1 = ((u1i1_x-u2i1_x).square().mean() + (u1i1_t-u2i1_t).square().mean()) / 2
        v1i1_x = torch.autograd.grad(v1i1_sum, xi1, create_graph=True)[0]
        v1i1_t = torch.autograd.grad(v1i1_sum, ti1, create_graph=True)[0]
        v2i1_x = torch.autograd.grad(v2i1_sum, xi1, create_graph=True)[0]
        v2i1_t = torch.autograd.grad(v2i1_sum, ti1, create_graph=True)[0]
        mse_fai1 += ((v1i1_x-v2i1_x).square().mean() + (v1i1_t-v2i1_t).square().mean()) / 2
        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1)/2; vavgi1 = (v1i1 + v2i1)/2;
        mse_u1avgi1 = (u1i1-uavgi1).square().mean(); mse_u2avgi1 = (u2i1-uavgi1).square().mean();
        mse_v1avgi1 = (v1i1-vavgi1).square().mean(); mse_v2avgi1 = (v2i1-vavgi1).square().mean();
        mse_bi = mse_u1avgi1 + mse_u2avgi1 + mse_v1avgi1 + mse_v2avgi1

        return mse_bi, mse_fai1

    def get_loss_xpinn(self):
        # Boundry loss
        mse_b1 = (self.ub1 - self.u1_net(torch.cat((self.xb1, self.tb1), 1))).square().mean() + (self.vb1 - self.v1_net(torch.cat((self.xb1, self.tb1), 1))).square().mean() 
        mse_b2 = (self.ub2 - self.u2_net(torch.cat((self.xb2, self.tb2), 1))).square().mean() + (self.vb2 - self.v2_net(torch.cat((self.xb2, self.tb2), 1))).square().mean() 
        mse_b3 = (self.ub3 - self.u3_net(torch.cat((self.xb3, self.tb3), 1))).square().mean() + (self.vb3 - self.v3_net(torch.cat((self.xb3, self.tb3), 1))).square().mean()
        mse_b4 = (self.ub4 - self.u4_net(torch.cat((self.xb4, self.tb4), 1))).square().mean() + (self.vb4 - self.v4_net(torch.cat((self.xb4, self.tb4), 1))).square().mean() 
         
        mse_b = mse_b1 + mse_b2 + mse_b3 + mse_b4

        # Residual/collocation loss
        f1a, f1b = self._residual(self.u1_net, self.v1_net, self.xf1, self.tf1)
        f2a, f2b = self._residual(self.u2_net, self.v2_net, self.xf2, self.tf2)
        f3a, f3b = self._residual(self.u3_net, self.v3_net, self.xf3, self.tf3)
        f4a, f4b = self._residual(self.u4_net, self.v4_net, self.xf4, self.tf4)
        mse_f = f1a.square().mean() + f1b.square().mean() + \
            f2a.square().mean() + f2b.square().mean() + \
            f3a.square().mean() + f3b.square().mean() + \
            f4a.square().mean() + f4b.square().mean()

        mse_bi1, mse_fai1 = self._interface(self.u1_net, self.v1_net, self.u2_net, self.v2_net, self.xi1, self.ti1)
        mse_bi2, mse_fai2 = self._interface(self.u2_net, self.v2_net, self.u3_net, self.v3_net, self.xi2, self.ti2)
        mse_bi3, mse_fai3 = self._interface(self.u3_net, self.v3_net, self.u4_net, self.v4_net, self.xi3, self.ti3)
        mse_bi, mse_fai = mse_bi1 + mse_bi2 + mse_bi3, mse_fai1 + mse_fai2 + mse_fai3

        # Sum losses
        loss = self.xpinn_w_b*mse_b + self.xpinn_w_rc*mse_f + self.xpinn_w_i*mse_bi + self.xpinn_w_ra*mse_fai
        return loss
    
    def train_adam_pinn(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        for n in tqdm(range(self.epoch)):
            loss = self.get_loss_pinn()
            current_loss = loss.item()
            #self.loss = loss.detach().cpu().numpy()
            self.loss_pinn.append(current_loss)
            if n%10000==0 and self.verbose == 1:
                print('epoch %d, loss: %g'%(n, current_loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if n > 0.9 * self.epoch:
                self.loss = current_loss
                current_L2 = self.L2_pinn(x, t)
                if self.best_l2_error[0] > current_L2[0]:
                    self.best_l2_error[0] = current_L2[0]
                if self.best_l2_error[1] > current_L2[1]:
                    self.best_l2_error[1] = current_L2[1]
    
    def train_lbfgs_pinn(self):
        optimizer = torch.optim.LBFGS(self.net_params_pinn, lr=0.1, max_iter=20000, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_pinn()
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        current_loss = self.get_loss_pinn().item()
        current_L2 = self.L2_pinn(x, t)
        print("LBFGS", current_loss, current_L2)
        if self.best_l2_error[0] > current_L2[0]:
            self.best_l2_error[0] = current_L2[0]
        if self.best_l2_error[1] > current_L2[1]:
            self.best_l2_error[1] = current_L2[1]
        return

    def train_adam_xpinn(self):
        optimizer = torch.optim.Adam(self.net_params_xpinn, lr=self.adam_lr)
        for n in tqdm(range(self.epoch)):
            loss = self.get_loss_xpinn()
            current_loss = loss.item()
            #self.loss = loss.detach().cpu().numpy()
            self.loss_xpinn.append(current_loss)
            if n%10000==0 and self.verbose == 1:
                print('epoch %d, loss: %g'%(n, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if n > 0.9 * self.epoch:
                self.loss = current_loss
                current_L2 = self.L2_xpinn(x1, t1, x2, t2, x3, t3, x4, t4)
                if self.best_l2_error[0] > current_L2[0]:
                    self.best_l2_error[0] = current_L2[0]
                if self.best_l2_error[1] > current_L2[1]:
                    self.best_l2_error[1] = current_L2[1]

    def train_lbfgs_xpinn(self):
        optimizer = torch.optim.LBFGS(self.net_params_xpinn, lr=0.1, max_iter=20000, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_xpinn()
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        current_loss = self.get_loss_xpinn().item()
        current_L2 = self.L2_xpinn(x1, t1, x2, t2, x3, t3, x4, t4)
        print("LBFGS", current_loss, current_L2)
        if self.best_l2_error[0] > current_L2[0]:
            self.best_l2_error[0] = current_L2[0]
        if self.best_l2_error[1] > current_L2[1]:
            self.best_l2_error[1] = current_L2[1]
        return

    def predict_pinn(self, x_star, y_star):
        x_star = torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1).to(device)
        y_star = torch.unsqueeze(torch.tensor(y_star, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u_pred = self.u_net(torch.cat((x_star, y_star), 1))
            v_pred = self.v_net(torch.cat((x_star, y_star), 1))
        return u_pred, v_pred
    
    def predict_xpinn(self, x_star1, y_star1, x_star2, y_star2, x_star3, y_star3, x_star4, y_star4):
        x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1).to(device)
        y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1).to(device)
        x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1).to(device)
        y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1).to(device)
        x_star3 = torch.unsqueeze(torch.tensor(x_star3, dtype=torch.float32),-1).to(device)
        y_star3 = torch.unsqueeze(torch.tensor(y_star3, dtype=torch.float32),-1).to(device)
        x_star4 = torch.unsqueeze(torch.tensor(x_star4, dtype=torch.float32),-1).to(device)
        y_star4 = torch.unsqueeze(torch.tensor(y_star4, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u1_pred = self.u1_net(torch.cat((x_star1, y_star1), 1))
            u2_pred = self.u2_net(torch.cat((x_star2, y_star2), 1))
            u3_pred = self.u3_net(torch.cat((x_star3, y_star3), 1))
            u4_pred = self.u4_net(torch.cat((x_star4, y_star4), 1))
            v1_pred = self.v1_net(torch.cat((x_star1, y_star1), 1))
            v2_pred = self.v2_net(torch.cat((x_star2, y_star2), 1))
            v3_pred = self.v3_net(torch.cat((x_star3, y_star3), 1))
            v4_pred = self.v4_net(torch.cat((x_star4, y_star4), 1))
        return u1_pred, u2_pred, u3_pred, u4_pred, v1_pred, v2_pred, v3_pred, v4_pred

    def L2_pinn(self, x_star, y_star):
        pinn_u_pred_20, pinn_v_pred_20 = self.predict_pinn(x_star, y_star)
        pinn_u_pred_20, pinn_v_pred_20 = pinn_u_pred_20.detach().cpu().numpy(), pinn_v_pred_20.detach().cpu().numpy()
        pinn_error_u_total_20 = np.linalg.norm(u-pinn_u_pred_20.flatten(),2)/np.linalg.norm(u,2)
        pinn_error_v_total_20 = np.linalg.norm(v-pinn_v_pred_20.flatten(),2)/np.linalg.norm(v,2)
        return pinn_error_u_total_20, pinn_error_v_total_20

    def error_plot_pinn(self):
        pinn_u_pred_20, pinn_v_pred_20 = (self.predict_pinn(x, t))
        pinn_u_pred_20, pinn_v_pred_20 = pinn_u_pred_20.detach().cpu().numpy(), pinn_v_pred_20.detach().cpu().numpy()
        pinn_u_error_20 = abs(u-pinn_u_pred_20.flatten()); pinn_v_error_20 = abs(v-pinn_v_pred_20.flatten());
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        idx = np.random.choice(x.shape[0], 10000, replace=False)
        triang_coarse = tri.Triangulation(x[idx], t[idx])
        plt.tricontourf(triang_coarse, pinn_u_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u Error')
        plt.colorbar()
        plt.savefig("fig/PINN_u_Burgers.pdf")
        plt.show()

        plt.tricontourf(triang_coarse, pinn_v_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers v Error')
        plt.colorbar()
        plt.savefig("fig/PINN_v_Burgers.pdf")
        plt.show()

    def error_save_pinn(self):
        pinn_u_pred_20 = (self.predict_pinn(x, t)).detach().cpu().numpy()
        pinn_error_20 = abs(u-pinn_u_pred_20.flatten())
        return pinn_error_20
        #np.savetxt("error/PINN_"+args.dataset+"_error_"+str(args.SEED), pinn_error_20)
    
    def plot_loss_pinn(self):
        plt.title("Burgers PINN Loss")
        plt.plot(self.loss_pinn)
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("fig/loss_pinn_burgers.pdf")

    def L2_xpinn(self, x_star1, y_star1, x_star2, y_star2, x_star3, y_star3, x_star4, y_star4):
        xpinn_u_pred1_20, xpinn_u_pred2_20, xpinn_u_pred3_20, xpinn_u_pred4_20, \
            xpinn_v_pred1_20, xpinn_v_pred2_20, xpinn_v_pred3_20, xpinn_v_pred4_20 = self.predict_xpinn(x_star1, y_star1, x_star2, y_star2, x_star3, y_star3, x_star4, y_star4)
        xpinn_u_pred1_20 = (xpinn_u_pred1_20).detach().cpu().numpy(); xpinn_u_pred2_20 = (xpinn_u_pred2_20).detach().cpu().numpy();
        xpinn_v_pred1_20 = (xpinn_v_pred1_20).detach().cpu().numpy(); xpinn_v_pred2_20 = (xpinn_v_pred2_20).detach().cpu().numpy();
        xpinn_u_pred3_20 = (xpinn_u_pred3_20).detach().cpu().numpy(); xpinn_u_pred4_20 = (xpinn_u_pred4_20).detach().cpu().numpy();
        xpinn_v_pred3_20 = (xpinn_v_pred3_20).detach().cpu().numpy(); xpinn_v_pred4_20 = (xpinn_v_pred4_20).detach().cpu().numpy();
        xpinn_u_pred_20 = np.concatenate([xpinn_u_pred1_20, xpinn_u_pred2_20, xpinn_u_pred3_20, xpinn_u_pred4_20]); xpinn_v_pred_20 = np.concatenate([xpinn_v_pred1_20, xpinn_v_pred2_20, xpinn_v_pred3_20, xpinn_v_pred4_20]);
        xpinn_u_vals = np.concatenate([u1, u2, u3, u4]); xpinn_v_vals = np.concatenate([v1, v2, v3, v4])
        #xpinn_error_20 = abs(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten())
        xpinn_error_u_total_20 = np.linalg.norm(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten(),2)/np.linalg.norm(xpinn_u_vals.flatten(),2)
        xpinn_error_v_total_20 = np.linalg.norm(xpinn_v_vals.flatten()-xpinn_v_pred_20.flatten(),2)/np.linalg.norm(xpinn_v_vals.flatten(),2)
        return xpinn_error_u_total_20, xpinn_error_v_total_20

    def error_plot_xpinn(self):
        xpinn_u_pred1_20, xpinn_u_pred2_20, xpinn_u_pred3_20, xpinn_u_pred4_20, \
            xpinn_v_pred1_20, xpinn_v_pred2_20, xpinn_v_pred3_20, xpinn_v_pred4_20 = self.predict_xpinn(x1, t1, x2, t2, x3, t3, x4, t4)
        xpinn_u_pred_20 = torch.cat([xpinn_u_pred1_20, xpinn_u_pred2_20, xpinn_u_pred3_20, xpinn_u_pred4_20]).detach().cpu().numpy()
        xpinn_v_pred_20 = torch.cat([xpinn_v_pred1_20, xpinn_v_pred2_20, xpinn_v_pred3_20, xpinn_v_pred4_20]).detach().cpu().numpy()
        xpinn_u_vals = np.concatenate([u1, u2, u3, u4])
        xpinn_error_20 = abs(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        #idx = np.random.choice(x1.shape[0]+x2.shape[0], 10000, replace=False)
        idx = np.arange(len(x))
        triang_coarse = tri.Triangulation(np.concatenate([x1, x2, x3, x4])[idx], np.concatenate([t1, t2, t3, t4])[idx])
        plt.tricontourf(triang_coarse, xpinn_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB XPINN U Error')
        plt.colorbar()
        plt.savefig("fig/XPINN_u_BB4.pdf")
        plt.show()
        xpinn_v_vals = np.concatenate([v1, v2, v3, v4])
        xpinn_error_20 = abs(xpinn_v_vals.flatten()-xpinn_v_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        #idx = np.random.choice(x1.shape[0]+x2.shape[0], 10000, replace=False)
        triang_coarse = tri.Triangulation(np.concatenate([x1, x2, x3, x4])[idx], np.concatenate([t1, t2, t3, t4])[idx])
        plt.tricontourf(triang_coarse, xpinn_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB XPINN V Error')
        plt.colorbar()
        plt.savefig("fig/XPINN_v_BB4.pdf")
        plt.show()

    def error_save_xpinn(self):
        xpinn_u_pred1_20, xpinn_u_pred2_20, xpinn_u_pred3_20, xpinn_u_pred4_20 = self.predict_xpinn(x1, t1, x2, t2, x3, t3, x4, t4)
        xpinn_u_pred1_20 = (xpinn_u_pred1_20).detach().cpu().numpy()
        xpinn_u_pred2_20 = (xpinn_u_pred2_20).detach().cpu().numpy()
        xpinn_u_pred_20 = np.concatenate([xpinn_u_pred1_20, xpinn_u_pred2_20])
        xpinn_u_vals = np.concatenate([u1, u2])
        xpinn_error_20 = abs(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten())
        return xpinn_error_20
        #np.savetxt("error/XPINN_"+args.dataset+"_error_"+str(args.SEED), xpinn_error_20)

    def plot_loss_xpinn(self):
        plt.title("Burgers XPINN Loss")
        plt.plot(self.loss_xpinn)
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("fig/loss_xpinn_burgers.pdf")

class SXPINN:
    def __init__(self):
        self.epoch = args.epochs
        self.verbose = 1
        self.adam_lr = args.lr
        self.pinn_w_rc = args.PINN_weights[0]; # Residual Collocation
        self.pinn_w_b = args.PINN_weights[1]; # Boundary

        # boundary points
        # PINN
        self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tb = torch.unsqueeze(torch.tensor(tb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        # collocation points
        # PINN
        self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.tf = torch.unsqueeze(torch.tensor(tf, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        E_layers = [args.SXPINN_E_h] * args.SXPINN_E_L + [1]
        G_layers = [args.input_dim] + [args.SXPINN_G_h] * (args.SXPINN_G_L - 1) + [4]
        #h_layers = [args.SXPINN_E_h] + [args.SXPINN_h_h] * (args.SXPINN_h_L - 1) + [1]
        h_layers = [args.input_dim] + [args.SXPINN_h_h] * (args.SXPINN_h_L - 1) + [args.SXPINN_E_h]

        self.net = SXPINN_Net(E_layers, G_layers, h_layers).to(device)
        
        if args.SXPINN_gate_trainable:
            self.net_params_sxpinn = list(self.net.parameters())
        else:
            self.net_params_sxpinn = list(self.net.u1_net.parameters()) + list(self.net.u2_net.parameters()) + list(self.net.h.parameters()) + \
                list(self.net.v1_net.parameters()) + list(self.net.v2_net.parameters())

        self.loss_sxpinn = []
        self.loss = 10
        self.best_l2_error = 1.0

        if args.save_loss:
            self.saved_loss = []
            self.saved_l2 = []

    def PDE1(self, u, u_t, u_x, v_x): 
        return u_t - 2 * u * u_x - 0.5 * v_x

    def PDE2(self, u, v, v_t, u_x, v_x, u_xxx): 
        return v_t - 0.5 * u_xxx - 2 * u * v_x - 2 * u_x * v
    
    def num_params(self):
        num_sxpinn = 0
        for p in self.net_params_sxpinn:
            num_sxpinn += len(p.reshape(-1))
        return num_sxpinn

    def get_loss_sxpinn(self):
        # Boundry loss
        mse_b = (self.ub - self.net.u_net(torch.cat((self.xb, self.tb), 1))).square().mean() + (self.vb - self.net.v_net(torch.cat((self.xb, self.tb), 1))).square().mean()
        
        # Residual/collocation loss
        u = self.net.u_net(torch.cat((self.xf, self.tf), 1))
        u_sum = u.sum()
        u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
        u_t = torch.autograd.grad(u_sum, self.tf, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx.sum(), self.xf, create_graph=True)[0]
        v = self.net.v_net(torch.cat((self.xf, self.tf), 1)); v_sum = v.sum();
        v_x = torch.autograd.grad(v_sum, self.xf, create_graph=True)[0]
        v_t = torch.autograd.grad(v_sum, self.tf, create_graph=True)[0]
        f1 = self.PDE1(u, u_t, u_x, v_x)
        f2 = self.PDE2(u, v, v_t, u_x, v_x, u_xxx) 

        mse_f = f1.square().mean() + f2.square().mean()
    
        # Sum losses
        loss = self.pinn_w_b*mse_b + self.pinn_w_rc*mse_f
        
        return loss
    
    def train_adam_sxpinn(self):
        # stage 1: pretrain gate_net
        self.net.w_gate = self.pretrain_gate_net(self.net.w_gate)
        optimizer = torch.optim.Adam(self.net_params_sxpinn, lr=self.adam_lr)
        for n in tqdm(range(self.epoch)):
            loss = self.get_loss_sxpinn()
            current_loss = loss.item()
            #self.loss = loss.detach().cpu().numpy()
            self.loss_sxpinn.append(current_loss)
            if n%10000==0 and self.verbose == 1:
                print('epoch %d, loss: %g'%(n, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.save_loss:
                self.saved_loss.append(current_loss)
                self.saved_l2.append(self.L2_sxpinn(x, t))
            if n > 0.9 * self.epoch:
                self.loss = current_loss
                current_L2 = self.L2_sxpinn(x, t)
                if sum(self.best_l2_error) > sum(current_L2):
                    self.best_l2_error = current_L2

 
    def pretrain_gate_net(self, w_gate, x0=-10, x1=15, t0=-3, t1=2):
        x, y = np.linspace(x0, x1, 100), np.linspace(t0, t1, 100)
        x, y = np.meshgrid(x, y)
        #u = np.exp((y - t1)) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        if args.SXPINN_Prior == 'MPINN':
            u = np.zeros((x.shape[0], 4))
            u[:, 0] += 0.8; u[:, 1] += 0.2/3; u[:, 2] += 0.2/3; u[:, 3] += 0.2/3;
        else:
            u = np.zeros((x.shape[0], 4))
            u1 = np.exp(y-t1); u4 = np.exp(t0-y); u2 = np.exp(-np.abs(y - t1 + 5/3)); u3 = np.exp(-np.abs(y - t1 + 10/3));
            u[:, 0] = u1.reshape(-1); u[:, 1] = u2.reshape(-1); u[:, 2] = u3.reshape(-1); u[:, 3] = u4.reshape(-1);
        x, y, u = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda(), torch.from_numpy(u).float().cuda()
        optimizer = torch.optim.Adam(w_gate.parameters(), lr=1e-3, weight_decay=1e-3)

        for epoch in range(1, 10000+1):
            loss = torch.nn.MSELoss()(w_gate(torch.cat((x, y), 1)), u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10000 == 0:
                print(epoch, loss.item())
        return w_gate
    
    def save_gate(self):
        x0=-10; x1=15; t0=-3; t1=2;
        x, y = np.linspace(x0, x1, 100), np.linspace(t0, t1, 100)
        x, y = np.meshgrid(x, y)
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
        u = self.net.w_gate(torch.cat((x, y), 1))
        x, y, u = x.reshape(-1), y.reshape(-1), u.reshape(-1, 4)
        x, y, u = x.detach().cpu().numpy(), y.detach().cpu().numpy(), u.detach().cpu().numpy()

        triang_coarse = tri.Triangulation(x, y)
        plt.tricontourf(triang_coarse, u[:, 0], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB Gate Network 1')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("fig/SXPINN_Gate1_"+args.SXPINN_Prior+"_BB4_"+str(args.SEED)+".pdf")
        plt.show()

        triang_coarse = tri.Triangulation(x, y)
        plt.tricontourf(triang_coarse, u[:, 1], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB Gate Network 2')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("fig/SXPINN_Gate2_"+args.SXPINN_Prior+"_BB4_"+str(args.SEED)+".pdf")
        plt.show()

        triang_coarse = tri.Triangulation(x, y)
        plt.tricontourf(triang_coarse, u[:, 2], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB Gate Network 3')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("fig/SXPINN_Gate3_"+args.SXPINN_Prior+"_BB4_"+str(args.SEED)+".pdf")
        plt.show()

        triang_coarse = tri.Triangulation(x, y)
        plt.tricontourf(triang_coarse, u[:, 3], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB Gate Network 4')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("fig/SXPINN_Gate4_"+args.SXPINN_Prior+"_BB4_"+str(args.SEED)+".pdf")
        plt.show()
        return

    def predict_sxpinn(self, x_star, y_star):
        x_star = torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1).to(device)
        y_star = torch.unsqueeze(torch.tensor(y_star, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u_pred = self.net.u_net(torch.cat((x_star, y_star), 1))
            v_pred = self.net.v_net(torch.cat((x_star, y_star), 1))
        return u_pred, v_pred
    
    def L2_sxpinn(self, x_star, y_star):
        sxpinn_u_pred_20, sxpinn_v_pred_20 = (self.predict_sxpinn(x_star, y_star))
        sxpinn_u_pred_20, sxpinn_v_pred_20 = sxpinn_u_pred_20.detach().cpu().numpy(), sxpinn_v_pred_20.detach().cpu().numpy()
        #pinn_error_20 = abs(u-pinn_u_pred_20.flatten())
        sxpinn_error_u_total_20 = np.linalg.norm(u-sxpinn_u_pred_20.flatten(),2)/np.linalg.norm(u,2)
        sxpinn_error_v_total_20 = np.linalg.norm(v-sxpinn_v_pred_20.flatten(),2)/np.linalg.norm(v,2)
        return sxpinn_error_u_total_20, sxpinn_error_v_total_20

    def error_plot_sxpinn(self):
        sxpinn_u_pred_20, sxpinn_v_pred_20 = (self.predict_sxpinn(x, t))
        sxpinn_u_pred_20, sxpinn_v_pred_20 = sxpinn_u_pred_20.detach().cpu().numpy(), sxpinn_v_pred_20.detach().cpu().numpy()
        sxpinn_u_error_20, sxpinn_v_error_20 = abs(u-sxpinn_u_pred_20.flatten()), abs(v-sxpinn_v_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        #idx = np.random.choice(x.shape[0], 10000, replace=False)
        idx = np.arange(x.shape[0])
        triang_coarse = tri.Triangulation(x[idx], t[idx])
        plt.tricontourf(triang_coarse, sxpinn_u_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB SXPINN u Error')
        plt.colorbar()
        plt.savefig("fig/SXPINN_BB4_U.pdf")
        plt.show()

        triang_coarse = tri.Triangulation(x[idx], t[idx])
        plt.tricontourf(triang_coarse, sxpinn_v_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('BB SXPINN v Error')
        plt.colorbar()
        plt.savefig("fig/SXPINN_BB4_V.pdf")
        plt.show()

    def error_save_sxpinn(self):
        sxpinn_u_pred_20, sxpinn_v_pred_20 = (self.predict_sxpinn(x, t))
        sxpinn_u_pred_20, sxpinn_v_pred_20 = sxpinn_u_pred_20.detach().cpu().numpy(), sxpinn_v_pred_20.detach().cpu().numpy()
        sxpinn_u_error_20, sxpinn_v_error_20 = abs(u-sxpinn_u_pred_20.flatten()), abs(v-sxpinn_v_pred_20.flatten())
        np.savetxt("error/SXPINN_BB_U_error_"+str(args.SEED), sxpinn_u_error_20)
        np.savetxt("error/SXPINN_BB_V_error_"+str(args.SEED), sxpinn_v_error_20)
    
    def plot_loss_sxpinn(self):
        plt.title("Burgers SXPINN Loss")
        plt.plot(self.loss_sxpinn)
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("fig/loss_sxpinn_burgers.pdf")

if args.PINN or args.XPINN:
    model = PINN_XPINN()
    print(model.num_params())
elif args.SXPINN:
    model = SXPINN()
    print(model.num_params())

if args.PINN:
    model.train_adam_pinn()
    print('PINN SEED: ', args.SEED, 'Train Loss: ', model.loss, 'Relative L2: ', model.best_l2_error)
    model.train_lbfgs_pinn()
    #.savetxt("saved_loss_l2/PINN"+"_BB4_error_"+str(args.SEED), model.best_abs_error)
    if args.plot_error:
        model.error_plot_pinn()
    if args.plot_loss:
        model.plot_loss_pinn()
    if args.save_error:
        model.error_save_pinn()
if args.XPINN:
    model.train_adam_xpinn()
    print('XPINN Seed: ', args.SEED, 'Train Loss: ', model.loss, 'Relative L2: ', model.best_l2_error)
    #model.train_lbfgs_xpinn()
    #np.savetxt("saved_loss_l2/XPINN"+str(args.XPINN_type)+"_BB4_error_"+str(args.SEED), model.best_abs_error)
    if args.plot_error:
        model.error_plot_xpinn()
    if args.plot_loss:
        model.plot_loss_xpinn()
    if args.save_error:
        model.error_save_xpinn()
if args.SXPINN:
    model.train_adam_sxpinn()
    print('SXPINN SEED: ', args.SEED, 'Train Loss: ', model.loss, 'Relative L2: ', model.best_l2_error)
    model.train_lbfgs_sxpinn()
    #np.savetxt("saved_loss_l2/APINN"+args.SXPINN_Prior[0]+"_BB4_error_"+str(args.SEED), model.best_abs_error)
    if args.plot_error:
        model.error_plot_sxpinn()
    if args.plot_loss:
        model.plot_loss_sxpinn()
    if args.save_error:
        model.error_save_sxpinn()
    if args.save_gate:
        model.save_gate()
    if args.save_loss:
        model.saved_loss = np.array(model.saved_loss)
        model.saved_l2 = np.array(model.saved_l2)
        np.savetxt("saved_loss_l2/BB4_loss_"+args.SEED+".txt", model.saved_loss)
        np.savetxt("saved_loss_l2/BB4_l2_"+args.SEED+".txt", model.saved_l2)