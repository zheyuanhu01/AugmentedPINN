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
parser.add_argument('--SEED', type=int, default=0) # Random seed
parser.add_argument('--dataset', type=str, default='Wave') # 'burgers', 'poisson', 'helmholtz', 'KG', 'Wave', 'Advection'
parser.add_argument('--adv_beta', type=float, default=40) 
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
parser.add_argument('--XPINN_L', type=int, default=6)

parser.add_argument('--SXPINN_G_h', type=int, default=20)
parser.add_argument('--SXPINN_G_L', type=int, default=2)
parser.add_argument('--SXPINN_E_h', type=int, default=20)
parser.add_argument('--SXPINN_E_L', type=int, default=4)
parser.add_argument('--SXPINN_h_h', type=int, default=20)
parser.add_argument('--SXPINN_h_L', type=int, default=3)
parser.add_argument('--SXPINN_gate_trainable', type=bool, default=True)

parser.add_argument('--PINN_weights', type=list, default=[1, 20])
parser.add_argument('--XPINN_weights', type=list, default=[1, 0, 20, 20, 1]) # [1, 0, 20, 20, 1]; [1, 1, 20, 20, 0]
## R; R-interface; B; B-interface; R-additional-interface

parser.add_argument('--Fourier', type=int, default=0)
parser.add_argument('--sigma', type=float, default=2.5)
parser.add_argument('--F_trainable', type=bool, default=False)

parser.add_argument('--plot_error', type=bool, default=False)
parser.add_argument('--plot_loss', type=bool, default=False)
parser.add_argument('--save_loss', type=bool, default=False)
parser.add_argument('--save_error', type=bool, default=False)
parser.add_argument('--save_gate', type=bool, default=False)
parser.add_argument('--save_subnet', type=bool, default=False)

args = parser.parse_args()


torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
device = torch.device(args.device)
INIT_DATASETS = ['KG', 'Wave']

if args.XPINN_weights == [1, 1, 20, 20, 0]:
    args.XPINN_type = "v1"
else:
    args.XPINN_type = "v2"

print(args)


def load_data_burgers(XPINN_type=1, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1

    data = scipy.io.loadmat('burgers_shock.mat') # You can download data at https://github.com/maziarraissi/PINNs/tree/master/main/Data
    y = data['t'].flatten() # 100; range [0, 1]
    x = data['x'].flatten() # 256; range [-1, 1]
    u = np.real(data['usol']) # 256, 100

    ### Boundary
    xb = np.concatenate([x, np.zeros(len(y))+1, np.zeros(len(y))-1])
    yb = np.concatenate([np.zeros(len(x)), y, y])
    ub = np.concatenate([u[:, 0], u[-1, :], u[0, :]])

    if XPINN_type == 1:
        threshold = 0
        xi1 = np.zeros((1000)) + threshold
        yi1 = np.linspace(0, 1, 1000)
        fi1 = np.zeros((1000))
    elif XPINN_type == 2:
        threshold = 0.5
        xi1 = np.linspace(-1, 1, 1000)
        yi1 = np.zeros((1000)) + threshold
        fi1 = np.zeros((1000))

    y, x = np.meshgrid(y, x)
    x, y, u = x.reshape(-1), y.reshape(-1), u.reshape(-1)
    f = np.zeros(len(x))

    ### Partition residual test points for SubNet 1
    if XPINN_type == 1:
        index_test_1 = np.where(x > threshold)[0]
    elif XPINN_type == 2:
        index_test_1 = np.where(y > threshold)[0]
    x1 = x[index_test_1]
    y1 = y[index_test_1]
    u1 = u[index_test_1]
    f1 = f[index_test_1]

    ### Partition residual test points for SubNet 2
    if XPINN_type == 1:
        index_test_2 = np.where(x <= threshold)[0]
    elif XPINN_type == 2:
        index_test_2 = np.where(y <= threshold)[0]
    x2 = x[index_test_2]
    y2 = y[index_test_2]
    u2 = u[index_test_2]
    f2 = f[index_test_2]

    ### Subdomain 1
    if XPINN_type == 1:
        index = np.where(xb > threshold)[0]
    elif XPINN_type == 2:
        index = np.where(yb > threshold)[0]
    xb1 = xb[index]
    yb1 = yb[index]
    ub1 = ub[index]

    ### Subdomain 2
    if XPINN_type == 1:
        index = np.where(xb <= threshold)[0]
    elif XPINN_type == 2:
        index = np.where(yb <= threshold)[0]
    xb2 = xb[index]
    yb2 = yb[index]
    ub2 = ub[index]

    N_u = 300 # Number of boundary points
    N_f = 10000 # Number of collocation points

    if XPINN_type == 1:
        N_u_1, N_u_2 = 150, 150 # Number of boundary points for each sub-domain
    elif XPINN_type == 2:
        N_u_1, N_u_2 = 98, 300-98 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u(x,t)')
        plt.colorbar()
        plt.show()

    return x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1

def load_data_poisson(save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    def func(x, y):
        ret = (x <= 0.75) & (x >= 0.25) & (y <= 0.75) & (y >= 0.25)
        return ret + 0.0

    def import_hdf5(filename):
        f = h5py.File(filename, 'r')
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        return np.array(data)

    num = 1001

    up = 0.75
    down = 0.25

    data = import_hdf5("Poisson.hdf5") # 1001 X 1001 X 3
    data = data.reshape(-1, 3)
    x, y, u = data[:, 0], data[:, 1], data[:, 2]
    x, y, u = x.reshape(-1), y.reshape(-1), u.reshape(-1)
    f = func(x, y)

    ### Boundary
    xb = np.concatenate([np.zeros((num)), np.ones((num)), np.linspace(0, 1, num), np.linspace(0, 1, num)], axis=0)
    yb = np.concatenate([np.linspace(0, 1, num), np.linspace(0, 1, num), np.zeros((num)), np.ones((num))], axis=0) 
    ub = np.zeros_like(xb)

    ### Partition residual test points for SubNet 1
    index_test_1 = np.where((x <= up) & (x >= down) & (y <= up) & (y >= down))[0]
    x1 = x[index_test_1]
    y1 = y[index_test_1]
    u1 = u[index_test_1]
    f1 = f[index_test_1]

    ### Partition residual test points for SubNet 2
    index_test_2 = np.where((x > up) | (x < down) | (y > up) | (y < down))[0]
    x2 = x[index_test_2]
    y2 = y[index_test_2]
    u2 = u[index_test_2]
    f2 = f[index_test_2]

    ### Interface
    xi1 = np.concatenate([np.zeros((num)) + down, np.zeros((num)) + up, \
        np.linspace(down, up, num), np.linspace(down, up, num)], axis=0)
    yi1 = np.concatenate([np.linspace(down, up, num), np.linspace(down, up, num), \
        np.zeros((num)) + down, np.zeros((num)) + up], axis=0)
    fi1 = func(xi1, yi1)

    ### Subdomain 1
    index = np.where((xb <= up) & (xb >= down) & (yb <= up) & (yb >= down))[0]
    xb1 = xb[index]
    yb1 = yb[index]
    ub1 = np.zeros_like(xb1)

    ### Subdomain 2
    index = np.where((xb > up) | (xb < down) | (yb > up) | (yb < down))[0]
    xb2 = xb[index]
    yb2 = yb[index]
    ub2 = np.zeros_like(xb2)

    N_u = 80 # Number of boundary points
    N_f = 400 # Number of collocation points
    N_i = 1000 # Number of collocation points

    N_u_1, N_u_2 = 0, 80 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 100, 300 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    ### No boundary values for subnet 1

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    # INterface selection
    idx = np.random.choice(xi1.shape[0], N_i, replace=False)
    xi1 = xi1[idx]; yi1 = yi1[idx]; fi1 = fi1[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Poisson u(x,y)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Poisson u(x,y)')
        plt.colorbar()
        plt.show()
    return x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1

def load_data_helmholtz(a1=1, a2=4, k=1, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    def func_u(x, y):
        return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)
    def func_q(x, y):
        return (- (a1 * np.pi)**2 - (a2 * np.pi)**2 + k**2) * np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)
        
    x, y = np.linspace(-1, 1, 201), np.linspace(-1, 1, 201); x, y = np.meshgrid(x, y); x, y = x.reshape(-1), y.reshape(-1);
    u, f = func_u(x, y), func_q(x, y)
    xb = np.concatenate([np.linspace(-1, 1, 201), np.linspace(-1, 1, 201), - np.ones(201), np.ones(201)], axis=0)
    yb = np.concatenate([- np.ones(201), np.ones(201), np.linspace(-1, 1, 201), np.linspace(-1, 1, 201)], axis=0)
    ub = func_u(xb, yb)

    xi1 = np.linspace(-1, 1, 400); yi1 = np.zeros((400)); fi1 = func_q(xi1, yi1); 

    index_test_1 = np.where(y > 0)[0]; x1 = x[index_test_1]; y1 = y[index_test_1]; u1 = u[index_test_1]; f1 = f[index_test_1];
    index_test_2 = np.where(y <= 0)[0]; x2 = x[index_test_2]; y2 = y[index_test_2]; u2 = u[index_test_2]; f2 = f[index_test_2]
    index = np.where(yb > 0)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
    index = np.where(yb <= 0)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];

    N_u = 400 # Number of boundary points
    N_f = 10000 # Number of collocation points

    N_u_1, N_u_2 = 200, 200 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Helmholtz u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Helmholtz u(x,t)')
        plt.colorbar()
        plt.show()
    return x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1

def load_data_KG(save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    x, y = sy.symbols('x, y'); u = x * sy.sin(5 * np.pi * y) + (x * y) ** 3; eq = u.diff(y).diff(y) - u.diff(x).diff(x) + u * u * u; n = u.diff(y);
    func_f = sy.lambdify([x, y], eq,'numpy')
    func_u = sy.lambdify([x, y], u,'numpy')
    func_n =  sy.lambdify([x, y], n,'numpy')

    x, y = np.linspace(0, 1, 201), np.linspace(0, 1, 201); x, y = np.meshgrid(x, y); x, y = x.reshape(-1), y.reshape(-1);
    u, f = func_u(x, y), func_f(x, y)
    xb = np.concatenate([np.linspace(0, 1, 201), np.zeros(201), np.ones(201)], axis=0)
    yb = np.concatenate([np.zeros(201), np.linspace(0, 1, 201), np.linspace(0, 1, 201)], axis=0)
    ub = func_u(xb, yb)
    xn = np.linspace(0, 1, 1001); yn = np.zeros(1001); fn = func_n(xn, yn);

    yi1 = np.linspace(0, 1, 400); xi1 = np.zeros((400)) + 0.5; fi1 = func_f(xi1, yi1); 

    index = np.where(x > 0.5)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
    index = np.where(x <= 0.5)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
    index = np.where(xb > 0.5)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
    index = np.where(xb <= 0.5)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];
    index = np.where(xn > 0.5)[0]; xn1 = xn[index]; yn1 = yn[index]; fn1 = fn[index];
    index = np.where(xn <= 0.5)[0]; xn2 = xn[index]; yn2 = yn[index]; fn2 = fn[index];

    N_u = 400 # Number of boundary points
    N_n = 200
    N_f = 10000 # Number of collocation points

    N_u_1, N_u_2 = 200, 200 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain
    N_n_1, N_n_2 = 100, 100

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    # Initial PINN selection
    idx = np.random.choice(fn.shape[0], N_n, replace=False)
    fn = fn[idx]; xn = xn[idx]; yn = yn[idx];

    # Initial XPINN 1 selection
    idx = np.random.choice(fn1.shape[0], N_n_1, replace=False)
    fn1 = fn1[idx]; xn1 = xn1[idx]; yn1 = yn1[idx];

    # Initial XPINN 2 selection
    idx = np.random.choice(fn2.shape[0], N_n_2, replace=False)
    fn2 = fn2[idx]; xn2 = xn2[idx]; yn2 = yn2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.scatter(xn1,yn1,s=3)
        plt.scatter(xn2,yn2,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Klein-Gordon u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Klein-Gordon u(x,t)')
        plt.colorbar()
        plt.show()
    return x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1, \
                    xn, yn, fn, xn1, yn1, fn1, xn2, yn2, fn2

def load_data_wave(XPINN_type=2, T=1, threshold=0.5, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    x, y = sy.symbols('x, y'); u = sy.sin(np.pi * x) * sy.cos(2 * np.pi * y) + 0.0 * sy.sin(4 * np.pi * x) * sy.cos(8 * np.pi * y); eq = u.diff(y).diff(y) - 4 *u.diff(x).diff(x); n = u.diff(y);
    func_u = sy.lambdify([x, y], u,'numpy')
    func_n =  sy.lambdify([x, y], n,'numpy')

    x, y = np.linspace(0, 1, 201), np.linspace(0, T, 201); x, y = np.meshgrid(x, y); x, y = x.reshape(-1), y.reshape(-1);
    u, f = func_u(x, y), np.zeros_like(x)
    xb = np.concatenate([np.linspace(0, 1, 201), np.zeros(201), np.ones(201)], axis=0)
    yb = np.concatenate([np.zeros(201), np.linspace(0, T, 201), np.linspace(0, T, 201)], axis=0)
    ub = func_u(xb, yb)
    xn = np.linspace(0, 1, 1001); yn = np.zeros(1001); fn = func_n(xn, yn);
    '''if XPINN_type == 1:
        yi1 = np.linspace(0, 1, 400); xi1 = np.zeros((400)) + 0.5; fi1 = np.zeros_like(xi1); 
        index = np.where(x > 0.5)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
        index = np.where(x <= 0.5)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
        index = np.where(xb > 0.5)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
        index = np.where(xb <= 0.5)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];
        index = np.where(xn > 0.5)[0]; xn1 = xn[index]; yn1 = yn[index]; fn1 = fn[index];
        index = np.where(xn <= 0.5)[0]; xn2 = xn[index]; yn2 = yn[index]; fn2 = fn[index];
    elif XPINN_type == 2:'''
    xi1 = np.linspace(0, 1, 400); yi1 = np.zeros((400)) + threshold; fi1 = np.zeros_like(xi1); 
    index = np.where(y > threshold)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
    index = np.where(y <= threshold)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
    index = np.where(yb > threshold)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
    index = np.where(yb <= threshold)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];
    index = np.where(yn > threshold)[0]; xn1 = xn[index]; yn1 = yn[index]; fn1 = fn[index];
    index = np.where(yn <= threshold)[0]; xn2 = xn[index]; yn2 = yn[index]; fn2 = fn[index];

    N_u = 400 # Number of boundary points
    N_n = 200
    N_f = 10000 # Number of collocation points

    N_u_1, N_u_2 = 200, 200 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain
    if XPINN_type == 1:
        N_n_1, N_n_2 = 100, 100
    elif XPINN_type == 2:
        N_n_1, N_n_2 = 0, 200

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    # Initial PINN selection
    idx = np.random.choice(fn.shape[0], N_n, replace=False)
    fn = fn[idx]; xn = xn[idx]; yn = yn[idx];

    # Initial XPINN 1 selection
    idx = np.random.choice(fn1.shape[0], N_n_1, replace=False)
    fn1 = fn1[idx]; xn1 = xn1[idx]; yn1 = yn1[idx];

    # Initial XPINN 2 selection
    idx = np.random.choice(fn2.shape[0], N_n_2, replace=False)
    fn2 = fn2[idx]; xn2 = xn2[idx]; yn2 = yn2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.scatter(xn1,yn1,s=3)
        plt.scatter(xn2,yn2,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Wave u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Wave u(x,t)')
        plt.colorbar()
        plt.show()
    return x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1, \
                    xn, yn, fn, xn1, yn1, fn1, xn2, yn2, fn2

def load_data_advection(XPINN_type=2, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    x, y = sy.symbols('x, y'); u = - sy.sin(np.pi * x - args.adv_beta * np.pi * y);
    func_u = sy.lambdify([x, y], u,'numpy')

    x, y = np.linspace(-1, 1, 201), np.linspace(0, 1, 201); x, y = np.meshgrid(x, y); x, y = x.reshape(-1), y.reshape(-1);
    u, f = func_u(x, y), np.zeros_like(x)
    #xb = np.linspace(-1, 1, 1001)
    xb = np.concatenate([np.linspace(-1, 1, 401), - np.ones(401), np.ones(401)], axis=0)
    #yb = np.zeros(1001)
    yb = np.concatenate([np.zeros(401), np.linspace(0, 1, 401), np.linspace(0, 1, 401)], axis=0)
    ub = func_u(xb, yb)
    if XPINN_type == 1:
        yi1 = np.linspace(0, 1, 400); xi1 = np.zeros((400)); fi1 = np.zeros_like(xi1); 
        index = np.where(x > 0.)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
        index = np.where(x <= 0.)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
        index = np.where(xb > 0.)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
        index = np.where(xb <= 0.)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];
    elif XPINN_type == 2:
        xi1 = np.linspace(-1, 1, 400); yi1 = np.zeros((400)) + 0.5; fi1 = np.zeros_like(xi1); 
        index = np.where(y > 0.5)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
        index = np.where(y <= 0.5)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
        index = np.where(yb > 0.5)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
        index = np.where(yb <= 0.5)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];

    N_u = 400 # Number of boundary points
    N_f = 10000 # Number of collocation points
    N_u_1, N_u_2 = 200, 200 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Wave u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Advection u(x,t)')
        plt.colorbar()
        plt.show()
    return x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1

def load_data_advection_square_pulse(XPINN_type=2, save_fig=False):
    args.input_dim = 2
    args.output_dim = 1
    x, y = sy.symbols('x, y'); u = (x - 0.5 * y < 0.2) & (x - 0.5 * y > -0.2)
    func_u = sy.lambdify([x, y], u,'numpy')

    x, y = np.linspace(-1, 1, 201), np.linspace(0, 1, 201); x, y = np.meshgrid(x, y); x, y = x.reshape(-1), y.reshape(-1);
    u, f = func_u(x, y) + 0.0, np.zeros_like(x)
    #xb = np.linspace(-1, 1, 1001)
    xb = np.concatenate([np.linspace(-1, 1, 401), - np.ones(401), np.ones(401)], axis=0)
    #yb = np.zeros(1001)
    yb = np.concatenate([np.zeros(401), np.linspace(0, 1, 401), np.linspace(0, 1, 401)], axis=0)
    ub = func_u(xb, yb)
    if XPINN_type == 1:
        yi1 = np.linspace(0, 1, 400); xi1 = np.zeros((400)); fi1 = np.zeros_like(xi1); 
        index = np.where(x > 0.)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
        index = np.where(x <= 0.)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
        index = np.where(xb > 0.)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
        index = np.where(xb <= 0.)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];
    elif XPINN_type == 2:
        xi1 = np.linspace(-1, 1, 400); yi1 = np.zeros((400)) + 0.5; fi1 = np.zeros_like(xi1); 
        index = np.where(y > 0.5)[0]; x1 = x[index]; y1 = y[index]; u1 = u[index]; f1 = f[index];
        index = np.where(y <= 0.5)[0]; x2 = x[index]; y2 = y[index]; u2 = u[index]; f2 = f[index];
        index = np.where(yb > 0.5)[0]; xb1 = xb[index]; yb1 = yb[index]; ub1 = ub[index];
        index = np.where(yb <= 0.5)[0]; xb2 = xb[index]; yb2 = yb[index]; ub2 = ub[index];

    N_u = 400 # Number of boundary points
    N_f = 10000 # Number of collocation points
    N_u_1, N_u_2 = 200, 200 # Number of boundary points for each sub-domain
    N_f_1, N_f_2 = 5000, 5000 # Number of residual points for each sub-domain

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    uf = u[idx]; ff = f[idx]; xf = x[idx]; yf = y[idx];

    # Collocation XPINN 1 selection
    idx = np.random.choice(u1.shape[0], N_f_1, replace=False)
    uf1 = u1[idx]; ff1 = f1[idx]; xf1 = x1[idx]; yf1 = y1[idx];

    # Collocation XPINN 2 selection
    idx = np.random.choice(u2.shape[0], N_f_2, replace=False)
    uf2 = u2[idx]; ff2 = f2[idx]; xf2 = x2[idx]; yf2 = y2[idx];

    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx];

    # Boundary XPINN 1 selection
    idx = np.random.choice(ub1.shape[0], N_u_1, replace=False)
    ub1 = ub1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

    # Boundary XPINN 2 selection
    idx = np.random.choice(ub2.shape[0], N_u_2, replace=False)
    ub2 = ub2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf1,yf1,s=3)
        plt.scatter(xf2,yf2,s=3)
        plt.scatter(xi1,yi1,s=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Advection u(x,t)')
        plt.scatter(xb1,yb1,c='b',s=3)
        plt.scatter(xb2,yb2,c='k',s=3)
        plt.legend(['Subnet 1 Residual','Subnet 2 Residual', 'Interface', 'Subnet 1 Boundary', 'Subnet 2 Boundary'])
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
        plt.tricontourf(triang_coarse, np.concatenate((u1,u2)), 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Advection u(x,t)')
        plt.colorbar()
        plt.show()

    return x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1

if args.dataset == 'burgers':
    x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1 = load_data_burgers()
elif args.dataset == 'poisson':
    x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1 = load_data_poisson()
elif args.dataset == 'helmholtz':
    x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1 = load_data_helmholtz()
elif args.dataset == 'KG':
    x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1, \
                    xn, yn, fn, xn1, yn1, fn1, xn2, yn2, fn2 = load_data_KG()
elif args.dataset == 'Wave':
    x, y, u, x1, y1, u1, x2, y2, u2, \
        xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, \
            xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, \
                xi1, yi1, fi1, \
                    xn, yn, fn, xn1, yn1, fn1, xn2, yn2, fn2 = load_data_wave(T=1, threshold=0.5)
elif args.dataset == 'Advection':
    x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1 = load_data_advection()
elif args.dataset == 'Advection_SP':
    x, y, u, x1, y1, u1, x2, y2, u2, xb, yb, ub, xb1, yb1, ub1, xb2, yb2, ub2, xf, yf, ff, xf1, yf1, ff1, xf2, yf2, ff2, xi1, yi1, fi1 = load_data_advection_square_pulse()

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
        x = nn.Sigmoid()(x)
        return x

class SXPINN_Net(nn.Module):
    def __init__(self, E_layers, G_layers, h_layers, act=nn.Tanh()):
        super(SXPINN_Net, self).__init__()
        self.u1_net, self.u2_net = Net(E_layers, act), Net(E_layers, act)
        self.w_gate = Gate_Net(G_layers, act)
        self.h = Net(h_layers, act)
        self.act = act
        
    def forward(self, x):
        out = self.w_gate(x) * self.u1_net(self.act(self.h(x))) + (1 - self.w_gate(x)) * self.u2_net(self.act(self.h(x)))
        return out
    
    def u1(self, x):
        return self.w_gate(x) * self.u1_net(self.act(self.h(x)))
    
    def u2(self, x):
        return (1 - self.w_gate(x)) * self.u2_net(self.act(self.h(x)))

'''class Fourier_Layer(nn.Module):
    def __init__(self, m, sigma, trainable):
        super(Fourier_Layer, self).__init__()
        if trainable:
            self.B = nn.Parameter(torch.randn(args.input_dim, m) * sigma)
        else:
            self.B = (torch.randn(args.input_dim, m) * sigma).to(device)
        
    def forward(self, x):
        y = torch.matmul(x, self.B)
        y = torch.cat([torch.cos(2 * np.pi * y), torch.sin(2 * np.pi * y), x], dim=-1)
        return y'''

class Fourier_Layer(nn.Module):
    def __init__(self, m, sigma, trainable):
        super(Fourier_Layer, self).__init__()
        if trainable:
            self.B = nn.Parameter(torch.randn(args.input_dim, m) * sigma)
        else:
            self.B1 = (torch.randn(args.input_dim, m) * sigma).to(device)
            self.B2 = (torch.randn(args.input_dim, m) * sigma).to(device)
        
    def forward(self, x):
        y1, y2 = torch.matmul(x, self.B1), torch.matmul(x, self.B2)
        x = torch.cat([torch.cos(2 * np.pi * y1) * torch.sin(2 * np.pi * y2), x], dim=-1)
        return x

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
        
        # boundary points --- mse
        # PINN
        #self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yb = torch.unsqueeze(torch.tensor(yb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        # XPINN
        #self.Xb1 = torch.tensor(Xb1, dtype=torch.float32)
        self.xb1 = torch.unsqueeze(torch.tensor(xb1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yb1 = torch.unsqueeze(torch.tensor(yb1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub1 = torch.unsqueeze(torch.tensor(ub1, dtype=torch.float32, requires_grad=True),-1).to(device)
        #self.Xb2 = torch.tensor(Xb2, dtype=torch.float32)
        self.xb2 = torch.unsqueeze(torch.tensor(xb2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yb2 = torch.unsqueeze(torch.tensor(yb2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub2 = torch.unsqueeze(torch.tensor(ub2, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        # collocation points --- residual
        # PINN
        self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yf = torch.unsqueeze(torch.tensor(yf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ff = torch.unsqueeze(torch.tensor(ff, dtype=torch.float32, requires_grad=True),-1).to(device)
        # XPINN
        self.xf1 = torch.unsqueeze(torch.tensor(xf1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yf1 = torch.unsqueeze(torch.tensor(yf1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ff1 = torch.unsqueeze(torch.tensor(ff1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.xf2 = torch.unsqueeze(torch.tensor(xf2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yf2 = torch.unsqueeze(torch.tensor(yf2, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ff2 = torch.unsqueeze(torch.tensor(ff2, dtype=torch.float32, requires_grad=True),-1).to(device)

        # interface points --- residual
        self.xi1 = torch.unsqueeze(torch.tensor(xi1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yi1 = torch.unsqueeze(torch.tensor(yi1, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.fi1 = torch.unsqueeze(torch.tensor(fi1, dtype=torch.float32, requires_grad=True),-1).to(device)

        if args.dataset in INIT_DATASETS:
            # PINN
            self.xn = torch.unsqueeze(torch.tensor(xn, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.yn = torch.unsqueeze(torch.tensor(yn, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.fn = torch.unsqueeze(torch.tensor(fn, dtype=torch.float32, requires_grad=True),-1).to(device)
            # XPINN
            self.xn1 = torch.unsqueeze(torch.tensor(xn1, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.yn1 = torch.unsqueeze(torch.tensor(yn1, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.fn1 = torch.unsqueeze(torch.tensor(fn1, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.xn2 = torch.unsqueeze(torch.tensor(xn2, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.yn2 = torch.unsqueeze(torch.tensor(yn2, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.fn2 = torch.unsqueeze(torch.tensor(fn2, dtype=torch.float32, requires_grad=True),-1).to(device)
          
        # Initalize Neural Networks
        if args.Fourier == 0:
            layers = [args.input_dim] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
            sub_layers = [args.input_dim] + [args.XPINN_h] * (args.XPINN_L - 1) + [args.output_dim]
        else:
            layers = [args.Fourier + 2] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
            sub_layers = [args.Fourier + 2] + [args.XPINN_h] * (args.XPINN_L - 1) + [args.output_dim]
        if args.Fourier == 0:
            self.u_net = Net(layers).to(device)
            self.u1_net = Net(sub_layers).to(device)
            self.u2_net = Net(sub_layers).to(device)
        else:
            self.u_net = nn.Sequential(Fourier_Layer(args.Fourier, args.sigma, args.F_trainable), Net(layers)).to(device)
            self.u1_net = nn.Sequential(Fourier_Layer(args.Fourier, args.sigma, args.F_trainable), Net(sub_layers)).to(device)
            self.u2_net = nn.Sequential(Fourier_Layer(args.Fourier, args.sigma, args.F_trainable), Net(sub_layers)).to(device)
        
        self.net_params_pinn = list(self.u_net.parameters())
        self.net_params_xpinn = list(self.u1_net.parameters()) + list(self.u2_net.parameters())

        self.loss_pinn = []
        self.loss_xpinn = []
        self.loss = 10

        self.best_l2_error = 1
        self.best_abs_error = 0
        
    def Burgers(self, u, u_x, u_y, u_xx): 
        return u_y + u * u_x - (0.01 / np.pi) * u_xx
    
    def Poisson(self, u_xx, u_yy, f): 
        return u_yy + u_xx - f
        
    def Helmholtz(self, u, u_xx, u_yy, f): 
        return u_yy + u_xx + u - f

    def KG(self, u, u_xx, u_yy, f):
        return u_yy - u_xx + u**3 - f
    
    def Wave(self, u_xx, u_yy):
        return u_yy - 4 * u_xx
    
    def Advection(self, u_x, u_y):
        return u_y + args.adv_beta * u_x

    def num_params(self):
        num_pinn, num_xpinn = 0, 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        for p in self.net_params_xpinn:
            num_xpinn += len(p.reshape(-1))
        return num_pinn, num_xpinn

    def get_loss_pinn(self):
        # Boundry loss
        mse_ub = (self.ub - self.u_net(torch.cat((self.xb, self.yb), 1))).square().mean()
        if args.dataset in INIT_DATASETS:
            un = self.u_net(torch.cat((self.xn, self.yn), 1))
            un_y = torch.autograd.grad(un.sum(), self.yn, create_graph=True)[0]
            mse_ub += (self.fn - un_y).square().mean()
        
        # Residual/collocation loss
        u = self.u_net(torch.cat((self.xf, self.yf), 1))
        u_sum = u.sum()
        if args.dataset == 'burgers':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            # Residuals
            f = self.Burgers(u, u_x, u_y, u_xx)
        elif args.dataset == 'poisson':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Poisson(u_xx, u_yy, self.ff)
        elif args.dataset == 'helmholtz':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Helmholtz(u, u_xx, u_yy, self.ff)
        elif args.dataset == 'KG':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.KG(u, u_xx, u_yy, self.ff)
        elif args.dataset == 'Wave':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Wave(u_xx, u_yy)
        elif args.dataset == 'Advection':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            # Residuals
            f = self.Advection(u_x, u_y)

        mse_f = f.square().mean()
    
        # Sum losses
        net_loss = self.pinn_w_b*mse_ub + self.pinn_w_rc*mse_f
        
        loss = net_loss 
        return loss
    
    def get_loss_xpinn(self):
        # Boundry loss
        if len(self.xb1) > 0:
            mse_ub1 = (self.ub1 - self.u1_net(torch.cat((self.xb1, self.yb1), 1))).square().mean() 
        else:
            mse_ub1 = 0
        if len(self.xb2) > 0:
            mse_ub2 = (self.ub2 - self.u2_net(torch.cat((self.xb2, self.yb2), 1))).square().mean() 
        else:
            mse_ub2 = 0
        if args.dataset in INIT_DATASETS: # if KG, then initial condition with differention
            if len(self.xn1) > 0:
                un1 = self.u1_net(torch.cat((self.xn1, self.yn1), 1))
                un1_y = torch.autograd.grad(un1.sum(), self.yn1, create_graph=True)[0]
                mse_ub1 += (self.fn1 - un1_y).square().mean()
            if len(self.xn2) > 0:
                un2 = self.u2_net(torch.cat((self.xn2, self.yn2), 1))
                un2_y = torch.autograd.grad(un2.sum(), self.yn2, create_graph=True)[0]
                mse_ub2 += (self.fn2 - un2_y).square().mean()
        
        # Residual/collocation loss
        # Sub-net 1 residual
        u1 = self.u1_net(torch.cat((self.xf1, self.yf1), 1))
        u1_sum = u1.sum()
        # Sub-net 2 residual
        u2 = self.u2_net(torch.cat((self.xf2, self.yf2), 1))
        u2_sum = u2.sum()
        # Sub-net 1, Interface 1
        u1i1 = self.u1_net(torch.cat((self.xi1, self.yi1), 1))
        u1i1_sum = u1i1.sum()
        # Sub-net 2, Interface 1
        u2i1 = self.u2_net(torch.cat((self.xi1, self.yi1), 1))
        u2i1_sum = u2i1.sum()
        if args.dataset == 'burgers':
            u1_x = torch.autograd.grad(u1_sum, self.xf1, create_graph=True)[0]
            u1_y = torch.autograd.grad(u1_sum, self.yf1, create_graph=True)[0]
            u1_xx = torch.autograd.grad(u1_x.sum(), self.xf1, create_graph=True)[0]
            u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
            u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
            u2_xx = torch.autograd.grad(u2_x.sum(), self.xf2, create_graph=True)[0]
            f1 = self.Burgers(u1, u1_x, u1_y, u1_xx)
            f2 = self.Burgers(u2, u2_x, u2_y, u2_xx)
            if self.xpinn_w_ri > 0:
                u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
                u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
                u1i1_xx = torch.autograd.grad(u1i1_x.sum(), self.xi1, create_graph=True)[0]
                u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
                u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
                u2i1_xx = torch.autograd.grad(u2i1_x.sum(), self.xi1, create_graph=True)[0]
                fi1 = self.Burgers(u1i1, u1i1_x, u1i1_y, u1i1_xx) - self.Burgers(u2i1, u2i1_x, u2i1_y, u2i1_xx) 
            if self.xpinn_w_ra > 0:
                u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
                u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
                u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
                u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
                mse_fai1 = ((u1i1_x-u2i1_x).square().mean() + (u1i1_y-u2i1_y).square().mean()) / 2
        elif args.dataset == 'poisson':
            u1_x = torch.autograd.grad(u1_sum, self.xf1, create_graph=True)[0]
            u1_y = torch.autograd.grad(u1_sum, self.yf1, create_graph=True)[0]
            u1_xx = torch.autograd.grad(u1_x.sum(), self.xf1, create_graph=True)[0]
            u1_yy = torch.autograd.grad(u1_y.sum(), self.yf1, create_graph=True)[0]
            u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
            u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
            u2_xx = torch.autograd.grad(u2_x.sum(), self.xf2, create_graph=True)[0]
            u2_yy = torch.autograd.grad(u2_y.sum(), self.yf2, create_graph=True)[0]
            f1 = self.Poisson(u1_xx, u1_yy, self.ff1)
            f2 = self.Poisson(u2_xx, u2_yy, self.ff2)
            u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
            u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
            u1i1_xx = torch.autograd.grad(u1i1_x.sum(), self.xi1, create_graph=True)[0]
            u1i1_yy = torch.autograd.grad(u1i1_y.sum(), self.yi1, create_graph=True)[0]
            u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
            u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
            u2i1_xx = torch.autograd.grad(u2i1_x.sum(), self.xi1, create_graph=True)[0]
            u2i1_yy = torch.autograd.grad(u2i1_y.sum(), self.yi1, create_graph=True)[0]
            fi1 = (self.Poisson(u1i1_xx, u1i1_yy, self.fi1) - self.Poisson(u2i1_xx, u2i1_yy, self.fi1) )
            mse_fai1 = ((u1i1_x-u2i1_x).square().mean() + (u1i1_y-u2i1_y).square().mean()) / 2
        elif args.dataset in ['helmholtz', 'KG', 'Wave']:
            u1_x = torch.autograd.grad(u1_sum, self.xf1, create_graph=True)[0]
            u1_y = torch.autograd.grad(u1_sum, self.yf1, create_graph=True)[0]
            u1_xx = torch.autograd.grad(u1_x.sum(), self.xf1, create_graph=True)[0]
            u1_yy = torch.autograd.grad(u1_y.sum(), self.yf1, create_graph=True)[0]
            u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
            u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
            u2_xx = torch.autograd.grad(u2_x.sum(), self.xf2, create_graph=True)[0]
            u2_yy = torch.autograd.grad(u2_y.sum(), self.yf2, create_graph=True)[0]
            if args.dataset == 'helmholtz':
                f1 = self.Helmholtz(u1, u1_xx, u1_yy, self.ff1)
                f2 = self.Helmholtz(u2, u2_xx, u2_yy, self.ff2)
            elif args.dataset == 'KG':
                f1 = self.KG(u1, u1_xx, u1_yy, self.ff1)
                f2 = self.KG(u2, u2_xx, u2_yy, self.ff2)
            elif args.dataset == 'Wave':
                f1 = self.Wave(u1_xx, u1_yy)
                f2 = self.Wave(u2_xx, u2_yy)
            if self.xpinn_w_ri > 0:
                u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
                u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
                u1i1_xx = torch.autograd.grad(u1i1_x.sum(), self.xi1, create_graph=True)[0]
                u1i1_yy = torch.autograd.grad(u1i1_y.sum(), self.yi1, create_graph=True)[0]
                u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
                u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
                u2i1_xx = torch.autograd.grad(u2i1_x.sum(), self.xi1, create_graph=True)[0]
                u2i1_yy = torch.autograd.grad(u2i1_y.sum(), self.yi1, create_graph=True)[0]
                if args.dataset == 'helmholtz':
                    fi1 = (self.Helmholtz(u1i1, u1i1_xx, u1i1_yy, self.fi1) - self.Helmholtz(u2i1, u2i1_xx, u2i1_yy, self.fi1) )
                elif args.dataset == 'KG':
                    fi1 = (self.KG(u1i1, u1i1_xx, u1i1_yy, self.fi1) - self.KG(u2i1, u2i1_xx, u2i1_yy, self.fi1) )
                elif args.dataset == 'Wave':
                    fi1 = (self.Wave(u1i1_xx, u1i1_yy) - self.Wave(u2i1_xx, u2i1_yy) )
            if self.xpinn_w_ra > 0:
                u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
                u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
                u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
                u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
                mse_fai1 = ((u1i1_x-u2i1_x).square().mean() + (u1i1_y-u2i1_y).square().mean()) / 2
        elif args.dataset == 'Advection':
            u1_x = torch.autograd.grad(u1_sum, self.xf1, create_graph=True)[0]
            u1_y = torch.autograd.grad(u1_sum, self.yf1, create_graph=True)[0]
            u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
            u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
            f1 = self.Advection(u1_x, u1_y)
            f2 = self.Advection(u2_x, u2_y)
            u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
            u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
            u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
            u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
            fi1 = (self.Advection(u1i1_x, u1i1_y) - self.Advection(u2i1_x, u2i1_y) )
            mse_fai1 = ((u1i1_x-u2i1_x).square().mean() + (u1i1_y-u2i1_y).square().mean()) / 2

        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1)/2  
        
        mse_u1avgi1 = (u1i1-uavgi1).square().mean()
        mse_u2avgi1 = (u2i1-uavgi1).square().mean()

        # Residuals 
        mse_f1 = f1.square().mean()
        mse_f2 = f2.square().mean()

        # Residual continuity conditions on the interfaces
        if self.xpinn_w_ri > 0:
            mse_fi1 = fi1.square().mean()
        else:
            mse_fi1 = 0
        if self.xpinn_w_ra == 0:
            mse_fai1 = 0
        
        # Sum losses
        net1_loss = self.xpinn_w_b*mse_ub1 + self.xpinn_w_rc*mse_f1 + self.xpinn_w_ri*mse_fi1 + self.xpinn_w_i*mse_u1avgi1
        net2_loss = self.xpinn_w_b*mse_ub2 + self.xpinn_w_rc*mse_f2 + self.xpinn_w_ri*mse_fi1 + self.xpinn_w_i*mse_u2avgi1

        #if args.dataset in ['poisson', 'helmholtz', 'KG']:
        net1_loss, net2_loss = net1_loss + self.xpinn_w_ra*mse_fai1, net2_loss + self.xpinn_w_ra*mse_fai1
        
        loss = net1_loss + net2_loss
        
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
                current_L2 = self.L2_pinn(x, y)
                if current_L2 < self.best_l2_error:
                    self.best_l2_error = current_L2
                    self.best_abs_error = self.error_save_pinn()
    
    def train_lbfgs_pinn(self):
        max_iter = 100000 if args.dataset == "helmholtz" else 20000
        optimizer = torch.optim.LBFGS(self.net_params_pinn, lr=0.1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_pinn()
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        current_loss = self.get_loss_pinn().item()
        current_L2 = self.L2_pinn(x, y)
        print("LBFGS", current_loss, current_L2)
        if current_L2 < self.best_l2_error:
            self.best_l2_error = current_L2
            self.best_abs_error = self.error_save_pinn()
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
                current_L2 = self.L2_xpinn(x1, y1, x2, y2)
                if current_L2 < self.best_l2_error:
                    self.best_l2_error = current_L2
                    self.best_abs_error = self.error_save_xpinn()

    def train_lbfgs_xpinn(self):
        max_iter = 20000# if args.dataset == "helmholtz" else 20000
        optimizer = torch.optim.LBFGS(self.net_params_xpinn, lr=0.1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_xpinn()
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        current_loss = self.get_loss_xpinn().item()
        current_L2 = self.L2_xpinn(x1, y1, x2, y2)
        print("LBFGS", current_loss, current_L2)
        if current_L2 < self.best_l2_error:
            self.best_l2_error = current_L2
            self.best_abs_error = self.error_save_xpinn()
        return
    
    def predict_pinn(self, x_star, y_star):
        x_star = torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1).to(device)
        y_star = torch.unsqueeze(torch.tensor(y_star, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u_pred = self.u_net(torch.cat((x_star, y_star), 1))
        return u_pred
    
    def predict_xpinn(self, x_star1, y_star1, x_star2, y_star2):
        x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1).to(device)
        y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1).to(device)
        x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1).to(device)
        y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u1_pred = self.u1_net(torch.cat((x_star1, y_star1), 1))
            u2_pred = self.u2_net(torch.cat((x_star2, y_star2), 1))
        return u1_pred, u2_pred

    def L2_pinn(self, x_star, y_star):
        pinn_u_pred_20 = (self.predict_pinn(x_star, y_star)).detach().cpu().numpy()
        #pinn_error_20 = abs(u-pinn_u_pred_20.flatten())
        pinn_error_u_total_20 = np.linalg.norm(u-pinn_u_pred_20.flatten(),2)/np.linalg.norm(u,2)
        return pinn_error_u_total_20

    def error_plot_pinn(self):
        pinn_u_pred_20 = (self.predict_pinn(x, y)).detach().cpu().numpy()
        pinn_error_20 = abs(u-pinn_u_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        idx = np.random.choice(x.shape[0], 10000, replace=False)
        triang_coarse = tri.Triangulation(x[idx], y[idx])
        plt.tricontourf(triang_coarse, pinn_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(args.dataset+' PINN Error')
        plt.colorbar()
        plt.savefig("fig/PINN_"+args.dataset+".pdf")

    def error_save_pinn(self):
        pinn_u_pred_20 = (self.predict_pinn(x, y)).detach().cpu().numpy()
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

    def L2_xpinn(self, x_star1, y_star1, x_star2, y_star2):
        xpinn_u_pred1_20, xpinn_u_pred2_20 = self.predict_xpinn(x_star1, y_star1, x_star2, y_star2)
        xpinn_u_pred1_20 = (xpinn_u_pred1_20).detach().cpu().numpy()
        xpinn_u_pred2_20 = (xpinn_u_pred2_20).detach().cpu().numpy()
        xpinn_u_pred_20 = np.concatenate([xpinn_u_pred1_20, xpinn_u_pred2_20])
        xpinn_u_vals = np.concatenate([u1, u2])
        #xpinn_error_20 = abs(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten())
        xpinn_error_u_total_20 = np.linalg.norm(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten(),2)/np.linalg.norm(xpinn_u_vals.flatten(),2)
        return xpinn_error_u_total_20

    def error_plot_xpinn(self):
        xpinn_u_pred1_20, xpinn_u_pred2_20 = self.predict_xpinn(x1, y1, x2, y2)
        xpinn_u_pred1_20 = (xpinn_u_pred1_20).detach().cpu().numpy()
        xpinn_u_pred2_20 = (xpinn_u_pred2_20).detach().cpu().numpy()
        xpinn_u_pred_20 = np.concatenate([xpinn_u_pred1_20, xpinn_u_pred2_20])
        xpinn_u_vals = np.concatenate([u1, u2])
        xpinn_error_20 = abs(xpinn_u_vals.flatten()-xpinn_u_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        idx = np.random.choice(x1.shape[0]+x2.shape[0], 10000, replace=False)
        triang_coarse = tri.Triangulation(np.concatenate([x1, x2])[idx], np.concatenate([y1, y2])[idx])
        plt.tricontourf(triang_coarse, xpinn_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(args.dataset+' XPINN Error')
        plt.colorbar()
        plt.savefig("fig/XPINN_"+args.dataset+".pdf")

    def error_save_xpinn(self):
        xpinn_u_pred1_20, xpinn_u_pred2_20 = self.predict_xpinn(x1, y1, x2, y2)
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

        # boundary points --- mse
        # PINN
        #self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yb = torch.unsqueeze(torch.tensor(yb, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
    
        # collocation points --- residual
        # PINN
        self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.yf = torch.unsqueeze(torch.tensor(yf, dtype=torch.float32, requires_grad=True),-1).to(device)
        self.ff = torch.unsqueeze(torch.tensor(ff, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        if args.dataset in INIT_DATASETS:
            # PINN
            self.xn = torch.unsqueeze(torch.tensor(xn, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.yn = torch.unsqueeze(torch.tensor(yn, dtype=torch.float32, requires_grad=True),-1).to(device)
            self.fn = torch.unsqueeze(torch.tensor(fn, dtype=torch.float32, requires_grad=True),-1).to(device)
        # Initalize Neural Networks
        #E_layers = [args.input_dim] + [args.SXPINN_E_h] * args.SXPINN_E_L
        E_layers = [args.SXPINN_E_h] * args.SXPINN_E_L + [1]
        G_layers = [args.input_dim] + [args.SXPINN_G_h] * (args.SXPINN_G_L - 1) + [1]
        #h_layers = [args.SXPINN_E_h] + [args.SXPINN_h_h] * (args.SXPINN_h_L - 1) + [1]
        h_layers = [args.input_dim] + [args.SXPINN_h_h] * (args.SXPINN_h_L - 1) + [args.SXPINN_E_h]

        self.u_net = SXPINN_Net(E_layers, G_layers, h_layers).to(device)
        
        if args.SXPINN_gate_trainable:
            self.net_params_sxpinn = list(self.u_net.parameters())
        else:
            self.net_params_sxpinn = list(self.u_net.u1_net.parameters()) + list(self.u_net.u2_net.parameters()) + list(self.u_net.h.parameters())
        self.loss_sxpinn = []
        self.loss = 10

        self.best_l2_error = 1.0
        self.best_abs_error = 0

        if args.save_loss:
            self.saved_loss = []
            self.saved_l2 = []
        
    def Burgers(self, u, u_x, u_y, u_xx): 
        return u_y + u * u_x - (0.01 / np.pi) * u_xx
    
    def Poisson(self, u_xx, u_yy, f): 
        return u_yy + u_xx - f # Poisson problem residual

    def Helmholtz(self, u, u_xx, u_yy, f): 
        return u_yy + u_xx + u - f

    def KG(self, u, u_xx, u_yy, f):
        return u_yy - u_xx + u**3 - f
    
    def Wave(self, u_xx, u_yy):
        return u_yy - 4 * u_xx

    def Advection(self, u_x, u_y):
        return u_y + args.adv_beta * u_x

    def num_params(self):
        num_sxpinn = 0
        for p in self.net_params_sxpinn:
            num_sxpinn += len(p.reshape(-1))
        return num_sxpinn

    def get_loss_sxpinn(self):
        # Boundry loss
        mse_ub = (self.ub - self.u_net(torch.cat((self.xb, self.yb), 1))).square().mean()
        if args.dataset in INIT_DATASETS:
            un = self.u_net(torch.cat((self.xn, self.yn), 1))
            un_y = torch.autograd.grad(un.sum(), self.yn, create_graph=True)[0]
            mse_ub += (self.fn - un_y).square().mean()

        # Residual/collocation loss
        u = self.u_net(torch.cat((self.xf, self.yf), 1))
        u_sum = u.sum()
        if args.dataset == 'burgers':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            # Residuals
            f = self.Burgers(u, u_x, u_y, u_xx)
        elif args.dataset == 'poisson':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Poisson(u_xx, u_yy, self.ff)
        elif args.dataset == 'helmholtz':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Helmholtz(u, u_xx, u_yy, self.ff)
        elif args.dataset == 'KG':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.KG(u, u_xx, u_yy, self.ff)
        elif args.dataset == 'Wave':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
            # Residuals
            f = self.Wave(u_xx, u_yy)
        elif args.dataset == 'Advection':
            u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
            u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
            # Residuals
            f = self.Advection(u_x, u_y)
        mse_f = f.square().mean()
    
        # Sum losses
        net_loss = self.pinn_w_b*mse_ub + self.pinn_w_rc*mse_f
        
        loss = net_loss 
        return loss
    
    def train_lbfgs_sxpinn(self):
        max_iter = 100000 if args.dataset == "helmholtz" else 20000
        lbfgs_lr = 5e-1 if args.dataset == "helmholtz" else 1e-1
        print(max_iter)
        optimizer = torch.optim.LBFGS(self.net_params_sxpinn, lr=lbfgs_lr, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_sxpinn()
            loss.backward(retain_graph=True)
            if args.save_loss:
                self.saved_loss.append(loss.item())
                self.saved_l2.append(self.L2_sxpinn(x, y))
            return loss
        optimizer.step(closure)
        current_loss = self.get_loss_sxpinn().item()
        current_L2 = self.L2_sxpinn(x, y)
        print("LBFGS", current_loss, current_L2)
        if current_L2 < self.best_l2_error:
            self.best_l2_error = current_L2
            self.best_abs_error = self.error_save_sxpinn()
        return
    
    def train_adam_sxpinn(self):
        # stage 1: pretrain gate_net
        self.u_net.w_gate = self.pretrain_gate_net(self.u_net.w_gate)
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
                self.saved_l2.append(self.L2_sxpinn(x, y))
            if n > 0.9 * self.epoch:
                self.loss = current_loss
                current_L2 = self.L2_sxpinn(x, y)
                if current_L2 < self.best_l2_error:
                    self.best_l2_error = current_L2
                    self.best_abs_error = self.error_save_sxpinn()
                if args.dataset == "Helmholtz" and current_loss < 4e-2:
                    return
 
    def pretrain_gate_net(self, w_gate):
        if args.dataset == 'burgers':
            x, y = np.linspace(-1, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(x-1) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'poisson':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            #u = np.ones_like(x) * 0.9
            u = np.exp(- (x-0.5)**2 - (y-0.5)**2) if args.SXPINN_Prior == 'XPINN' else 0.9 * np.ones_like(x)
        elif args.dataset == 'helmholtz':
            x, y = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(y-1) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'KG':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-x) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'Wave':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-y) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'Advection':
            x, y = np.linspace(-1, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-y) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        #print(u)
        x, y, u = x.reshape(-1, 1), y.reshape(-1, 1), u.reshape(-1, 1)
        x, y, u = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device), torch.from_numpy(u).float().to(device)
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
        if args.dataset == 'burgers':
            x, y = np.linspace(-1, 1, 20), np.linspace(0, 1, 20)
            x, y = np.meshgrid(x, y)
            u = np.exp(x-1) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'poisson':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(- (x-0.5)**2 - (y-0.5)**2)
        elif args.dataset == 'helmholtz':
            x, y = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(y-1) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'KG':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-x) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'Wave':
            x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-y) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        elif args.dataset == 'Advection':
            x, y = np.linspace(-1, 1, 100), np.linspace(0, 1, 100)
            x, y = np.meshgrid(x, y)
            u = np.exp(-y) if args.SXPINN_Prior == 'XPINN' else 0.8 * np.ones_like(x)
        x, y, u = x.reshape(-1, 1), y.reshape(-1, 1), u.reshape(-1, 1)
        x, y, u = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device), torch.from_numpy(u).float().to(device)
        u = self.u_net.w_gate(torch.cat((x, y), 1))
        x, y, u = x.reshape(-1), y.reshape(-1), u.reshape(-1)
        x, y, u = x.detach().cpu().numpy(), y.detach().cpu().numpy(), u.detach().cpu().numpy()
        #print(u)

        triang_coarse = tri.Triangulation(x, y)
        plt.tricontourf(triang_coarse, u, 100 ,cmap='jet')
        plt.xlabel('x')
        if args.dataset in ['burgers', 'KG', 'Wave']:
            plt.ylabel('t')
        else:
            plt.ylabel('y')
        if args.dataset == "burgers":
            plt.title('Burgers Gate Network')
        elif args.dataset == "helmholtz":
            plt.title('Helmholtz Gate Network')
        else:
            plt.title(args.dataset+' Gate Network')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("fig/SXPINN_Gate_"+args.SXPINN_Prior+"_"+args.dataset+"_"+str(args.SEED)+".pdf")
        plt.savefig("fig/SXPINN_Gate_"+args.SXPINN_Prior+"_"+args.dataset+"_"+str(args.SEED)+".png")
        return

    def predict_sxpinn(self, x_star, y_star):
        x_star = torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1).to(device)
        y_star = torch.unsqueeze(torch.tensor(y_star, dtype=torch.float32),-1).to(device)
        with torch.no_grad():
            u_pred = self.u_net(torch.cat((x_star, y_star), 1))
        return u_pred
    
    def L2_sxpinn(self, x_star, y_star):
        sxpinn_u_pred_20 = (self.predict_sxpinn(x_star, y_star)).detach().cpu().numpy()
        #pinn_error_20 = abs(u-pinn_u_pred_20.flatten())
        sxpinn_error_u_total_20 = np.linalg.norm(u-sxpinn_u_pred_20.flatten(),2)/np.linalg.norm(u,2)
        return sxpinn_error_u_total_20

    def error_plot_sxpinn(self):
        sxpinn_u_pred_20 = (self.predict_sxpinn(x, y)).detach().cpu().numpy()
        sxpinn_error_20 = abs(u-sxpinn_u_pred_20.flatten())
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        idx = np.random.choice(x.shape[0], 10000, replace=False)
        triang_coarse = tri.Triangulation(x[idx], y[idx])
        plt.tricontourf(triang_coarse, sxpinn_error_20[idx], 100 ,cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(args.dataset+' SXPINN Error')
        plt.colorbar()
        plt.savefig("fig/SXPINN_"+args.dataset+".pdf")

    def error_save_sxpinn(self):
        sxpinn_u_pred_20 = (self.predict_sxpinn(x, y)).detach().cpu().numpy()
        sxpinn_error_20 = abs(u-sxpinn_u_pred_20.flatten())
        return sxpinn_error_20
        #np.savetxt("error/SXPINN_"+args.dataset+"_error_"+str(args.SEED), sxpinn_error_20)
    
    def save_subnet_sxpinn(self, x, y):
        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32),-1).to(device)
        y = torch.unsqueeze(torch.tensor(y, dtype=torch.float32),-1).to(device)
        u1 = self.u_net.u1(torch.cat((x, y), 1))
        u2 = self.u_net.u2(torch.cat((x, y), 1))
        u1, u2 = u1.detach().cpu().numpy().reshape(-1), u2.detach().cpu().numpy().reshape(-1)
        x, y = x.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy().reshape(-1)
        idx = np.random.choice(x.shape[0], 10000, replace=False)

        minimum, maximum = min(u1.min(), u2.min()), max(u1.max(), u2.max())

        plt.rcParams["figure.figsize"] = (12.0,5.0)
        plt.subplot(121)
        triang_coarse = tri.Triangulation(x[idx], y[idx])
        plt.tricontourf(triang_coarse, u1[idx], 100 ,cmap='jet')
        plt.clim(minimum, maximum)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(args.dataset+' SXPINN Sub-Net 1')
        plt.subplot(122)
        triang_coarse = tri.Triangulation(x[idx], y[idx])
        plt.tricontourf(triang_coarse, u2[idx], 100 ,cmap='jet')
        plt.clim(minimum, maximum)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(args.dataset+' SXPINN Sub-Net 2')

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.015, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig("APINN_Net/SXPINN_Subnet_"+args.dataset+".pdf")
        return
    
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
    np.savetxt("saved_loss_l2/PINN"+"_"+args.dataset+"_error_"+str(args.SEED), model.best_abs_error)
    if args.plot_error:
        model.error_plot_pinn()
    if args.plot_loss:
        model.plot_loss_pinn()
    if args.save_error:
        model.error_save_pinn()

if args.XPINN:
    model.train_adam_xpinn()
    print('XPINN Seed: ', args.SEED, 'Train Loss: ', model.loss, 'Relative L2: ', model.best_l2_error)
    model.train_lbfgs_xpinn()
    np.savetxt("saved_loss_l2/XPINN"+str(args.XPINN_type)+"_"+args.dataset+"_error_"+str(args.SEED), model.best_abs_error)
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
    np.savetxt("saved_loss_l2/APINN"+args.SXPINN_Prior[0]+"_"+args.dataset+"_error_"+str(args.SEED), model.best_abs_error)
    if args.plot_error:
        model.error_plot_sxpinn()
    if args.plot_loss:
        model.plot_loss_sxpinn()
    if args.save_error:
        model.error_save_sxpinn()
    if args.save_gate:
        model.save_gate()
    if args.save_subnet:
        model.save_subnet_sxpinn(x, y)
    if args.save_loss:
        model.saved_loss = np.array(model.saved_loss)
        model.saved_l2 = np.array(model.saved_l2)
        np.savetxt("saved_loss_l2/"+args.dataset+"_loss_"+str(args.SEED)+".txt", model.saved_loss)
        np.savetxt("saved_loss_l2/"+args.dataset+"_l2_"+str(args.SEED)+".txt", model.saved_l2)