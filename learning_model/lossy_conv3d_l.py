import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# ==========================================
# 1. 优先加载数据 & 确定维度
# ==========================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on: {device}")

mat_file = '/Users/zyanxin/Documents/code/GEM_N/data/FDTD_3D_cavity_clean_data.mat'

# 默认设置（如果没找到文件）
default_nmax = 500 

target_signal = None
nmax = default_nmax

try:
    if os.path.exists(mat_file):
        print(f"Found data file: {mat_file}")
        mat = scipy.io.loadmat(mat_file)
        
        # --- 检查数据结构 ---
        # 假设 Ez 是 (nx, ny, nz, time) 或者 (time, ...)
        # 请根据你 mat 文件的实际结构调整这里的索引
        if 'Ez' in mat:
            raw_data = mat['Ez']
            print(f"Raw data shape: {raw_data.shape}")
            
            # 自动判断时间轴在哪里
            # 通常 FDTD 数据最后一维是时间
            data_time_steps = raw_data.shape[-1] 
            
            # 更新 nmax 以匹配数据长度！
            if data_time_steps < default_nmax:
                print(f"注意：数据长度 ({data_time_steps}) 小于预设 nmax ({default_nmax})。")
                print(f"自动将仿真步数 nmax调整为 -> {data_time_steps}")
                nmax = data_time_steps
            else:
                nmax = default_nmax
            
            # 提取中心点数据 (假设前三维是空间)
            # 确保索引不要越界，取空间中心
            cx, cy, cz = raw_data.shape[0]//2, raw_data.shape[1]//2, raw_data.shape[2]//2
            gt_numpy = raw_data[cx, cy, cz, :nmax]
            
            target_signal = torch.from_numpy(gt_numpy).float().to(device)
            target_signal = target_signal.view(nmax, 1)
            
            # 归一化 Ground Truth
            gt_max = torch.max(torch.abs(target_signal))
            if gt_max > 0:
                target_signal /= gt_max
            print("数据加载并归一化完成。")
            
        else:
            raise ValueError("Key 'Ez' not found in .mat file")
    else:
        print("未找到文件，使用默认 nmax 和虚拟目标。")
        target_signal = torch.zeros(nmax, 1).to(device)

except Exception as e:
    print(f"Error Loading Data: {e}")
    print("使用了 Fallback 逻辑，这可能导致 Loss=NaN，请检查文件路径和格式。")
    nmax = 417 # 强制设为你报错里提到的那个数字，防止 crash
    target_signal = torch.zeros(nmax, 1).to(device)


# ==========================================
# 2. 配置其他参数
# ==========================================
dtype_torch = torch.float32 
nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
shape_max = (1, 3, nx + 1, ny + 1, nz + 1)
c0 = 2.99792458e8
dt = 0.99 / (c0 * np.sqrt(3.0/(dx*dx))) 

# ==========================================
# 3. 模型定义 (保持不变)
# ==========================================
class LearnableFDTD(nn.Module):
    def __init__(self, nx, ny, nz, dx, dy, dz, dt, nmax):
        super(LearnableFDTD, self).__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt = dt
        self.nmax = nmax
        
        # Learnable Sigma: 初始化更稳健，防止一开始就NaN
        self.raw_sigma = nn.Parameter(torch.zeros((1, 1, nx+1, ny+1, nz+1)) - 5.0) 
        self.bias = nn.Parameter(torch.zeros(shape_max) * 1e-5)
        self.E_init = nn.Parameter(torch.zeros(shape_max))
        
        self.eps_r = 1.0
        self.C_H = dt / (m0 * dx)
        
        self.register_buffer('K_E', self._get_curl_kernel('backward'))
        self.register_buffer('K_H', self._get_curl_kernel('forward'))
        self.register_buffer('mask', self._get_pec_mask())
        
        # Source
        t = torch.arange(1, nmax + 2, dtype=dtype_torch) * dt
        rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
        self.register_buffer('source_waveform', 
             srcconst * ((t/dt) - ndelay) * torch.exp(-(((t/dt) - ndelay)**2 / (tau**2))))
        self.Is, self.Js, self.Ks = 9, 9, 9

    def _get_pec_mask(self):
        mask = torch.ones(shape_max, dtype=dtype_torch)
        mask[:, 0, :, 0, :] = 0; mask[:, 0, :, -1, :] = 0
        mask[:, 0, :, :, 0] = 0; mask[:, 0, :, :, -1] = 0; mask[:, 0, -1, :, :] = 0 
        mask[:, 1, 0, :, :] = 0; mask[:, 1, -1, :, :] = 0
        mask[:, 1, :, :, 0] = 0; mask[:, 1, :, :, -1] = 0; mask[:, 1, :, -1, :] = 0 
        mask[:, 2, 0, :, :] = 0; mask[:, 2, -1, :, :] = 0
        mask[:, 2, :, 0, :] = 0; mask[:, 2, :, -1, :] = 0; mask[:, 2, :, :, -1] = 0 
        return mask

    def _get_curl_kernel(self, mode):
        k = torch.zeros((3, 3, 3, 3, 3), dtype=dtype_torch)
        def set_k(out_c, in_c, axis, sign):
            center = [1, 1, 1]
            if mode == 'backward': neighbor = [1,1,1]; neighbor[axis]-=1; w=[1.0, -1.0]
            else: neighbor = [1,1,1]; neighbor[axis]+=1; w=[-1.0, 1.0]
            k[out_c, in_c, 1, 1, 1] = w[0] * sign
            k[out_c, in_c, neighbor[0], neighbor[1], neighbor[2]] = w[1] * sign
        set_k(0, 2, 1, +1); set_k(0, 1, 2, -1)
        set_k(1, 0, 2, +1); set_k(1, 2, 0, -1)
        set_k(2, 1, 0, +1); set_k(2, 0, 1, -1)
        return k

    def forward(self):
        sigma = F.softplus(self.raw_sigma) 
        EA = (e0 * self.eps_r / self.dt) + 0.5 * sigma
        EB = (e0 * self.eps_r / self.dt) - 0.5 * sigma
        CA = EB / EA
        CB = 1.0 / (EA * self.dx)
        
        E = self.E_init.clone() + self.bias
        H = torch.zeros_like(E)
        
        sensor_history = [] 

        for n in range(self.nmax):
            H_pad = F.pad(H, (1,1,1,1,1,1))
            curl_E = F.conv3d(H_pad, self.K_E)
            H = H - self.C_H * curl_E
            
            E_pad = F.pad(E, (1,1,1,1,1,1))
            curl_H = F.conv3d(E_pad, self.K_H)
            E = E * CA + curl_H * CB
            E = E * self.mask + self.bias * 0.01

            # Soft Source Injection
            src_val = self.source_waveform[n]
            # 手动构建 sparse update 以节省内存并保持计算图
            # 仅在源位置加值
            E[:, 2, self.Is, self.Js, self.Ks] = E[:, 2, self.Is, self.Js, self.Ks] + src_val
            
            sensor_history.append(E[:, 2, self.Is, self.Js, self.Ks])
        
        return torch.stack(sensor_history), E

# ==========================================
# 4. 训练循环
# ==========================================
model = LearnableFDTD(nx, ny, nz, dx, dy, dz, dt, nmax).to(device)
optimizer = torch.optim.Adam([
    {'params': model.raw_sigma, 'lr': 0.05},
    {'params': model.bias, 'lr': 0.001}
])

print(f"\nStart Training (Steps={nmax})...")
losses = []

for epoch in range(50):
    optimizer.zero_grad()
    
    # Forward
    pred_waveform, _ = model()
    
    # 归一化预测结果 (Add epsilon to avoid div by zero)
    pred_max = torch.max(torch.abs(pred_waveform))
    if pred_max > 1e-9:
        pred_norm = pred_waveform / pred_max
    else:
        pred_norm = pred_waveform # 如果全是0就不归一化了，防止NaN

    # Loss
    if target_signal is not None:
        loss = F.mse_loss(pred_norm, target_signal)
    else:
        loss = torch.tensor(0.0, requires_grad=True) # Dummy
    
    if torch.isnan(loss):
        print(f"Epoch {epoch}: Loss is NaN! Stopping.")
        break

    loss.backward()
    
    # 梯度裁剪 (防止梯度爆炸)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 5 == 0:
        sigma_mean = F.softplus(model.raw_sigma).mean().item()
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | Mean Sigma: {sigma_mean:.4f}")

# Plot
plt.figure()
plt.plot(target_signal.cpu().detach().numpy(), label='GT')
plt.plot(pred_norm.cpu().detach().numpy(), '--', label='Pred')
plt.legend()
plt.show()