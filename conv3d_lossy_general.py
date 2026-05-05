import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from scipy.constants import mu_0 as m0, epsilon_0 as e0, speed_of_light as c0

# 论文级绘图设置
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "lines.linewidth": 2
})

# ==========================================
# 0. 全局配置
# ==========================================
# 【关键修改】强制使用双精度 float64
DTYPE_NP = np.float64
DTYPE_TORCH = torch.float64

nmax = 500
nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3

Is, Js, Ks = 9, 9, 9       
obs_points = [(11, 11, 9), (15, 15, 15)] # 多个观测点

dt = 0.99 / (c0 * np.sqrt(3.0 / (dx**2)))
print(f"Global Config: dt {dt*1e12:.3f}ps, Precision: Float64")

# ==========================================
# 1. Ground Truth (CPU - Float64)
# ==========================================
def run_cpu_ground_truth(target_sigma):
    print(f"\n[Ground Truth] Running CPU FDTD (Sigma={target_sigma})...")
    
    Ex = np.zeros((nx, ny + 1, nz + 1), dtype=DTYPE_NP)
    Ey = np.zeros((nx + 1, ny, nz + 1), dtype=DTYPE_NP)
    Ez = np.zeros((nx + 1, ny + 1, nz), dtype=DTYPE_NP)
    Hx = np.zeros((nx + 1, ny, nz), dtype=DTYPE_NP)
    Hy = np.zeros((nx, ny + 1, nz), dtype=DTYPE_NP)
    Hz = np.zeros((nx, ny, nz + 1), dtype=DTYPE_NP)
    
    eps_r = 1.0
    # Coefficients
    EA = (e0 * eps_r / dt) + 0.5 * target_sigma
    EB = (e0 * eps_r / dt) - 0.5 * target_sigma
    CA = EB / EA
    CB = 1.0 / (EA * dx) # Simplified assuming dx=dy=dz
    
    HA = (m0 / dt)
    DA = 1.0
    DB = 1.0 / (HA * dx)
    
    # Source
    t_axis = np.arange(1, nmax + 1, dtype=DTYPE_NP) * dt
    rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * 3.0e11
    source = srcconst * ((t_axis/dt) - ndelay) * np.exp(-(((t_axis/dt) - ndelay)**2 / (tau**2)))
        
    recorded_data = [np.zeros(nmax, dtype=DTYPE_NP) for _ in obs_points]
    
    for n in range(nmax):
        # Update E
        # Standard Yee Update (simplified for readability, exact physics)
        Ex[:, 1:ny, 1:nz] = CA * Ex[:, 1:ny, 1:nz] + CB * ((Hz[:, 1:ny, 1:nz] - Hz[:, 0:ny-1, 1:nz]) - (Hy[:, 1:ny, 1:nz] - Hy[:, 1:ny, 0:nz-1]))
        Ey[1:nx, :, 1:nz] = CA * Ey[1:nx, :, 1:nz] + CB * ((Hx[1:nx, :, 1:nz] - Hx[1:nx, :, 0:nz-1]) - (Hz[1:nx, :, 1:nz] - Hz[0:nx-1, :, 1:nz]))
        Ez[1:nx, 1:ny, :] = CA * Ez[1:nx, 1:ny, :] + CB * ((Hy[1:nx, 1:ny, :] - Hy[0:nx-1, 1:ny, :]) - (Hx[1:nx, 1:ny, :] - Hx[1:nx, 0:ny-1, :]))
                            
        Ez[Is, Js, Ks] += source[n]
        
        # Update H
        Hx[:, :, :] = DA * Hx[:, :, :] - DB * ((Ez[:, 1:ny+1, :] - Ez[:, 0:ny, :]) - (Ey[:, :, 1:nz+1] - Ey[:, :, 0:nz]))
        Hy[:, :, :] = DA * Hy[:, :, :] - DB * ((Ex[:, :, 1:nz+1] - Ex[:, :, 0:nz]) - (Ez[1:nx+1, :, :] - Ez[0:nx, :, :]))
        Hz[:, :, :] = DA * Hz[:, :, :] - DB * ((Ey[1:nx+1, :, :] - Ey[0:nx, :, :]) - (Ex[:, 1:ny+1, :] - Ex[:, 0:ny, :]))
        
        for i, (io, jo, ko) in enumerate(obs_points):
            recorded_data[i][n] = Ez[io, jo, ko]
            
    return np.stack(recorded_data, axis=0)

# ==========================================
# 2. NeCLO Solver (Float64)
# ==========================================
class DifferentiableFDTD(nn.Module):
    def __init__(self, init_sigma=0.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 【关键】dtype=DTYPE_TORCH (Float64)
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=DTYPE_TORCH, device=self.device))
        
        self.register_buffer('eps_r', torch.tensor(1.0, dtype=DTYPE_TORCH))
        self.register_buffer('dt', torch.tensor(dt, dtype=DTYPE_TORCH))
        self.register_buffer('dx', torch.tensor(dx, dtype=DTYPE_TORCH))
        self.register_buffer('K_E', self._get_curl_kernel('backward'))
        self.register_buffer('K_H', self._get_curl_kernel('forward'))
        self.register_buffer('mask', self._get_mask())

    def _get_curl_kernel(self, mode):
        # Kernel 也要是 Float64
        k = torch.zeros((3, 3, 3, 3, 3), dtype=DTYPE_TORCH, device=self.device)
        def set_k(out_c, in_c, axis, sign):
            center, neighbor = [1, 1, 1], [1, 1, 1]
            if mode == 'backward': w_c, w_n = 1.0, -1.0; neighbor[axis] -= 1
            else: w_c, w_n = -1.0, 1.0; neighbor[axis] += 1
            k[out_c, in_c, center[0], center[1], center[2]] = w_c * sign
            k[out_c, in_c, neighbor[0], neighbor[1], neighbor[2]] = w_n * sign
        set_k(0, 2, 1, +1); set_k(0, 1, 2, -1)
        set_k(1, 0, 2, +1); set_k(1, 2, 0, -1)
        set_k(2, 1, 0, +1); set_k(2, 0, 1, -1)
        return k

    def _get_mask(self):
        m = torch.ones((1, 3, nx+1, ny+1, nz+1), dtype=DTYPE_TORCH, device=self.device)
        # Simple PEC boundary
        m[:, :, 0, :, :] = 0; m[:, :, -1, :, :] = 0
        m[:, :, :, 0, :] = 0; m[:, :, :, -1, :] = 0
        m[:, :, :, :, 0] = 0; m[:, :, :, :, -1] = 0
        return m

    def forward(self, n_steps, source_waveform):
        # 场初始化 Float64
        E = torch.zeros((1, 3, nx+1, ny+1, nz+1), dtype=DTYPE_TORCH, device=self.device)
        H = torch.zeros((1, 3, nx+1, ny+1, nz+1), dtype=DTYPE_TORCH, device=self.device)
        
        # Coefficients (Strictly matching CPU)
        EA = (e0 * self.eps_r / self.dt) + 0.5 * self.sigma
        EB = (e0 * self.eps_r / self.dt) - 0.5 * self.sigma
        CA = EB / EA
        CB = 1.0 / (EA * self.dx)
        CH = self.dt / (m0 * self.dx)
        
        outputs = []
        for n in range(n_steps):
            curl_E = F.conv3d(E, self.K_H, padding=1)
            H = H - CH * curl_E
            
            curl_H = F.conv3d(H, self.K_E, padding=1)
            E = CA * E + CB * curl_H
            E = E * self.mask
            
            E[:, 2, Is, Js, Ks] += source_waveform[n]
            
            current_vals = []
            for (io, jo, ko) in obs_points:
                current_vals.append(E[:, 2, io, jo, ko])
            outputs.append(torch.stack(current_vals))
        return torch.stack(outputs).squeeze(-1).t()
    

# ==========================================
# 3. Training
# ==========================================
if __name__ == "__main__":
    target_sigma = 4.0
    gt_np = run_cpu_ground_truth(target_sigma) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observed_data = torch.tensor(gt_np, dtype=DTYPE_TORCH, device=device)
    
    # Source
    t_axis = torch.arange(1, nmax + 1, dtype=DTYPE_TORCH, device=device) * dt
    rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * 3.0e11
    source_waveform = srcconst * ((t_axis/dt) - ndelay) * torch.exp(-(((t_axis/dt) - ndelay)**2 / (tau**2)))

    print("\nStarting Training (Float64 Precision)...")
    
    init_sigma = 1.0 
    model = DifferentiableFDTD(init_sigma)
    
    # 稍微调小学习率，因为 float64 梯度更精准但可能更敏感
    optimizer = torch.optim.Adam([model.sigma], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.8, verbose=True)
    
    loss_history = []
    sigma_history = []
    
    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(nmax, source_waveform)
        
        # 【技巧】把 Loss 放大 1e4 倍，防止数值下溢
        loss = F.mse_loss(pred, observed_data) * 1e4
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        with torch.no_grad():
            model.sigma.clamp_(min=0.0)
            
        loss_val = loss.item()
        sigma_val = model.sigma.item()
        loss_history.append(loss_val)
        sigma_history.append(sigma_val)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:<3} | Loss: {loss_val:.2e} | Sigma: {sigma_val:.4f}")

    print(f"Final Learned Sigma: {model.sigma.item():.4f} (Target: {target_sigma})")
    
    # --- Plotting ---
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sigma_history, 'r-o', label='Learned $\sigma$')
    ax1.axhline(target_sigma, color='k', linestyle='--', label=f'Ground Truth ({target_sigma})')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Sigma (S/m)'); ax1.legend()
    ax1.set_title('(a) Convergence')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(observed_data.cpu().numpy()[0], 'k-', lw=3, alpha=0.3, label='Ground Truth')
    with torch.no_grad():
        pred_np = model(nmax, source_waveform).cpu().numpy()[0]
    ax2.plot(pred_np, 'r--', label='NeCLO Prediction')
    ax2.set_xlabel('Time Step'); ax2.set_ylabel('Ez'); ax2.legend()
    ax2.set_title('(b) Waveform Match')
    
    plt.tight_layout()
    plt.show()