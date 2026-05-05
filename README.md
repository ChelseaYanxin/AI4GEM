# AI4GEM
run command：
```
python3 scripts/run_mat_fdt.py --backend gnn --rel-only --compare-gt --metrics-out metrics_vs_time.csv --max-steps 12
```
```
python3 scripts/run_mat_fdt.py --backend gnn --max-steps 20 --compare-gt --use-gt-dt --rel-only
```

初始设置主要分三类：数值网格/物理参数、初始场与源、仿真控制与度量。下面按代码位置整理（不改代码）。

数值与物理参数
默认 dtype: 在文件开头 torch.set_default_dtype(torch.float64) → 后续全部张量转为 float64。
网格尺寸与坐标:
从 .mat 中的 x, y 构造；若不存在则用场 Ez0 的 shape，假设方形区域 50 mm × 50 mm (Lx=Lz=50e-3)，均匀划分。
计算 dx, dz 为平均相邻差值或基于 L/(n−1)。
介质参数:
eps_map 默认全域 ε0=8.854187817e-12，若 .mat 提供 eps 则用。
mu_map 默认 μ0=4π·1e−7，若 .mat 提供 mu 则用。
sigma_map 默认全零，若 .mat 提供 sigma 则用。
Uniform damping：--sigma-uniform（默认 1e-7）加到 sigma_map。
时间步长 dt:
默认 dt = cfl_scale / (c0 * sqrt(1/dx^2 + 1/dz^2))，cfl_scale 默认 0.07。
若加 --use-gt-dt 并有 t，用 mean(diff(t)) 覆盖。
边界条件:
仅对 Ez 应用 PEC：四边置零。若提供 idx_bc 且长度 <= nx*nz，则按索引单点置零；否则回退用四边。
不对 Hx, Hy 做夹边。
初始场与源
初始场:
Ez_t: 取 .mat 中 Ez[:,:,0] 或单帧 Ez；若没有 Hx/Hy 3D，则初始 Hx/Hy 全零张量。
源注入:
--source auto：若初始 |Ez| 峰值 < 1e-9，则使用高斯脉冲，否则不加源。
高斯脉冲：src_steps 控制脉冲宽度，中心在 0.5*src_steps，幅度 src_amp，时间包络 exp(-0.5*((n-t0)/spread)^2)，spread = max(1, 0.2*src_steps)。
连续波 cw：正弦 sin(2π f n dt)，频率 src_freq_ghz GHz，前 src_steps 线性升幅。
源位置：网格中心 (cx=nx//2, cy=nz//2)，直接加到 Ez_t[0,0,cx,cy]。
--source none：不注入。
介质电导对初始并无特别处理，只在更新中通过 sigma_map 生效（dense 后端；gnn 后端视实现而定）。
仿真控制
后端选择: --backend dense (默认) 使用 GEMTE; --backend gnn 使用 GEMTEGraph2D。
步数确定:
若 Ez 是 3D [nx, ny, nt] → steps=nt；否则用 len(t)；再受 --max-steps 截断。
GIF 捕捉:
--gif / --gif-hx / --gif-hy 控制是否记录帧；--gif-every 间隔；--gif-fps 帧率；路径参数决定输出文件名。
可视化与色标:
--norm（sym / abs / zero-one / percentile）及 --percentiles；--share-scale 共用色标；--triptych 生成三联图。
--cmap 手动 colormap，否则 Ez/H 用 'turbo' (sym/percentile) 或 'viridis' (其它)。
度量与 CSV:
--compare-gt 启用逐步误差计算，使用 Ez_gt[:, :, gt_idx] 对齐（gt_idx = n + gt_offset 截断在范围内）。
--gt-offset 时间帧对齐偏移。