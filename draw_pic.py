import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. 配置风格 (修复部分)
# ==========================================
# 尝试使用 Times New Roman，如果系统没有则回退到 DejaVu Serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern (类 LaTeX 数学字体)
plt.rcParams['font.size'] = 12

# 统一色调 (Academic Colors)
COLOR_POS = '#D9534F'  # Red for +1
COLOR_NEG = '#428BCA'  # Blue for -1
COLOR_BG  = '#F9F9F9'  # Light Grey for 0
COLOR_GRID = '#DDDDDD' 
COLOR_TEXT = '#333333'

def draw_grid(ax, rows, cols, data, offset=(0,0), title="", title_y=1.05):
    """绘制基础网格函数"""
    dx, dy = offset
    
    # 绘制格子
    for i in range(rows):
        for j in range(cols):
            val = data[i][j]
            x = j + dx
            y = (rows - 1 - i) + dy # Invert y to match matrix indexing
            
            # 决定颜色
            if val == 1:
                facecolor = COLOR_POS
                textcolor = 'white'
                text = '+1'
            elif val == -1:
                facecolor = COLOR_NEG
                textcolor = 'white'
                text = '-1'
            else:
                facecolor = COLOR_BG
                textcolor = '#AAAAAA'
                text = '0'
            
            # 画方块
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='white', facecolor=facecolor)
            ax.add_patch(rect)
            
            # 写字
            ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', 
                    color=textcolor, fontweight='bold', fontsize=14)
            
    # 边框和标题
    ax.set_xlim(dx, dx + cols)
    ax.set_ylim(dy, dy + rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        # 修正：去掉了 \textbf，改用 \mathbf，并手动处理空格 (\ )
        ax.text(dx + cols/2, dy + rows + 0.2, title, ha='center', va='bottom', fontsize=13, color=COLOR_TEXT)

# ==========================================
# 2. 绘图主程序
# ==========================================
fig = plt.figure(figsize=(12, 5), dpi=300)

# --- 子图 A: TEMS v1 (1D Finite Difference) ---
ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])

# 构造数据
grid_v1 = np.zeros((5, 5))
grid_v1[1, 1] = -1 # Input i
grid_v1[1, 2] = 1  # Input i+1

# 修复：\textbf -> \mathbf, 增加 \ 转义空格
draw_grid(ax1, 5, 5, grid_v1, title=r"$\mathbf{TEMS\ v1:}$ Slice-based Difference")

# 添加标注
ax1.text(1.5, 3.5, r"$E_i$", color='white', ha='center', va='top', fontsize=9, alpha=0.8)
ax1.text(2.5, 3.5, r"$E_{i+1}$", color='white', ha='center', va='top', fontsize=9, alpha=0.8)

# 卷积核框
kernel_rect = patches.FancyBboxPatch((1.05, 3.05), 1.9, 0.9, boxstyle="round,pad=0.1", 
                                     linewidth=2, edgecolor=COLOR_TEXT, facecolor='none', linestyle='--')
ax1.add_patch(kernel_rect)
ax1.text(2, 4.3, r"Kernel $[-1, 1]$", ha='center', color=COLOR_TEXT, fontsize=10, fontweight='bold')

# 公式
ax1.text(2.5, -0.8, r"$\partial_x E \approx 1 \cdot E_{i+1} + (-1) \cdot E_i$", 
         ha='center', fontsize=13, bbox=dict(facecolor='#F0F0F0', edgecolor='none', pad=5))


# --- 子图 B: TEMS v2 (Unified Kernels) ---
ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])

# Kernel 1: d/dy (Hz)
k1 = np.zeros((3, 3))
k1[1, 1] = 1
k1[2, 1] = -1 

# Kernel 2: d/dz (Hy)
k2 = np.zeros((3, 3))
k2[1, 1] = 1
k2[1, 2] = -1 

# 绘制
draw_grid(ax2, 3, 3, k1, offset=(0, 1), title=r"Kernel for $\partial_y H_z$")
draw_grid(ax2, 3, 3, k2, offset=(4, 1), title=r"Kernel for $\partial_z H_y$")

# 添加轴标
for x_base in [0, 4]:
    ax2.text(x_base - 0.3, 1.5, r"$j+1$", va='center', ha='right', fontsize=10, color='gray')
    ax2.text(x_base - 0.3, 2.5, r"$j$",   va='center', ha='right', fontsize=10, color='gray')
    ax2.text(x_base - 0.3, 3.5, r"$j-1$", va='center', ha='right', fontsize=10, color='gray')
    
    ax2.text(x_base + 0.5, 0.7, r"$k-1$", ha='center', va='top', fontsize=10, color='gray')
    ax2.text(x_base + 1.5, 0.7, r"$k$",   ha='center', va='top', fontsize=10, color='gray')
    ax2.text(x_base + 2.5, 0.7, r"$k+1$", ha='center', va='top', fontsize=10, color='gray')

# 修复：\textbf -> \mathbf, 增加 \ 转义空格
ax2.text(3.5, 5.2, r"$\mathbf{TEMS\ v2:}$ Unified 3D Curl Kernels", ha='center', fontsize=14, color=COLOR_TEXT)

# 物理公式
ax2.text(3.5, -0.8, r"$[\nabla \times H]_x = \partial_y H_z - \partial_z H_y$", 
         ha='center', fontsize=13, bbox=dict(facecolor='#F0F0F0', edgecolor='none', pad=5))

# ==========================================
# 3. 添加图例标注 (a) (b)
# ==========================================
fig.text(0.02, 0.95, "(a)", fontsize=16, fontweight='bold', color=COLOR_TEXT)
fig.text(0.52, 0.95, "(b)", fontsize=16, fontweight='bold', color=COLOR_TEXT)

# 保存
plt.savefig("unified_tems_kernels.png", bbox_inches='tight', pad_inches=0.1)
print("Finished! Saved as 'unified_tems_kernels.png'")
plt.show()