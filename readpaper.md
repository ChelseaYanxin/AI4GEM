## 1. Background
神经网络其实可以直接学出 PDE 的解（根据 Universal Approximation Theorem —— 万能逼近定理）
**问题**
* 神经网络版的 PDE 求解器（叫“神经替代模型”）很难扩展；
* 只能处理固定大小的区域（domain）、固定的边界条件；
* 当问题变大或参数分布复杂时，模型精度会下降、训练也变得不稳定。

## 2. Solution

**传统迭代：**
1. **迭代预条件器 (iterative preconditioners)**  
   * 先求出残差 (r = b - A x)，然后用近似 (A ^ {-1}) 来快速修正误差；  
   * 比如共轭梯度法 (Conjugate Gradient) 或 GMRES。
2. **分区迭代法 (domain decomposition)**  
   * 把大域划分为多个小子域（subdomains），每个子域单独求解；  
   * 再通过迭代让全局解一致（常见方法如 Schwarz method, multigrid）。

提出一个**混合神经-迭代 PDE 求解框架**
采用**域分解（Domain Decomposition, DDM）结合子域神经网络预处理器（subdomain preconditioner）**来加速求解
以二维 **Maxwell 方程（电磁波方程）** 为例：

方法架构分两层：

**① 子域级别（Subdomain level）**

* 每个子域的 PDE 表示为 (A x = b)，其中 (A) 包含微分算子 + 边界条件。
* 训练一个 **Fourier Neural Operator (FNO)** 网络，输入：
  * 残差 (r)
  * 辅助信息（介电常数分布、源项、PML 边界等）  
    输出：
  * 对应的误差 (e)
* 也就是说，这个神经网络学的是一个**近似的逆算子 (A^{-1})**。
* 它被当作一个**通用预条件器（general preconditioner）**，嵌入到 F-GMRES 迭代算法中。

网络预测这一步误差要往哪修正，GMRES 再根据这些方向迭代逼近真实解。

**② 全局级别（Global level）**

* 使用 **domain decomposition**将全局区域划分为多个小区域；
* 子域解之间通过边界条件迭代更新，直到全局一致；
* 还结合了「coarse space correction」技术，用于加速全局收敛。


