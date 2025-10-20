from scipy.io import loadmat

# 读取 .mat 文件
data = loadmat('2D_TM_all_processed_jiequ_200_300.mat')

# 查看有哪些变量
print(data.keys())