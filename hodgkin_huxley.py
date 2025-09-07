import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley模型参数
gNa = 120.0  # 钠通道最大电导 (mS/cm^2)
gK = 36.0    # 钾通道最大电导 (mS/cm^2)
gL = 0.3     # 泄漏通道最大电导 (mS/cm^2)
ENa = 50.0   # 钠离子平衡电位 (mV)
EK = -77.0   # 钾离子平衡电位 (mV)
EL = -54.387 # 泄漏通道平衡电位 (mV)
Cm = 1.0     # 膜电容 (uF/cm^2)

# 时间参数
t_max = 50.0  # ms

dt = 0.01    # 步长 ms
t = np.arange(0, t_max, dt)

# 外部电流刺激
def I_ext(time):
    # 在10-40ms之间施加1.5 uA/cm^2的电流
    return 1.5 if 10 < time < 40 else 0.0

# 门控变量的alpha和beta函数
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)
def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

# 初始化变量
V = np.zeros_like(t)
V[0] = -65.0  # 初始膜电位 (mV)
m = np.zeros_like(t)
h = np.zeros_like(t)
n = np.zeros_like(t)

# 初始门控变量
m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

# 主循环
def run_hh():
    for i in range(1, len(t)):
        # 计算离子流
        INa = gNa * m[i-1]**3 * h[i-1] * (V[i-1] - ENa)
        IK = gK * n[i-1]**4 * (V[i-1] - EK)
        IL = gL * (V[i-1] - EL)
        # 更新膜电位
        V[i] = V[i-1] + dt * (I_ext(t[i-1]) - INa - IK - IL) / Cm
        # 更新门控变量
        m[i] = m[i-1] + dt * (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1])
        h[i] = h[i-1] + dt * (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1])
        n[i] = n[i-1] + dt * (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1])

run_hh()

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(t, V, label='膜电位 (mV)')
plt.xlabel('时间 (ms)')
plt.ylabel('膜电位 (mV)')
plt.title('Hodgkin-Huxley 神经元膜电位模拟')
plt.legend()
plt.grid(True)
plt.show()
