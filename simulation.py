import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import constants
import utilities

def calculate_temperature(K_wave, sample_rate):
    dt = 1. / sample_rate
    m = 40 * constants.atomic_mass
    pi = np.pi
    hbar = constants.hbar
    kB = constants.Boltzmann
    a0 = 5.2917721067e-11
    T = 1.5e-6
    sigma = 4 * pi * (157 * a0)**2
    omega = 70 * 2 * pi
    N = 3.2e6
    E = 3 * N * kB * T
    beta = 1 / (kB * T)
    mu = -6.87081756e-29
    gamma = 1 / 135

    beta_list = []
    mu_list = []
    for i in range(len(K_wave)-1):
        # 用三点多项式外推下一个beta和mu
        if i > 3:
            beta = 3 * beta_list[-1] - 3 * beta_list[-2] + beta_list[-3]
            mu = 3 * mu_list[-1] - 3 * mu_list[-2] + mu_list[-3]
        # 使能量无量纲化，取缩放因子为N / E
        energy_scale = N / E
        E_scaled = E * energy_scale
        m_scaled = m * energy_scale
        hbar_scaled = hbar * energy_scale
        K_scaled = K_wave[i] * energy_scale
        beta_scaled = beta / energy_scale
        mu_scaled = mu * energy_scale
        density_of_state_scale = 1 / (hbar_scaled * omega)**3
        # 牛顿迭代法求解beta和mu
        # print("t = %e: N = %f, E = %e"%(i * dt, N, E))
        beta_scaled, mu_scaled = calculate_beta_and_mu(N, E_scaled, K_scaled, beta_scaled, mu_scaled, density_of_state_scale)
        # 更新带量纲的beta和mu
        beta = beta_scaled * energy_scale
        mu = mu_scaled / energy_scale
        beta_list.append(beta)
        mu_list.append(mu)
        E_F = (6 * N)**(1/3) * hbar * omega
        T_over_TF = 1 / (beta * E_F)
        # print("beta = %e, mu = %e, T/T_F = %e"%(beta, mu, T_over_TF))
        # 计算各部分的N和E损失
        dN_1 = calculate_dN_1(K_scaled, beta_scaled, mu_scaled, density_of_state_scale)[0] * dt * m_scaled * sigma / (pi**2 * hbar_scaled**3)
        dE_1 = calculate_dE_1(K_scaled, beta_scaled, mu_scaled, density_of_state_scale)[0] * dt * m_scaled * sigma / (pi**2 * hbar_scaled**3)
        dN_2 = gamma * N * dt
        dE_2 = gamma * E_scaled * dt
        if (K_wave[i] - K_wave[i+1]) > 0:
            K_next_scaled = K_wave[i+1] * energy_scale
            dN_3 = calculate_dN_3(K_next_scaled, K_scaled, beta_scaled, mu_scaled, density_of_state_scale)[0]
            dE_3 = calculate_dE_3(K_next_scaled, K_scaled, beta_scaled, mu_scaled, density_of_state_scale)[0]
        else:
            dN_3 = 0
            dE_3 = 0
        # 更新N和E
        N = N - dN_1 - dN_2 - dN_3
        E = E - (dE_1 + dE_2 + dE_3) / energy_scale
        
    # 计算最后的T/T_F
    E_F = (6 * N)**(1/3) * hbar * omega
    T_over_TF = 1 / (beta * E_F)
    print("N = %f, T/T_F = %e"%(N, T_over_TF))
    return T_over_TF

def N_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * x**2) / (1 + np.exp(beta * (x - mu)))

def E_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * x**3) / (1 + np.exp(beta * (x - mu)))

def J_11_kernel(x, beta, mu, density_of_state_scale):
    return -0.5 * (density_of_state_scale * x**2 * (x - mu)) / (2 + np.exp(beta * (x - mu)) + np.exp(-beta * (x - mu)))

def J_12_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * beta * x**2)  / (2 + np.exp(beta * (x - mu)) + np.exp(-beta * (x - mu)))

def J_21_kernel(x, beta, mu, density_of_state_scale):
    return -0.5 * (density_of_state_scale * x**3 * (x - mu)) / (2 + np.exp(beta * (x - mu)) + np.exp(-beta * (x - mu)))

def J_22_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * beta * x**3) / (2 + np.exp(beta * (x - mu)) + np.exp(-beta * (x - mu)))

def N_E_diff(x, N, E, K, density_of_state_scale):
    beta = float(x[0])
    mu = float(x[1])
    lower_limit = min(0, mu + 600 / beta) if beta > 0 else max(0, mu + 600 / beta)
    upper_limit = min(K, mu + 600 / beta) if beta > 0 else max(K, mu + 600 / beta)
    N_diff = integrate.quad(N_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0] - N
    E_diff = integrate.quad(E_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0] - E
    return [N_diff, E_diff]

def Jacobi(x, N, E, K, density_of_state_scale):
    beta = float(x[0])
    mu = float(x[1])
    lower_limit = max(min(0, mu + 600 / abs(beta)), mu - 600 / abs(beta))
    upper_limit = max(min(K, mu + 600 / abs(beta)), mu - 600 / abs(beta))
    J_11 = integrate.quad(J_11_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0]
    J_12 = integrate.quad(J_12_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0]
    J_21 = integrate.quad(J_21_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0]
    J_22 = integrate.quad(J_22_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))[0]
    return [[J_11, J_12], [J_21, J_22]]

def calculate_beta_and_mu(N, E, K, beta, mu, density_of_state_scale):
    return optimize.fsolve(N_E_diff, [beta, mu], args=(N, E, K, density_of_state_scale), fprime=Jacobi)

def dN_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * y**2) / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

def calculate_dN_1(K, beta, mu, density_of_state_scale):
    if beta > 0:
        min_x = min(K, mu + 1800 / beta) if beta > 0 else max(0, mu + 1800 / beta)
        max_x = min(2 * K, mu + 1800 / beta) if beta > 0 else max(2 * K, mu + 1800 / beta)
        min_y = lambda x: min(max(0, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
        max_y = lambda x: min(max(2 * K - x, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
        min_z = lambda x, y: max(min(x + y - K, mu + 600 / beta), x + y - (mu + 600 / beta))
        max_z = lambda x, y: max(min(K, mu + 600 / beta), x + y - (mu + 600 / beta))
    else:
        min_x = max(0, mu + 1800 / beta)
        max_x = max(2 * K, mu + 1800 / beta)
        min_y = lambda x: max(min(0, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
        max_y = lambda x: max(min(2 * K - x, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
        min_z = lambda x, y: min(max(x + y - K, mu + 600 / beta), x + y - (mu + 600 / beta))
        max_z = lambda x, y: min(max(K, mu + 600 / beta), x + y - (mu + 600 / beta))
    return  integrate.tplquad(dN_1_kernel, min_x, max_x, min_y, max_y, min_z, max_z, args=(beta, mu, density_of_state_scale))

def dE_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * x * y**2) / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

def calculate_dE_1(K, beta, mu, density_of_state_scale):
    min_x = min(K, mu + 1800 / beta)
    max_x = min(2 * K, mu + 1800 / beta)
    min_y = lambda x: min(max(0, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
    max_y = lambda x: min(max(2 * K - x, mu - 600 / beta), (2 * mu + 1200 / beta) - x)
    min_z = lambda x, y: max(min(x + y - K, mu + 600 / beta), x + y - (mu + 600 / beta))
    max_z = lambda x, y: max(min(K, mu + 600 / beta), x + y - (mu + 600 / beta))
    return  integrate.tplquad(dE_1_kernel, min_x, max_x, min_y, max_y, min_z, max_z, args=(beta, mu, density_of_state_scale))

def dN_3_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * x**2) / (1 + np.exp(beta * (x - mu)))

def calculate_dN_3(K_prime, K, beta, mu, density_of_state_scale):
    lower_limit = min(K_prime, mu + 600 / beta)
    upper_limit = min(K, mu + 600 / beta)
    return integrate.quad(dN_3_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))

def dE_3_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * (density_of_state_scale * x**3) / (1 + np.exp(beta * (x - mu)))

def calculate_dE_3(K_prime, K, beta, mu, density_of_state_scale):
    lower_limit = min(K_prime, mu + 600 / beta)
    upper_limit = min(K, mu + 600 / beta)
    return integrate.quad(dE_3_kernel, lower_limit, upper_limit, args=(beta, mu, density_of_state_scale))

def main():
    kB = constants.Boltzmann
    T = 1.5e-6
    K_0 = 12 * kB * T

    sample_rate = 20
    tf = 10
    t_step = 1 / sample_rate
    t = np.arange(0., tf, t_step)
    K_wave = K_0 * np.exp(-t * (np.log(25) / tf))
    print("tf = %f:"%tf)
    result = calculate_temperature(K_wave, sample_rate)
    print(result)

def calculate_trap():
    pi = np.pi
    c = constants.c
    # Li原子质量
    m = 6 * constants.atomic_mass
    # 束腰半径
    omega_0 = 31.5e-6
    # 瑞利距离
    z_R = pi * omega_0**2 / 1064e-9
    # 原子共振频率
    omega_a = 2 * pi * c / 671e-9
    # 高斯光频率
    omega_l = 2 * pi * c / 1064e-9
    # 原子线宽
    Gamma_a = 5.87e6
    # 原子极化率
    alpha = - 3 * pi * c**2 / (2 * omega_a**3) * (Gamma_a / (omega_a - omega_l) + Gamma_a / (omega_a + omega_l))
    # 激光功率
    P = 5
    # 高斯光中心光强
    I_0 = 2 * P / (pi * omega_0**2)
    # 势阱最低点
    U_0 = alpha * I_0
    # 径向频率
    omega_x = np.sqrt(abs(4 * U_0 / (m * omega_0**2)))
    omega_y = np.sqrt(abs(4 * U_0 / (m * omega_0**2)))
    # 轴向频率
    omega_z = np.sqrt(abs(2 * U_0 / (m * z_R**2)))

    print("U_0 = %f"%(U_0*1e28))
    print("omega_x = %f"%omega_x)
    print("omega_y = %f"%omega_y)
    print("omega_z = %f"%omega_z)

def prework():
    pi = np.pi
    hbar = constants.hbar
    kB = constants.Boltzmann
    T = 1.5e-6
    N = 3.2e6
    E = 3 * N * kB * T
    K = 4 * E / N
    beta = 1 / (kB * T)
    mu = -6.87081756e-29
    omega = 70 * 2 * pi
    density_of_state_scale = 1 / (hbar * omega)**3

    print([beta, mu])
    result = calculate_beta_and_mu(N, E, K, beta, mu, density_of_state_scale)
    print(result)

if __name__ == '__main__':
    main()