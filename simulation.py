import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import constants
import utilities

def calculate_temperature(K_wave, omega_r, omega_z, sample_rate):
    dt = 1. / sample_rate
    hbar = constants.hbar
    kB = constants.Boltzmann
    a0 = 5.2917721067e-11
    T = 10e-6
    sigma = 4 * np.pi * (137 * a0)**2
    m = 6 * constants.atomic_mass
    N = 8649427.09721088
    E = 2.251483454531782e-21
    beta = 1 / (kB * T)
    mu = -3e-28
    gamma = 1 / 1000

    for i in range(len(K_wave)-1):
        density_of_state_scale = 1 / (hbar**3 * omega_r[i]**2 * omega_z[i])
        beta, mu = calculate_beta_and_mu(N, E, K_wave[i], beta, mu, density_of_state_scale)
        dN_1 = calculate_dN_1(K_wave[i], beta, mu, density_of_state_scale)[0] * dt * m * sigma / (np.pi**2 * hbar**3)
        dE_1 = calculate_dE_1(K_wave[i], beta, mu, density_of_state_scale)[0] * dt * m * sigma / (np.pi**2 * hbar**3)
        dN_2 = gamma * N * dt
        dE_2 = gamma * E * dt
        if (K_wave[i] - K_wave[i+1]) > 0:
            dN_3 = calculate_dN_3(K_wave[i+1], K_wave[i], beta, mu, density_of_state_scale)[0]
            dE_3 = calculate_dE_3(K_wave[i+1], K_wave[i], beta, mu, density_of_state_scale)[0]
        else:
            dN_3 = 0
            dE_3 = 0
        N = N - dN_1 - dN_2 - dN_3
        E = E - dE_1 - dE_2 - dE_3

    beta, _ = calculate_beta_and_mu(N, E, K_wave[-1], beta, mu, density_of_state_scale)
    E_F = (6 * N)**(1/3) * hbar * (omega_r[-1]**2 * omega_z[-1])**(1/3)
    T_over_TF = 1 / (beta * E_F)
    print("N = %f, T/T_F = %f"%(N, T_over_TF))
    return T_over_TF

def N_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * x**2 / (1 + np.exp(beta * (x - mu)))

def E_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * x**3 / (1 + np.exp(beta * (x - mu)))

def J_11_kernel(x, beta, mu, density_of_state_scale):
    return -0.5 * density_of_state_scale * x**2 * (x - mu) / (2 + np.exp(beta * (x - mu) + np.exp(-beta * (x - mu))))

def J_12_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * beta * x**2 / (2 + np.exp(beta * (x - mu) + np.exp(-beta * (x - mu))))

def J_21_kernel(x, beta, mu, density_of_state_scale):
    return -0.5 * density_of_state_scale * x**3 * (x - mu) / (2 + np.exp(beta * (x - mu) + np.exp(-beta * (x - mu))))

def J_22_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * beta * x**3 / (2 + np.exp(beta * (x - mu) + np.exp(-beta * (x - mu))))

def N_E_diff(x, N, E, K, density_of_state_scale):
    beta = float(x[0])
    mu = float(x[1])
    N_diff = integrate.quad(N_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0] - N
    E_diff = integrate.quad(E_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0] - E
    return [N_diff, E_diff]

def Jacobi(x, N, E, K, density_of_state_scale):
    beta = float(x[0])
    mu = float(x[1])
    J_11 = integrate.quad(J_11_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0]
    J_12 = integrate.quad(J_12_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0]
    J_21 = integrate.quad(J_21_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0]
    J_22 = integrate.quad(J_22_kernel, 0, K, args=(beta, mu, density_of_state_scale))[0]
    return [[J_11, J_12], [J_21, J_22]]

def calculate_beta_and_mu(N, E, K, beta, mu, density_of_state_scale):
    return optimize.fsolve(N_E_diff, [beta, mu], args=(N, E, K, density_of_state_scale), fprime=Jacobi)

# def calculate_beta_and_mu(N, E, K, beta_0, mu_0, density_of_state_scale):
#     beta = beta_0
#     mu = mu_0
#     while True:
#         f_1 = calculate_N(K, beta, mu, density_of_state_scale) / N - 1
#         f_2 = calculate_E(K, beta, mu, density_of_state_scale) / E - 1
#         f = np.matrix([[f_1], [f_2]])
#         J_11 = partial_beta_N(K, beta, mu, density_of_state_scale) / N
#         J_12 = partial_mu_N(K, beta, mu, density_of_state_scale) / N
#         J_21 = partial_beta_E(K, beta, mu, density_of_state_scale) / E
#         J_22 = partial_mu_E(K, beta, mu, density_of_state_scale) / E
#         J = np.matrix([[J_11, J_12], [J_21, J_22]])
#         delta = - np.dot(J.I, f)
#         if abs(delta[0,0]) < 0.001 * abs(beta) and abs(delta[1,0]) < 0.001 * abs(mu):
#             break
#         beta = beta + delta[0,0]
#         mu = mu + delta[1,0]
#     return beta, mu

def calculate_N(K, beta, mu, density_of_state_scale):
    de = 0.00001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    N_e = e**2 / (1 + np.exp(beta_e_minus_mu))
    N = 0.5 * np.sum(N_e) * de * density_of_state_scale
    return N

def calculate_E(K, beta, mu, density_of_state_scale):
    de = 0.00001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    E_e = e**3 / (1 + np.exp(beta_e_minus_mu))
    E = 0.5 * np.sum(E_e) * de * density_of_state_scale
    return E

# def partial_beta_N(K, beta, mu, density_of_state_scale):
#     de = 0.00001 * K
#     e = np.arange(0, K, de)
#     beta_e_minus_mu = beta * (e - mu)
#     partial_beta_N_e = e**2 * (e - mu) / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
#     partial_beta_N = -0.5 * np.sum(partial_beta_N_e) * de * density_of_state_scale
#     return partial_beta_N

# def partial_mu_N(K, beta, mu, density_of_state_scale):
#     de = 0.00001 * K
#     e = np.arange(0, K, de)
#     beta_e_minus_mu = beta * (e - mu)
#     partial_mu_N_e = e**2 / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
#     partial_mu_N = 0.5 * beta * np.sum(partial_mu_N_e) * de * density_of_state_scale
#     return partial_mu_N

# def partial_beta_E(K, beta, mu, density_of_state_scale):
#     de = 0.0001 * K
#     e = np.arange(0, K, de)
#     beta_e_minus_mu = beta * (e - mu)
#     partial_beta_E_e = e**3 * (e - mu) / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
#     partial_beta_E = -0.5 * np.sum(partial_beta_E_e) * de * density_of_state_scale
#     return partial_beta_E

# def partial_mu_E(K, beta, mu, density_of_state_scale):
#     de = 0.00001 * K
#     e = np.arange(0, K, de)
#     beta_e_minus_mu = beta * (e - mu)
#     partial_mu_E_e = e**3 / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
#     partial_mu_E = 0.5 * beta * np.sum(partial_mu_E_e) * de * density_of_state_scale
#     return partial_mu_E

def dN_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * y**2 / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

def calculate_dN_1(K, beta, mu, density_of_state_scale):
    max_y = lambda x: 2 * K - x
    min_z = lambda x, y: x + y - K
    return  integrate.tplquad(dN_1_kernel, K, 2 * K, 0, max_y, min_z, K, args=(beta, mu, density_of_state_scale))

def dE_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * x * y**2 / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

def calculate_dE_1(K, beta, mu, density_of_state_scale):
    max_y = lambda x: 2 * K - x
    min_z = lambda x, y: x + y - K
    return  integrate.tplquad(dE_1_kernel, K, 2 * K, 0, max_y, min_z, K, args=(beta, mu, density_of_state_scale))

def dN_3_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * x**2 / (1 + np.exp(beta * (x - mu)))

def calculate_dN_3(K_prime, K, beta, mu, density_of_state_scale):
    return integrate.quad(dN_3_kernel, K_prime, K, args=(beta, mu, density_of_state_scale))

def dE_3_kernel(x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * x**3 / (1 + np.exp(beta * (x - mu)))

def calculate_dE_3(K_prime, K, beta, mu, density_of_state_scale):
    return integrate.quad(dE_3_kernel, K_prime, K, args=(beta, mu, density_of_state_scale))

def main():
    K_0 = 4.2644e-28
    omega_r_0 = 13135.56
    omega_z_0 = 99.86535

    sample_rate = 200
    # t_step = 1 / sample_rate
    # t = np.arange(0., 3.21, t_step)
    params = np.array([-4.71779894, 2.65982562, -0.84004817, -2.42029854, -0.011407, 0.64039272, -0.28129203])
    K_wave = utilities.waveform(K_0, K_0 / 25, 3.21, sample_rate, params)
    # K_wave = K_0 * np.exp(-t)
    omega_r = omega_r_0 * np.sqrt(K_wave / K_wave[0])
    omega_z = omega_z_0 * np.sqrt(K_wave / K_wave[0])
    calculate_temperature(K_wave, omega_r, omega_z, sample_rate)

def calculate_N_E():
    hbar = constants.hbar
    kB = constants.Boltzmann
    T = 10e-6
    K_0 = 4.2644e-28
    omega_r_0 = 13135.56
    omega_z_0 = 99.86535
    density_of_state_scale = 1 / (hbar**3 * omega_r_0**2 * omega_z_0)
    beta = 1 / (kB * T)
    mu = -3e-28

    N = calculate_N(K_0, beta, mu, density_of_state_scale)
    E = calculate_E(K_0, beta, mu, density_of_state_scale)
    print(N)
    print(E)

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

if __name__ == '__main__':
    main()