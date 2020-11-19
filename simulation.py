import numpy as np
from scipy import integrate
from scipy import constants

def calculate_temperature(K_wave, sample_rate):
    dt = 1. / sample_rate
    N = 3.2e6
    E = 3 * N * constants.Boltzmann * 1.5e-6
    density_of_state_scale = 1 / (constants.hbar**3 * (70 * 2 * np.pi)**3)
    beta = 1 / (constants.Boltzmann * 1.5e-6)
    mu = 0.
    m = 40 * constants.atomic_mass
    sigma = 4 * np.pi * (157 * 5.2917721067e-11)**2
    gamma = 1. / 135
    for i in range(len(K_wave)-1):
        beta, mu = calculate_beta_and_mu(N, E, K_wave[i], beta, mu, density_of_state_scale)
        dN_1 = calculate_dN_1(K_wave[i], beta, mu, density_of_state_scale) * dt * m * sigma / (np.pi**2 * constants.hbar**3)
        dE_1 = calculate_dE_1(K_wave[i], beta, mu, density_of_state_scale) * dt * m * sigma / (np.pi**2 * constants.hbar**3)
        dN_2 = gamma * N * dt
        dE_2 = gamma * E * dt
        if (K_wave[i] - K_wave[i+1]) > 0:
            dN_3 = calculate_dN_3(K_wave[i+1], K_wave[i], beta, mu, density_of_state_scale)
            dE_3 = calculate_dE_3(K_wave[i+1], K_wave[i], beta, mu, density_of_state_scale)
        else:
            dN_3 = 0
            dE_3 = 0
        N = N - dN_1 - dN_2 - dN_3
        E = E - dE_1 - dE_2 - dE_3
    beta, _ = calculate_beta_and_mu(N, E, K_wave[-1], beta, mu, density_of_state_scale)
    E_F = (6 * N)**(1/3) * constants.hbar * 70 * 2 * np.pi
    return 1 / (beta * E_F)


def calculate_beta_and_mu(N, E, K, beta_0, mu_0, density_of_state_scale):
    beta = beta_0
    mu = mu_0
    while True:
        f_1 = calculate_N(K, beta, mu, density_of_state_scale) - N
        f_2 = calculate_E(K, beta, mu, density_of_state_scale) - E
        f = np.matrix([[f_1], [f_2]])
        J_11 = partial_beta_N(K, beta, mu, density_of_state_scale)
        J_12 = partial_mu_N(K, beta, mu, density_of_state_scale)
        J_21 = partial_beta_E(K, beta, mu, density_of_state_scale)
        J_22 = partial_mu_E(K, beta, mu, density_of_state_scale)
        J = np.matrix([[J_11, J_12], [J_21, J_22]])
        delta = - np.dot(J.I, f)
        if abs(delta[0,0]) < 0.001 * abs(beta) and abs(delta[1,0]) < 0.001 * abs(mu):
            break
        beta = beta - delta[0,0]
        mu = mu - delta[1,0]
    return beta, mu

def calculate_N(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    N_e = e**2 / (1 + np.exp(beta_e_minus_mu))
    N = 0.5 * np.sum(N_e) * de * density_of_state_scale
    return N

def calculate_E(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    E_e = e**3 / (1 + np.exp(beta_e_minus_mu))
    E = 0.5 * np.sum(E_e) * de * density_of_state_scale
    return E

def partial_beta_N(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    partial_beta_N_e = e**2 * (e - mu) / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
    partial_beta_N = 0.5 * np.sum(partial_beta_N_e) * de * density_of_state_scale
    return partial_beta_N

def partial_mu_N(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    partial_mu_N_e = e**2 / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
    partial_mu_N = 0.5 * beta * np.sum(partial_mu_N_e) * de * density_of_state_scale
    return partial_mu_N

def partial_beta_E(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    partial_beta_E_e = e**3 * (e - mu) / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
    partial_beta_E = 0.5 * np.sum(partial_beta_E_e) * de * density_of_state_scale
    return partial_beta_E

def partial_mu_E(K, beta, mu, density_of_state_scale):
    de = 0.0001 * K
    e = np.arange(0, K, de)
    beta_e_minus_mu = beta * (e - mu)
    partial_mu_E_e = e**3 / (2 + np.exp(beta_e_minus_mu) + np.exp(-beta_e_minus_mu))
    partial_mu_E = 0.5 * beta * np.sum(partial_mu_E_e) * de * density_of_state_scale
    return partial_mu_E

def dN_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * y**2 / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

def calculate_dN_1(K, beta, mu, density_of_state_scale):
    max_y = lambda x: 2 * K - x
    min_z = lambda x, y: x + y - K
    return  integrate.tplquad(dN_1_kernel, K, 2 * K, 0, max_y, min_z, K, args=(beta, mu, density_of_state_scale))

def dE_1_kernel(z, y, x, beta, mu, density_of_state_scale):
    return 0.5 * density_of_state_scale * y**3 / (1 + np.exp(beta * (x + y - z - mu))) / (1 + np.exp(beta * (z - mu))) / (1 + np.exp(-beta * (y - mu)))

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


if __name__ == '__main__':
    t = np.arange(0., 50., 1.)
    K_0 = 12 * constants.Boltzmann * 1.5e-6
    K_wave = K_0 * np.exp(-t*0.15)
    temperature = calculate_temperature(K_wave, 1.0)
    print(temperature)