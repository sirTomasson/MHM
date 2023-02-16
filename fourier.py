import numpy as np
import scipy as sp


def fourier_ab(time, signal, n):
    a, b, omega = [], [], []
    T = time[-1]
    for k in range(n):
        omega_k = 2 * np.pi * k * (1 / T)
        ya = signal * np.cos(time * omega_k)
        yb = signal * np.sin(time * omega_k)
        ak = (1 / T) * sp.integrate.trapz(ya, time)
        bk = (1 / T) * sp.integrate.trapz(yb, time)

        a.append(ak)
        b.append(bk)
        omega.append(omega_k)

    return omega, a, b


def inv_fourier_ab(a, b, time, omega):
    signal = np.zeros(len(time))
    for omega_k, ak, bk in zip(omega[1:], a[1:], b[1:]):
        signal += ak * np.cos(omega_k * time) + bk * np.sin(omega_k * time)

    return a[0] + 2 * signal


def fourier_c(time, signal, n):
    c, omega = [], []
    T = time[-1]
    for k in range(-n, n + 1):
        omega_k = 2 * np.pi * k * (1 / T)
        y = signal * np.exp(1j * omega_k * time)
        ck = (1 / T) * sp.integrate.trapz(y, time)

        c.append(ck)
        omega.append(omega_k)

    return omega, c


def inv_fourier_c(c, time, omega):
    signal = np.full(len(time), 0j)
    for ck, omega_k in zip(c, omega):
        signal += ck * np.exp(-1j * omega_k * time)

    return signal.real


def polynomial_signal(time):
    a = np.array([[0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1],
                  [16, 8, 4, 2, 1],
                  [81, 27, 9, 3, 1],
                  [256, 64, 16, 4, 1]])
    b = np.array([0, 1, -1, 2, 0])
    x = np.linalg.solve(a, b)

    signal = x[0] * time ** 4 + x[1] * time ** 3 + x[2] * time ** 2 + x[3] * time + x[4]
    return signal


def polynomial_signal_non_periodic(time):
    a = np.array([[0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1],
                  [16, 8, 4, 2, 1],
                  [81, 27, 9, 3, 1],
                  [256, 64, 16, 4, 1]])
    b = np.array([0, 1, -1, 2, 1])
    x = np.linalg.solve(a, b)

    signal = x[0] * time ** 4 + x[1] * time ** 3 + x[2] * time ** 2 + x[3] * time + x[4]
    return signal
