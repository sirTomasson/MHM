import numpy as np
import fourier
import matplotlib.pyplot as plt
import scipy as sp

# 1.1 and 1.2
n = 5

time = np.linspace(0, 1, 101)
signal = np.sin(time * 2*np.pi)

omega, a, b = fourier.fourier_ab(time, signal, n)
print(f"Omega: {omega[:2]}, A:{a[:2]}, B: {b[:2]}")

# 1.3 and 1.4
signal_inv = fourier.inv_fourier_ab(a, b, time, omega)
print("sig_in")
print(signal[:6])

print("sig_inv")
print(signal_inv[:6])

# 1.5 Interesting (polynomial) signal
time = np.linspace(0, 4, 201)
signal = fourier.polynomial_signal(time)

plt.figure()
plt.plot(time, signal, label='original')

ns = [2, 5, 10, 20, 50, 100, 150]

# 1.6 Test inverse Fourier with our interesting (polynomial) signal
for n in ns:
    omega, a, b = fourier.fourier_ab(time, signal, n=n)

    signal_inv = fourier.inv_fourier_ab(a, b, time, omega)
    plt.plot(time, signal_inv, label=f"n={n}")

plt.legend()
plt.title("Inverse 'simple' Fourier with our interesting (polynomial) signal")
plt.show()


# 1.7 and 1.8
n = 5

time = np.linspace(0, 1, 101)
signal = np.sin(time * 2*np.pi)

omega, c = fourier.fourier_c(time, signal, n)
print(f"Omega: {omega[:2]}, C:{c[:4]}")

signal_inv = fourier.inv_fourier_c(c, time, omega)
print("sig_in")
print(signal[:6])

print("sig_inv")
print(signal_inv[:6])


# 1.9
time = np.linspace(0, 4, 201)
signal = fourier.polynomial_signal(time)

plt.figure()
plt.plot(time, signal, label="original")

for n in ns:
    omega, c = fourier.fourier_c(time, signal, n)
    signal_inv = fourier.inv_fourier_c(c, time, omega)
    plt.plot(time, signal_inv, label=f"n={n}")

plt.legend()
plt.title("Inverse 'complex' Fourier with our interesting (polynomial) signal")
plt.show()


# 1.10
n = 5
f, (ax1, ax2) = plt.subplots(1, 2)

omega, c = fourier.fourier_c(time, signal, n)
signal_inv = fourier.inv_fourier_c(c, time, omega)

mod = np.abs(c)
phase = np.angle(c)

ax1.plot(time, signal_inv, label=f"n={n}")
ax1.set_title("Signal")
ax1.legend()

ax2.plot(mod, label="modulus")
ax2.plot(phase, label="phase")
ax2.set_title("Modulus and phase")
ax2.legend()

plt.show()

# 1.11
time = np.linspace(0, 4, 801)
signal = fourier.polynomial_signal_non_periodic(time)

plt.figure()
plt.plot(time, signal, label="original")

for n in ns:
    omega, c = fourier.fourier_c(time, signal, n)
    signal_inv = fourier.inv_fourier_c(c, time, omega)
    plt.plot(time, signal_inv, label=f"n={n}")

plt.legend()
plt.title("Inverse 'complex' Fourier with our interesting (polynomial) non-periodic signal")
plt.show()


# 1.12
time = np.linspace(0, 1, 101)
signal = np.sin(time * 2*np.pi)

plt.figure()
plt.plot(time, signal, label="original", linewidth=4.0)

c = sp.fft.fft(signal)
signal_inv = sp.fft.ifft(c)

plt.plot(time, signal_inv.real, label=f"n={n}")
plt.legend()
plt.title("Sinusoid signal and its inverse using Scipy methods")
plt.show()


# 1.13
time = np.linspace(0, 4, 201)
signal = fourier.polynomial_signal(time)

plt.figure()
plt.plot(time, signal, label="original", linewidth=4.0)

c = sp.fft.fft(signal)
signal_inv = sp.fft.ifft(c)

plt.plot(time, signal_inv.real, label=f"ifft")
plt.legend()
plt.title("Polynomial Signal and its inverse using Scipy methods")
plt.show()


# 1.14
time = np.linspace(0, 4, 201)
signal = fourier.polynomial_signal_non_periodic(time)

plt.figure()
plt.plot(time, signal, label="original", linewidth=4.0)

c = sp.fft.fft(signal)
signal_inv = sp.fft.ifft(c)

plt.plot(time, signal_inv.real, label=f"ifft")
plt.legend()
plt.title("Non-periodic signal and its inverse using Scipy methods")
plt.show()

