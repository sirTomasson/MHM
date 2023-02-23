import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.signal as sig
import scipy.integrate as integ

df_gies = pd.DataFrame()
df_TN2 = pd.read_csv('data/TN000011.afp2',
                     delimiter='\t',
                     usecols=[0, 1, 2, 3],
                     names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TNR = pd.read_csv('data/TN000012.afp2',
                     delimiter='\t',
                     usecols=[0, 1, 2, 3],
                     names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TNL = pd.read_csv('data/TN000013.afp2',
                     delimiter='\t',
                     usecols=[0, 1, 2, 3],
                     names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_gies['Fy_2'] = df_TN2.sum(axis=1)
df_gies['Fy_R'] = df_TNR.sum(axis=1)
df_gies['Fy_L'] = df_TNL.sum(axis=1)


df_calibration = pd.DataFrame()
df_TN000003 = pd.read_csv('data/TN000003.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_calibration['Fy'] = df_TN000003.sum(axis=1)


def plot_df(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        plt.figure()
        plt.plot(df[col], label=col)
        plt.xlim([0, 50000])
        plt.ylim([0, 10.0])
        plt.legend()


def calibrate():
    stepsize = 30000
    voltages = []
    for i in range(0, len(df_calibration), stepsize):
        start = i + 5000
        df = df_calibration[start:start + 20000]
        voltages.append(df.mean().values[0])

    voltages = np.array(voltages)
    real_forces = np.array([0, 10, 20, 30, 40, 50, 60, 86.1, 115.22, 86.1, 60, 50, 40, 30, 20, 10, 0])

    reg = LinearRegression()
    reg.fit(real_forces.reshape(-1, 1), voltages)

    a = reg.coef_
    return lambda y: (y) / a[0]


def get_empty_baseline(Fy):
    empty_start = 0
    empty_end = 3000
    return np.mean(Fy[empty_start:empty_end])


def get_baselines(Fy):
    baseline_start = 7000
    baseline_end = 11000
    return np.mean(Fy[baseline_start:baseline_end])


def calculate_height(Fy, jump_boundaries, height):
    empty = get_empty_baseline(Fy)
    Fy = Fy - empty
    baseline = get_baselines(Fy)

    mass = regression(baseline)
    Fg = mass * -gravity

    jump_start = jump_boundaries[0]
    Fy_jump = Fy[jump_start:jump_boundaries[1]]
    peaks, _ = sig.find_peaks(Fy_jump, height=height)
    if len(peaks > 0):
        jump_end = peaks[0] + jump_start

        Fy_jump = Fy[jump_start:jump_end]
        Fy_jump = np.array([regression(volt) for volt in Fy_jump]) * 9.81
        a = (Fy_jump + Fg) / mass

        v = integ.cumtrapz(a) / 1000
        r = integ.cumtrapz(v) / 1000
        height = max(r)

        print(f"Height {height}")

        plt.figure()
        plt.plot(r)

        return height

col = 'Fy_2'
plot_df(df_gies, [col])

regression = calibrate()
gravity = 9.81
height = calculate_height(df_gies[col], [25000, 27000], 4.5)

plt.show()