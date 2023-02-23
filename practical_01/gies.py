import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.signal as sig
import scipy.integrate as integ
import readndf
from clean_optotrack import interpolate_advanced

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


def subtract_baseline(Fy):
    empty_start = 0
    empty_end = 3000
    return Fy - np.mean(Fy[empty_start:empty_end])


def get_baseline(Fy):
    baseline_start = 7000
    baseline_end = 11000
    return np.mean(Fy[baseline_start:baseline_end])


def calculate_height_plate(Fy, jump_boundaries, height):
    Fy = subtract_baseline(Fy)
    baseline = get_baseline(Fy)

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

# col = 'Fy_2'
# plot_df(df_gies, [col])
regression = calibrate()
gravity = 9.81
# height = calculate_height(df_gies[col], [25000, 27000], 4.5)

column_names = ['hip', 'knee', 'ankle', 'shoulder']
com_offsets = [0.625, 0.6, 0.714]  # shin, thigh, torso
weight_proportions = []
"""Only load the data of the Z-axis"""
df_TN11 = pd.DataFrame(readndf.readndf("data/TN000011.ndf")[2])  # only interested in Z?
df_TN11.columns = column_names
df_TN11 = df_TN11.apply(interpolate_advanced)

plt.figure()
plt.plot(df_TN11, label=column_names)
plt.xlim([0, 12000])
plt.ylim([-50.0, 2000.0])
plt.ylabel('marker position [mm]')
plt.title('Interpolated signal')
plt.legend()



def com(series):
    com_offsets = [0.714, 0.6, 0.625]  # torso, shin, thigh
    weight_proportions = [53 / 77, 14 / 77, 10 / 77]

    vals = []
    for col in ['shoulder', 'hip', 'knee', 'ankle']:
        vals.append(series[col])

    coms = []
    for i in range(3):
        relative_com = (vals[i] - vals[i + 1]) * com_offsets[i]
        coms.append(relative_com + vals[i + 1])

    com = 0
    for proportion, center in zip(weight_proportions, coms):
        com += proportion * center
    print(f"Com {com}")
    return com


def get_baseline_com(df):
    start = 1500
    end = 3000
    res = df[start:end].mean()

    return com(res)

def calculate_height_optotrack(df):
    baseline_com = get_baseline_com(df)

    # for col in df.columns:
    #     df[col] -= baselines[col]

    # plt.figure()
    # plt.plot(df, label=column_names)
    # plt.xlim([0, 12000])
    # plt.ylim([-50.0, 2000.0])
    # plt.ylabel('marker position [mm]')
    # plt.title('Interpolated signal')
    # plt.legend()


calculate_height_optotrack(df_TN11)

plt.show()
