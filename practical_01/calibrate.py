import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_measured_voltages():
    df_measured = pd.DataFrame()
    df_TN000003 = pd.read_csv('data/TN000003.afp2',
                              delimiter='\t',
                              usecols=[0, 1, 2, 3],
                              names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

    df_measured['V'] = df_TN000003.sum(axis=1)
    return df_measured


def get_mean_voltages(mv):
    step = 30000
    v = []
    for k in range(0, len(mv), step):
        s = int(k + (step / 3))
        df = mv[s:s + 10000]
        v.append(df.mean().values[0])

    return v


def calibrate(w, v, reg):
    reg.fit(w.reshape(-1, 1), v)
    return reg


def weight(v):
    return v / calibration.coef_


real_forces = np.array([0, 10, 20, 30, 40, 50, 60, 86.1, 115.22, 86.1, 60, 50, 40, 30, 20, 10, 0])
df_measured_voltages = get_measured_voltages()
measured_voltages = df_measured_voltages.to_numpy().reshape(-1)
mean_voltages = get_mean_voltages(df_measured_voltages)

calibration = LinearRegression()
calibration = calibrate(real_forces, mean_voltages, calibration)
