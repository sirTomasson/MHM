import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Colyn's data
df_colyn = pd.DataFrame()
df_TN000004 = pd.read_csv('data/TN000004.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TN000005 = pd.read_csv('data/TN000005.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TN000006 = pd.read_csv('data/TN000006.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_colyn['Fy_2legs'] = df_TN000004.sum(axis=1)
df_colyn['Fy_L'] = df_TN000005.sum(axis=1)
df_colyn['Fy_R'] = df_TN000006.sum(axis=1)
plt.plot(df_colyn, label=['Fy_2legs [V]', 'Fy_L [V]', 'Fy_R [V]'])
plt.xlim([0, 50000])
plt.ylim([0, 5.0])
plt.legend()
plt.title("Colyn's jumps")
plt.show()

df_calibration = pd.DataFrame()
df_TN000003 = pd.read_csv('data/TN000003.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_calibration['Fy'] = df_TN000003.sum(axis=1)

step_size = 30000
window_size = 1000
voltages = []
for i in range(0, len(df_calibration), step_size):
    start = int(i + (step_size/3))
    df = df_calibration[start:start+10000]
    voltages.append(df.mean().values[0])

y = []
x = []
step_size = 30000
current = 0
for i in range(0, len(voltages)):
    v = voltages[i]
    y.extend([v, v])
    x.extend([current, current+step_size])
    current += step_size

plt.plot(df_calibration, label='measured (V)')
plt.plot(x, y, label='mean (V)')
plt.ylim([0, 2.5])
plt.legend()
plt.title("Measured and mean voltages")
plt.show()

real_forces = np.array([0, 10, 20, 30, 40, 50, 60, 86.1, 115.22, 86.1, 60, 50, 40, 30, 20, 10, 0])
voltages = np.array(voltages)
factors = real_forces / voltages

plt.plot(factors, label='factors [kg/V]')
plt.legend()
plt.ylim([40, 50])
plt.title("Factors")
plt.show()

step_size = 30000
window_size = 1000
drift = []
for i in range(0, len(df_calibration), step_size):
    start = i + 5000
    sample_start = df_calibration[start:start + 1000].mean().values[0]
    sample_end = df_calibration[start + 20000:start + 21000].mean().values[0]
    diff = sample_start - sample_end
    drift.append(diff)

plt.plot(drift, label='drift [V]')
plt.legend()
plt.title("Drift")
plt.show()

reg = LinearRegression()
reg.fit(real_forces.reshape(-1, 1), voltages)

w_test = np.arange(0, np.max(real_forces), 1).reshape(-1, 1)
v_pred = reg.predict(w_test)

plt.scatter(real_forces[0:9], voltages[0:9], label="add", marker="x")
plt.scatter(real_forces[8:17], voltages[8:17],  label="down", marker="+")
plt.plot(w_test, v_pred, label="predict", linewidth=1)
plt.ylabel("measured [V]")
plt.xlabel("weight [kg]")
plt.legend()
plt.title("Calibration")
plt.show()
