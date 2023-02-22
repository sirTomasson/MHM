import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
plt.plot(df_colyn, label=['Fy_2legs', 'Fy_L', 'Fy_R'])
plt.xlim([0, 50000])
plt.ylim([0, 5.0])
plt.legend()
plt.show()

df_calibration = pd.DataFrame()
df_TN000003 = pd.read_csv('data/TN000003.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_calibration['Fy'] = df_TN000003.sum(axis=1)
plt.plot(df_calibration, label='calibration')
plt.legend()
plt.show()

stepsize = 30000
window_size = 1000
calibration = []
for i in range(0, len(df_calibration), stepsize):
    start = int(i + (stepsize/3))
    df = df_calibration[start:start+10000]
    calibration.append(df.mean().values[0])

plt.plot(calibration, label='calibration')
plt.legend()
plt.show()

real_forces = np.array([0, 10, 20, 30, 40, 50, 60, 86.1, 115.22, 86.1, 60, 50, 40, 30, 20, 10, 0])
calibration = np.array(calibration)
factors = real_forces / calibration

plt.plot(factors, label='factors')
plt.legend()
plt.ylim([40, 50])
plt.show()

stepsize = 30000
window_size = 1000
diffs = []
for i in range(0, len(df_calibration), stepsize):
    start = i + 5000
    sample_start = df_calibration[start:start + 1000].mean().values[0]
    sample_end = df_calibration[start + 20000:start + 21000].mean().values[0]
    diff = sample_start - sample_end
    diffs.append(diff)


plt.plot(diffs, label='diffs')
plt.legend()
plt.show()