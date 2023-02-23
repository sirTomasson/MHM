import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

import calibrate

# Youri's data
df_youri = pd.DataFrame()
df_TN000004 = pd.read_csv('data/TN000007.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TN000005 = pd.read_csv('data/TN000009.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_TN000006 = pd.read_csv('data/TN000010.afp2',
                          delimiter='\t',
                          usecols=[0, 1, 2, 3],
                          names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_youri['2legs [V]'] = df_TN000004.sum(axis=1)
df_youri['L [V]'] = df_TN000005.sum(axis=1)
df_youri['R [V]'] = df_TN000006.sum(axis=1)
df_youri['2legs [V]'] -= df_youri['2legs [V]'][0:1000].mean()
df_youri['L [V]'] -= df_youri['L [V]'][0:1000].mean()
df_youri['R [V]'] -= df_youri['R [V]'][0:1000].mean()

plt.plot(df_youri, label=['2legs [V]', 'L [V]', 'R [V]'])
plt.xlim([0, 50000])
plt.ylim([0, 5.0])
plt.legend()
plt.title("Youri's jumps")
plt.show()

weight_v = df_youri['2legs [V]'][8000:10000].mean()
weight = calibrate.weight(weight_v)
print(f'Youri weighs {weight} kg')

plt.plot(df_youri['2legs [V]'], label='2legs [V]')
plt.xlim([10000, 40000])
plt.ylim([0, 5.0])
plt.legend()
plt.title("Youri's jumps")
plt.show()


def calculate_jump_height(m, jump, label):
    g = 9.81
    Fy = calibrate.weight(jump) * g
    Fg = m * -g
    F = Fy + Fg
    a = F / m
    v = sp.integrate.cumtrapz(a) / 1000
    r = sp.integrate.cumtrapz(v) / 1000

    max = np.max(r)
    plt.plot(r, label=f'jump{label};height = {max:.2f}')
    return max


plt.plot(df_youri['2legs [V]'], label='2legs [V]')
plt.xlim([18000, 22000])
plt.ylim([0, 5.0])
plt.legend()
plt.title("Youri's 3rd jump")
plt.show()

jump1 = df_youri['2legs [V]'][14000:18000].to_numpy().reshape(-1)
jump2 = df_youri['2legs [V]'][19000:21500].to_numpy().reshape(-1)
jump3 = df_youri['2legs [V]'][22000:27000].to_numpy().reshape(-1)
jump4 = df_youri['2legs [V]'][30000:37000].to_numpy().reshape(-1)

jumps = [jump1, jump2, jump3, jump4]
for i in range(0, len(jumps)):
    calculate_jump_height(weight, jumps[i], i)

plt.ylabel('height (m)')
plt.xlabel('time (ms)')
plt.title('jump heights [2 legs]')
plt.legend()
plt.show()

# height_jump1 = calculate_jump_height(weight, jump1, label=1)
# height_jump2 = calculate_jump_height(weight, jump2, label=2)
# height_jump3 = calculate_jump_height(weight, jump3, label=3)
# height_jump4 = calculate_jump_height(weight, jump4, label=4)

