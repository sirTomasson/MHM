import matplotlib.pyplot as plt
import numpy as np
import calibrate

real_forces = np.array([0, 10, 20, 30, 40, 50, 60, 86.1, 115.22, 86.1, 60, 50, 40, 30, 20, 10, 0])
df_measured_voltages = calibrate.get_measured_voltages()
measured_voltages = calibrate.df_measured_voltages.to_numpy().reshape(-1)
mean_voltages = calibrate.get_mean_voltages(df_measured_voltages)
voltages = mean_voltages

y = []
x = []
step_size = 30000
current = 0
for i in range(0, len(voltages)):
    y.extend([voltages[i], voltages[i]])
    x.extend([current, current + step_size])
    current += step_size

plt.plot(measured_voltages, label='measured (V)')
plt.plot(x, y, label='mean (V)')
plt.ylim([0, 2.5])
plt.legend()
plt.title("Measured and mean voltages")
plt.show()

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
for i in range(0, len(measured_voltages), step_size):
    start = i + 5000
    sample_start = df_measured_voltages[start:start + 1000].mean().values[0]
    sample_end = df_measured_voltages[start + 20000:start + 21000].mean().values[0]
    diff = sample_start - sample_end
    drift.append(diff)

plt.plot(drift, label='drift [V]')
plt.legend()
plt.title("Drift")
plt.show()

reg = calibrate.calibrate(real_forces, voltages)
w_test = np.arange(0, np.max(real_forces), 1).reshape(-1, 1)
v_pred = reg.predict(w_test)

plt.scatter(real_forces[0:9], voltages[0:9], label="add", marker="x")
plt.scatter(real_forces[8:17], voltages[8:17], label="down", marker="+")
plt.plot(w_test, v_pred, label="predict", linewidth=1)
plt.ylabel("measured [V]")
plt.xlabel("weight [kg]")
plt.legend()
plt.title("Calibration")
plt.show()