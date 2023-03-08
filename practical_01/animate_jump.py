import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import readndf
import clean_optotrack
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from functools import partial


def plot_joints(jointx, jointz, joints=None):
    if joints is None:
        joints = ['hip', 'knee', 'ankle', 'shoulder']
    for x, z, joint in zip(jointx, jointz, joints):
        plt.scatter(x, z, label=joint, marker='+')

    plt.xlim([0, 120])
    plt.ylim([0, 200])
    plt.legend()
    plt.show()


df_jointz = pd.DataFrame(readndf.readndf("data/TN000004.ndf")[2])  # Z
df_jointx = pd.DataFrame(readndf.readndf("data/TN000004.ndf")[0])  # X
df_jointz = df_jointz.apply(clean_optotrack.interpolate_advanced)
df_jointx = df_jointx.apply(clean_optotrack.interpolate_advanced)

df_jointz_temp = pd.DataFrame()
df_jointz_temp['ankle'] = df_jointz.to_numpy()[:, 2]
df_jointz_temp['knee'] = df_jointz.to_numpy()[:, 1]
df_jointz_temp['hip'] = df_jointz.to_numpy()[:, 0]
df_jointz_temp['shoulder'] = df_jointz.to_numpy()[:, 3]

df_jointx_temp = pd.DataFrame()
df_jointx_temp['ankle'] = df_jointx.to_numpy()[:, 2]
df_jointx_temp['knee'] = df_jointx.to_numpy()[:, 1]
df_jointx_temp['hip'] = df_jointx.to_numpy()[:, 0]
df_jointx_temp['shoulder'] = df_jointx.to_numpy()[:, 3]

jointz = df_jointz_temp.to_numpy() / 10
jointx = df_jointx_temp.to_numpy() / 10

# plt.scatter(jointx[0], jointz[0],  label=['hip', 'knee', 'ankle', 'shoulder'])
# plt.xlim([0, 120])
# plt.ylim([50, 200])
# plt.legend()
# plt.show()
# plot_joints(jointx[0], jointz[0])

fig, ax = plt.subplots()
line1, = ax.plot([], [], marker='+')


def init():
    ax.set_xlim(-120, 120)
    ax.set_ylim(0, 250)
    return line1,


def update(frame, ln, x, y):
    ln.set_data(x[frame], y[frame])
    return ln,


ani = FuncAnimation(
    fig, partial(update, ln=line1, x=jointx, y=jointz),
    frames=range(0, len(jointx), 100),
    init_func=init, blit=True)

plt.show()

# writer = PillowWriter(fps=30)
# ani.save('anitmate_jump.gif', writer=writer)
