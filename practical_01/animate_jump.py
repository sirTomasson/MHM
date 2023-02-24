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

jointz = df_jointz.to_numpy() / 10
jointx = df_jointx.to_numpy() / 10

# plt.scatter(jointx[0], jointz[0],  label=['hip', 'knee', 'ankle', 'shoulder'])
# plt.xlim([0, 120])
# plt.ylim([50, 200])
# plt.legend()
# plt.show()
# plot_joints(jointx[0], jointz[0])

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'ro')


def init():
    ax.set_xlim(-120, 120)
    ax.set_ylim(0, 250)
    return line1,


def update(frame, ln, x, y):
    ln.set_data(x[frame], y[frame])
    return ln,


ani = FuncAnimation(
    fig, partial(update, ln=line1, x=jointx, y=jointz),
    frames=range(0, len(jointx), 10),
    init_func=init, blit=True)

plt.show()

writer = PillowWriter(fps=30)
ani.save('anitmate_jump.gif', writer=writer)
