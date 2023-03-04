import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.signal as sig
import scipy.integrate as integ
import readndf
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from functools import partial

#region DATA

def make_plate_df(file2, fileL, fileR):
    df_gies = pd.DataFrame()
    df_plate_2 = pd.read_csv(f'data/{file2}', delimiter='\t', usecols=[0, 1, 2, 3],
                             names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])
    df_plate_R = pd.read_csv(f'data/{fileR}', delimiter='\t', usecols=[0, 1, 2, 3],
                             names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])
    df_plate_L = pd.read_csv(f'data/{fileL}', delimiter='\t', usecols=[0, 1, 2, 3],
                             names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

    df_gies['Fy_2'] = df_plate_2.sum(axis=1)
    df_gies['Fy_R'] = df_plate_R.sum(axis=1)
    df_gies['Fy_L'] = df_plate_L.sum(axis=1)

    return df_gies

df_gies = make_plate_df('TN000011.afp2', 'TN000013.afp2', 'TN000012.afp2')
df_marjolein = make_plate_df('TN000016.afp2', 'TN000015.afp2', 'TN000014.afp2')

df_calibration = pd.DataFrame()
df_TN000003 = pd.read_csv('data/TN000003.afp2', delimiter='\t', usecols=[0, 1, 2, 3], names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])
df_calibration['Fy'] = df_TN000003.sum(axis=1)

df_opto_2 = pd.DataFrame(readndf.readndf("data/TN000011.ndf")[2])
df_opto_R = pd.DataFrame(readndf.readndf("data/TN000012.ndf")[2])
df_opto_L = pd.DataFrame(readndf.readndf("data/TN000013.ndf")[2])

#endregion

#region FORCE PLATE

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
    return lambda y: y / a[0]


def subtract_empty_baseline(Fy):
    empty_start = 0
    empty_end = 4500
    return Fy - np.mean(Fy[empty_start:empty_end])


def get_mass(Fy):
    baseline_start = 7000
    baseline_end = 11000
    baseline = np.mean(Fy[baseline_start:baseline_end])
    return regression(baseline)


def calculate_height_plate(Fy, jump_boundaries, height):
    Fy = subtract_empty_baseline(Fy)
    mass = get_mass(Fy)

    Fg = mass * -gravity

    Fy_jump = Fy[jump_boundaries[0]:jump_boundaries[1]]
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
plot_df(df_marjolein, [col])
regression = calibrate()
gravity = 9.81
height = calculate_height_plate(df_marjolein[col], [12000, 15000], 3.0)

#endregion

#region OPTOTRACK
def clip_data(unclipped, high_clip, low_clip):
    ''' Clip unclipped between high_clip and low_clip.
    unclipped contains a single column of unclipped data.'''

    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped)
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_high_clip = (np_unclipped > high_clip) | (np_unclipped < low_clip)
    np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
    return np_clipped.tolist()


# perform moving average in forward and backwards action and calculate the mean of this
def ewma_fb(df_column, span):
    ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
    # Forwards EWMA.
    fwd = pd.Series.ewm(df_column, span=span).mean()
    # Backwards EWMA.
    bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma

def interpolate_advanced(series):
    df = pd.DataFrame()
    # plt.plot(series, label='raw')
    df['y_clipped'] = clip_data(series, 2000, -2000)
    df['y_ewma_fb'] = ewma_fb(df['y_clipped'], 3)
    # df['y_remove_outliers'] = remove_outliers(df['y_clipped'].tolist(), df['y_ewma_fb'].tolist(), 0.1)
    df['y_interpolated'] = df['y_ewma_fb'].interpolate()
    return df['y_interpolated']

column_names = ['hip', 'knee', 'ankle', 'shoulder']
com_offsets = [0.625, 0.6, 0.714]  # shin, thigh, torso
weight_proportions = []

def plot_optotrack(df):
    plt.figure()
    plt.plot(df, label=column_names)
    plt.xlim([0, 12000])
    plt.ylim([-50.0, 2000.0])
    plt.ylabel('marker position [mm]')
    plt.title('Interpolated signal')
    plt.legend()


def com(series):
    # torso, thigh, shin
    com_offsets = [0.714, 0.6, 0.625]
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

    return com

def hip(series):
    return series['hip']

def calculate_heights_optotrack(df, boundaries=[1500, 2500], plot=False, just_hip=False):
    if isinstance(df, str):
        print(f"File {df}")
        df = pd.DataFrame(readndf.readndf(f"data/{df}")[2])
    df.columns = column_names
    df = df.apply(interpolate_advanced)
    func = hip if just_hip else com

    baseline_com = func(df[boundaries[0]:boundaries[1]].mean())
    if plot:
        plot_optotrack(df)

    df['com'] = df.apply(func, axis=1)
    coms = np.array(df['com'] - baseline_com) / 10
    if plot:
        plt.figure()
        plt.plot(coms)

    max_jump = np.nanmax(coms)
    peaks, _ = sig.find_peaks(coms, height=max_jump / 2, distance=100)
    heights = [coms[peak] for peak in peaks]
    print(f"Jumps { heights}")



calculate_heights_optotrack(df_opto_2)
calculate_heights_optotrack(df_opto_2, just_hip=True)
# calculate_heights_optotrack(df_opto_R)
# calculate_heights_optotrack(df_opto_L, boundaries=[2500, 3500])
# calculate_heights_optotrack('TN000024.ndf', boundaries=[1200, 1700], plot=True)

#endregion

#region ANIMATION

def animate_ndf(file, save=False, speed=1):
    ndf = readndf.readndf(f"data/{file}")
    df_z = pd.DataFrame(ndf[2])  # Z
    df_x = pd.DataFrame(ndf[0])  # X
    df_z = df_z.apply(interpolate_advanced)
    df_x = df_x.apply(interpolate_advanced)

    jointz = df_z.to_numpy() / 10
    jointx = df_x.to_numpy() / 10

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
        frames=range(0, len(jointx)),
        init_func=init, blit=True, interval=1000 / 200 / speed)

    plt.show()

    if save:
        writer = PillowWriter(fps=60)
        ani.save('anitmate_jump.gif', writer=writer)

#endregion


plt.show()
