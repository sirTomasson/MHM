import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.signal as sig
import scipy.integrate as integ
import readndf

#region DATA

df_gies = pd.DataFrame()
df_plate_2 = pd.read_csv('data/TN000011.afp2', delimiter='\t', usecols=[0, 1, 2, 3], names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])
df_plate_R = pd.read_csv('data/TN000012.afp2', delimiter='\t', usecols=[0, 1, 2, 3], names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])
df_plate_L = pd.read_csv('data/TN000013.afp2', delimiter='\t', usecols=[0, 1, 2, 3], names=['Fy0', 'Fy1', 'Fy2', 'Fy3'])

df_gies['Fy_2'] = df_plate_2.sum(axis=1)
df_gies['Fy_R'] = df_plate_R.sum(axis=1)
df_gies['Fy_L'] = df_plate_L.sum(axis=1)

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
    empty_end = 3000
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

def calculate_heights_optotrack(df, boundaries=[1500, 2500], plot=False):
    df.columns = column_names
    df = df.apply(interpolate_advanced)
    baseline_com = com(df[boundaries[0]:boundaries[1]].mean())
    if plot:
        plot_optotrack(df)

    df['com'] = df.apply(com, axis=1)
    coms = np.array(df['com'] - baseline_com) / 10
    if plot:
        plt.figure()
        plt.plot(coms)

    max_jump = np.nanmax(coms)
    peaks, _ = sig.find_peaks(coms, height=max_jump / 2, distance=100)
    heights = [coms[peak] for peak in peaks]
    print(f"Jumps { heights}")


calculate_heights_optotrack(df_opto_2)
calculate_heights_optotrack(df_opto_R)
calculate_heights_optotrack(df_opto_L, boundaries=[2500, 3500])

#endregion

plt.show()
