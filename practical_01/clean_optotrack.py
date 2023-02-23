import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import readndf

from scipy.interpolate import interp1d
from scipy.signal import medfilt


# Clips data between range high and low, e.g. -2000, 2000 and fills with np.nan
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


# Possible not necessary
# def remove_outliers(spikey, fbewma, delta):
#     ''' Remove data from df_spikey that is > delta from fbewma. '''
#     np_spikey = np.array(spikey)
#     np_fbewma = np.array(fbewma)
#     cond_delta = (np.abs(np_spikey - np_fbewma) > delta)
#     np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
#     return np_remove_outliers

# Applies median filter over WHOLE signal with window of 3 and performs interpolation
def interpolate_basic(raw, thres=-2000):
    medians = medfilt(raw, 3)
    f = interp1d(raw, medians, kind='linear')
    return f(raw)


# Applies clipping(extreme values) and interpolation using moving weighted average over data
def interpolate_advanced(series):
    df = pd.DataFrame()
    # plt.plot(series, label='raw')
    df['y_clipped'] = clip_data(series, 2000, -2000)
    df['y_ewma_fb'] = ewma_fb(df['y_clipped'], 3)
    # df['y_remove_outliers'] = remove_outliers(df['y_clipped'].tolist(), df['y_ewma_fb'].tolist(), 0.1)
    df['y_interpolated'] = df['y_ewma_fb'].interpolate()
    return df['y_interpolated']


column_names = ['hip', 'knee', 'ankle', 'shoulder']
"""Only load the data of the Z-axis"""
df_MTN000004 = pd.DataFrame(readndf.readndf("data/TN000004.ndf")[2])  # only interested in Z?
df_MTN000004.columns = column_names

plt.plot(df_MTN000004, label=column_names)
plt.xlim([0, 18000])
plt.ylim([-2000.0, 2000.0])
plt.title('Raw signal')
plt.ylabel('marker position [mm]')
plt.legend()
plt.show()

df_MTN000004 = df_MTN000004.apply(interpolate_advanced)
plt.plot(df_MTN000004, label=column_names)
plt.xlim([0, 18000])
plt.ylim([-50.0, 2000.0])
plt.ylabel('marker position [mm]')
plt.title('Interpolated signal')
plt.legend()
plt.show()
