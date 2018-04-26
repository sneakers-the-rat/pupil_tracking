import pandas as pd
from pandas import ewma
import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance
from collections import deque as dq

def clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list, c_list, g_list):
    # wrap in dataframe
    params = pd.DataFrame({'x': x_list, 'y': y_list,
                                 'a': a_list, 'b': b_list,
                                 't': t_list, 'v': v_list,
                                 'n': n_list, 'c': c_list,
                                 'g': g_list})
    params = params.astype({'n': np.int})

    # redo coordinates so a is always larger than b, thetas are consistent
    revs = params['b'] > params['a']
    as_temp = params.loc[revs, 'a']
    bs_temp = params.loc[revs, 'b']
    params.loc[revs, 'a'] = bs_temp
    params.loc[revs, 'b'] = as_temp
    ts_temp = params.loc[revs, 't']
    ts_temp = (ts_temp + np.pi / 2) % np.pi
    params.loc[revs, 't'] = ts_temp

    # now make all thetas between 0 and pi
    params['t'] = (params['t'] + np.pi) % np.pi

    # add eccentricity column
    params['e'] = params['b'] / params['a']

    # remove nans
    params.dropna(axis=0, how='any', inplace=True)

    return params

def basic_filter(params, ix, iy, rad, e_thresh=0.5, bright=True):
    # remove extremely bad ellipses
    # ix, iy, rad: circle params for largest possible pupil
    # e_thresh: acceptable eccentricity threshold (minor/major)
    # bright: whether the pupil is bright or dark

    # remove nanas
    #params = params.dropna()

    # remove ellipses outside some max allowed circle
    #in_circle = np.where((params.x-iy)**2+(params.y-ix)**2 < rad**2)[0]
    #params = params.iloc[in_circle,:]
    #
    # # and with negative x or y values
    #params = params[np.logical_and(params.x>0, params.y>0)]
    #
    # # or outside say 5 stdevs
    # for col in ['a', 'b', 'x', 'y']:
    #     col_std = params[col].std()
    #     col_mean = params[col].mean()
    #     params = params[np.logical_and(params[col]<col_mean+col_std*5,
    #                                    params[col]>col_mean-col_std*5)]


    # remove extremely oblong ones
    params = params[params.e > e_thresh]

    # and extremely big/tiny ones
    params = params[params.a > rad/6.]

    # threshold based on mean value
    # DONT THINK THIS WORKS, CUTS A LOTTA GOOD ONES OUT
    # try:
    #     thresh = filters.threshold_otsu(params['v'])
    #     if bright:
    #         params = params[params.v > thresh]
    #     else:
    #         params = params[params.v < thresh]
    # except:
    #     pass


    return params

def filter_outliers(params, outlier_params = ('x','y','e','v','n'),
                    neighbors=1000, outlier_thresh=0.1):
    scaler = RobustScaler()
    of = LocalOutlierFactor(n_jobs=7, n_neighbors=neighbors, metric='minkowski',
                            p=len(outlier_params), contamination=outlier_thresh)

    keys = params.keys()
    if 'n' in keys:
        #keys = [k for k in keys if k != 'n']
        params_scaled = scaler.fit_transform(params.loc[:, keys])
        #params_scaled = np.column_stack((params_scaled, params['n']))
        #keys.append('n')
        params_scaled = pd.DataFrame(params_scaled, columns=keys)
    else:
        params_scaled = scaler.fit_transform(params.loc[:, keys])
        params_scaled = pd.DataFrame(params_scaled, columns=keys)
        params_scaled['n'] = params.index
        params_scaled.set_index(params.index)

    inliers = of.fit_predict(params_scaled.loc[:, outlier_params])

    plot_params(params, inliers)

    # remove and return
    params_filtered = params[inliers == 1]
    return params_filtered




        #if we still have indices, start again at first point but left append
        # if len(inds)>0:
        #     pt_i = first_i
        #
        #     while True:
        #         # get dict of connected points and distances
        #         # filtered by whether the index hasn't been added yet
        #         connected_pts = {k: dists[pt_i, k] for k in np.where(dists[pt_i, :])[0] if k in inds}
        #
        #         # if we get nothing, we're either done or we have to go back to the first pt
        #         if len(connected_pts) == 0:
        #             break
        #
        #         # find point with min distance (take horiz/vert points before diags)
        #         pt_i = min(connected_pts, key=connected_pts.get)
        #         new_pts.appendleft(e_pts[inds.pop(inds.index(pt_i))])



def smooth_estimates(params, max_frames=0, hl=3):
    if 'n' in params.keys():
        params_smooth = params.groupby('n').mean()

        #params_smooth = params.set_index('n')

    else:
        params_smooth = params.copy()

    # filter each parameter column:
    for p in ['a','b','x','y']:
        fwd = params_smooth[p].ewm(halflife=hl).mean()
        bwd = params_smooth[p][::-1].ewm(halflife=hl).mean()
        smoothed = np.mean(np.column_stack((fwd, bwd)), axis=1)
        params_smooth[p] = smoothed

    # reindex and interpolate
    if max_frames != 0:
        frame_inds = np.linspace(0, max_frames, max_frames+1, dtype=np.int)
    else:
        max_frames = params['n'].max()
        frame_inds = np.linspace(0, max_frames, max_frames + 1, dtype=np.int)

    params_smooth = params_smooth.reindex(frame_inds)

    for p in params_smooth.keys():
        params_smooth[p].interpolate(method='cubic', limit_direction='both', inplace=True)

    return params_smooth


def interp_columns(params, max_frames=0):
    if max_frames != 0:
        frame_inds = np.linspace(0, max_frames, max_frames+1, dtype=np.int)
    else:
        try:
            max_frames = params['n'].max()
            frame_inds = np.linspace(0, max_frames, max_frames + 1, dtype=np.int)
        except:
            max_frames = np.max(params.index)
            frame_inds = np.linspace(0, max_frames, max_frames + 1, dtype=np.int)

    params_reind = params.reindex(frame_inds)

    for p in params_reind.keys():
        params_reind[p].interpolate(method='cubic', limit_direction='both', inplace=True)

    return params_reind






def plot_params(params, color='k'):
    keys = params.keys()

    if 'n' in keys:
        fig, ax = plt.subplots(len(keys)-1, 1)
        keys = [k for k in keys if k != 'n']
        for i, x in enumerate(keys):
            ax[i].scatter(params['n'], params[x], s=0.5, alpha=0.2, c=color)
            ax[i].set_ylabel(x)
    else:
        fig, ax = plt.subplots(len(keys), 1)

        for i, x in enumerate(keys):
            ax[i].scatter(params.index, params[x], s=0.5, alpha=0.2, c=color)
            ax[i].set_ylabel(x)


#def