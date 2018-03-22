import pandas as pd
from pandas import ewma
import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance
from collections import deque as dq

def clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list):
    # wrap in dataframe
    params = pd.DataFrame({'x': x_list, 'y': y_list,
                                 'a': a_list, 'b': b_list,
                                 't': t_list, 'v': v_list,
                                 'n': n_list})
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

    return params

def basic_filter(params, ix, iy, rad, e_thresh=0.5, bright=True):
    # remove extremely bad ellipses
    # ix, iy, rad: circle params for largest possible pupil
    # e_thresh: acceptable eccentricity threshold (minor/major)
    # bright: whether the pupil is bright or dark

    # remove ellipses outside some max allowed circle
    in_circle = np.where((params.x-ix)**2+(params.y-iy)**2 < rad**2)[0]
    params = params.loc[in_circle,:]

    # remove extremely oblong ones
    params = params[params.e > e_thresh]

    # and extremely tiny ones
    params = params[params.a > rad/4]

    # threshold based on mean value
    thresh = filters.threshold_otsu(params['v'])
    if bright:
        params = params[params.v > thresh]
    else:
        params = params[params.v < thresh]
    return params

def filter_outliers(params, outlier_params = ('x','y','e','v','n'),
                    neighbors=1000):
    scaler = RobustScaler()
    of = LocalOutlierFactor(n_jobs=7, n_neighbors=neighbors)

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




















def smooth_estimates(params, max_frames=0, hl=20):
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

def prasad_lines(edge):
    # edge should be a list of ordered coordinates
    # all credit to http://ieeexplore.ieee.org/document/6166585/
    # adapted from MATLAB scripts here: https://docs.google.com/open?id=0B10RxHxW3I92dG9SU0pNMV84alk
    # don't expect a lot of commenting from me here,
    # I don't claim to *understand* it, I just transcribed

    x = edge[:,0]
    y = edge[:,1]

    first = 0
    last = len(edge)-1

    seglist = []
    seglist.append([x[0], y[0]])

    D = []
    precision = []
    reliability = []
    sum_dev = []
    D_temp = []


    while first<last:

        mdev_results = prasad_maxlinedev(x[first:last], y[first:last])
        print(mdev_results['d_max'])
        print(mdev_results['del_tol_max'])

        while mdev_results['d_max'] > mdev_results['del_tol_max']:
            print(last)
            last = mdev_results['index_d_max']+first
            print(last)
            mdev_results = prasad_maxlinedev(x[first:last], y[first:last])

        D.append(mdev_results['S_max'])
        seglist.append([x[last], y[last]])
        precision.append(mdev_results['precision'])
        reliability.append(mdev_results['reliability'])
        sum_dev.append(mdev_results['sum_dev'])

        first = last
        last = len(x)-1

    precision_edge = np.mean(precision)
    reliability_edge = np.sum(sum_dev)/np.sum(D)

    return seglist, precision_edge, reliability_edge



def prasad_maxlinedev(x, y):
    # all credit to http://ieeexplore.ieee.org/document/6166585/
    # adapted from MATLAB scripts here: https://docs.google.com/open?id=0B10RxHxW3I92dG9SU0pNMV84alk

    x = x.astype(np.float)
    y = y.astype(np.float)

    results = {}

    first = 0
    last = len(x)-1

    X = np.array([[x[0], y[0]], [x[last], y[last]]])
    A = np.array([
        [(y[0]-y[last]) / (y[0]*x[last] - y[last]*x[0])],
        [(x[0]-x[last]) / (x[0]*y[last] - x[last]*y[0])]
    ])

    if np.isnan(A[0]) and np.isnan(A[1]):
        devmat = np.column_stack((x-x[first], y-y[first])) ** 2
        dev = np.abs(np.sqrt(np.sum(devmat, axis=1)))
    elif np.isinf(A[0]) and np.isinf(A[1]):
        c = x[0]/y[0]
        devmat = np.column_stack((
            x[:]/np.sqrt(1+c**2),
            -c*y[:]/np.sqrt(1+c**2)
        ))
        dev = np.abs(np.sum(devmat, axis=1))
    else:
        devmat = np.column_stack((x, y))
        dev = np.abs(np.matmul(devmat, A)-1.)/np.sqrt(np.sum(A**2))

    results['d_max'] = np.max(dev)
    results['index_d_max'] = np.argmax(dev)
    results['precision'] = np.linalg.norm(dev, ord=2)/np.sqrt(float(last))
    s_mat = np.column_stack((x-x[first], y-y[first])) ** 2
    results['S_max'] = np.max(np.sqrt(np.sum(s_mat, axis=1)))
    results['reliability'] = np.sum(dev)/results['S_max']
    results['sum_dev'] = np.sum(dev)
    results['del_phi_max'] = prasad_digital_error(results['S_max'])
    results['del_tol_max'] = np.tan((results['del_phi_max']*results['S_max']))
    return results

def prasad_digital_error(ss):
    # all credit to http://ieeexplore.ieee.org/document/6166585/
    # adapted from MATLAB scripts here: https://docs.google.com/open?id=0B10RxHxW3I92dG9SU0pNMV84alk

    phii = np.arange(0, np.pi*2, np.pi / 360)

    s, phi = np.meshgrid(ss, phii)

    term1 = []

    term1.append(np.abs(np.cos(phi)))
    term1.append(np.abs(np.sin(phi)))
    term1.append(np.abs(np.sin(phi) + np.cos(phi)))
    term1.append(np.abs(np.sin(phi) - np.cos(phi)))

    term1.append(np.abs(np.cos(phi)))
    term1.append(np.abs(np.sin(phi)))
    term1.append(np.abs(np.sin(phi) + np.cos(phi)))
    term1.append(np.abs(np.sin(phi) - np.cos(phi)))

    tt2 = []
    tt2.append((np.sin(phi))/ s)
    tt2.append((np.cos(phi))/ s)
    tt2.append((np.sin(phi) - np.cos(phi))/ s)
    tt2.append((np.sin(phi) + np.cos(phi))/ s)

    tt2.append(-(np.sin(phi))/ s)
    tt2.append(-(np.cos(phi))/ s)
    tt2.append(-(np.sin(phi) - np.cos(phi))/ s)
    tt2.append(-(np.sin(phi) + np.cos(phi))/ s)


    term2 = []
    term2.append(s* (1 - tt2[0] + tt2[0]**2))
    term2.append(s* (1 - tt2[1] + tt2[1]**2))
    term2.append(s* (1 - tt2[2] + tt2[2]**2))
    term2.append(s* (1 - tt2[3] + tt2[3]**2))

    term2.append(s* (1 - tt2[4] + tt2[4]**2))
    term2.append(s* (1 - tt2[5] + tt2[5]**2))
    term2.append(s* (1 - tt2[6] + tt2[6]**2))
    term2.append(s* (1 - tt2[7] + tt2[7]**2))


    case_value = []
    for c_i in range(8):
        ss = s[:,0]
        t1 = term1[c_i]
        t2 = term2[c_i]
        case_value.append((1/ ss ** 2) * t1 * t2)

    return np.max(case_value)

