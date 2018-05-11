from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import filters
from time import time
from skvideo import io

import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import feature, morphology, img_as_float, draw
from itertools import count
from time import time

import cProfile
import imops
import model
import fitutils

import imops

import multiprocessing as mp

def crop(im, roi):
    return im[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]


vid_file = '/home/lab/pupil_vids/ira1.mp4'
vid = cv2.VideoCapture(vid_file)
nframes = vid.get(cv2.CAP_PROP_FRAME_COUNT)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = crop(frame, roi)
frame.shape

vol = np.ndarray(shape=(frame.shape[0], frame.shape[1], nframes), dtype=np.uint8)

vid = cv2.VideoCapture(vid_file)
for i in range(int(nframes)):
    if i % 1000 == 0:
        print("frame {} of {}".format(i, nframes))
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = crop(frame, roi)
    #frame = exposure.equalize_hist(frame)
    vol[:,:,i] = frame

vol_orig = vol.copy()

frame_filter = frame.copy()
low = 0.0; high = 1.0
for i in range(10):
    frame_filter = filters.rank.enhance_contrast_percentile(frame_filter, morphology.disk(5), p0=low, p1=high)
    low, high = low+.05, high-.05
    ax[0].clear()
    ax[0].imshow(frame_filter)
    plt.pause(.001)


vol = vol_orig.copy()
vol = vol[:,:,0:5000]
vol = img_as_float(vol)
vol = 1.-vol

vol = logistic_image(vol, this_log)
vol_sob = ndimage.generic_gradient_magnitude(vol, ndimage.sobel)
vol_sob = (vol_sob-np.min(vol_sob.flatten()))/(np.max(vol_sob.flatten())-np.min(vol_sob.flatten()))

params = np.ndarray(shape=(5,5000))
params[0,:] = ix
params[1,:] = iy
params[2,:] = rad
params[3,:] = rad
params[4,:] = 0

ellipse_mask = np.ndarray(shape=vol_sob.shape, dtype=np.bool)
ypts = np.linspace(0,vol_sob.shape[0]-1,vol_sob.shape[0])
xpts = np.linspace(0,vol_sob.shape[1]-1,vol_sob.shape[1])

ypts, xpts = np.meshgrid(range(vol_sob.shape[1]),range(vol_sob.shape[0]))
ypts, xpts = ypts.astype(np.float), xpts.astype(np.float)

def ellipse_pts(param_vect):
    ix, iy, a, b, t = param_vect

    shape_y = 265
    shape_x = 145

    ypts, xpts = np.meshgrid(range(shape_y), range(shape_x))
    ypts, xpts = ypts.astype(np.float), xpts.astype(np.float)

    athet = np.cos(t)
    bthet = np.sin(t)

    xpts = xpts*athet - ypts*bthet
    ypts = xpts*bthet + ypts*athet


    ellipse_pts = np.logical_and((((xpts-iy)/b)**2 + ((ypts-ix)/a)**2)<=1.02,
                                 (((xpts - iy) / b) ** 2 + ((ypts - ix) / a) ** 2) >= .98)
    return ellipse_pts

el_pts = ellipse_pts(ix, iy, rad, rad, 0., vol[:,:,0].shape)

ax[0].imshow(ellipse_pts)
ax[0].imshow(vol[:,:,0])

ellipse_mask = np.ndarray(shape=vol_sob.shape, dtype=np.bool)

def norm(arr):
    return (arr-np.min(arr.flatten()))/(np.max(arr.flatten())-np.min(arr.flatten()))

params = np.ndarray(shape=(5,5000))
params[0,:] = ix
params[1,:] = iy
params[2,:] = rad
params[3,:] = rad
params[4,:] = 0


def pupil_tube(params, vol_sob):
    params = params.reshape(5,5000)
    dot = np.ndarray(shape=(5000,))
    for i in xrange(params.shape[1]):
        ellipse_mask = ellipse_pts(params[:,i])
        dot[i] = np.mean((1.-vol_sob[ellipse_mask,i])**2)


    param_diff = np.diff(params, axis=1)**2
    param_diff = norm(param_diff)+1.
    param_diff = np.mean(param_diff, axis=0)

    dot[1:] = dot[1:]*param_diff

    oblongitude = params[2,:]/params[3,:]
    oblongitude[oblongitude>1] = 1./oblongitude[oblongitude>1]
    oblongitude = oblongitude ** 2
    oblongitude = 1.-oblongitude

    dot = dot * oblongitude

    dot[np.isnan(dot)] = 0.

    return dot

from itertools import count
acount = count()
def callback():
    print(acount.next())


params = params.flatten()
from scipy.optimize import least_squares
amin = least_squares(pupil_tube, params, verbose=2, args=(vol_sob,))


    # dot prod
    im_dot = (1.-vol_sob[ellipse_mask])**2
    im_dot = np.mean((1.-vol_sob[ellipse_mask])**2, axis=2)
    im_dot = np.multiply(ellipse_mask.flatten(), vol_sob.flatten()).reshape(ellipse_mask.shape)
    im_dot = (1.-im_dot)**2











fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
grid_x, grid_y = np.meshgrid(range(vol.shape[1]), range(vol.shape[0]))
for i in range(100):
    ax.clear()
    ax.plot_surface(grid_x, grid_y, vol[:,:,i])
    plt.pause(0.01)


fig, ax = plt.subplots(2,1)
ax[0].imshow(vol[:,:,500])

ax[1].clear()
ax[1].hist(vol[:,:,100].flatten(), bins=30)

def infer_sigmoid(img, mask):
    X = img.reshape(-1, 1)
    y = mask.flatten()
    logistic = LogisticRegression()
    logistic.fit(X, y)
    return logistic

def logistic_image(img, logistic):
    # actually sigmoid adjust instead of skimage bs reverse parameterization
    preds = logistic.predict_proba(img.reshape(-1, 1))
    img_mult = np.multiply(img.flatten(), preds[:,1])
    img_mult = img_mult.reshape(img.shape)
    return img_mult


#############
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = feature.ORB(frame1,None)
kp2, des2 = sift.detectAndCompute(frame2,None)

#################
def ng_structure_tensor(frame):
    kernel = np.array([[-3,0,3],[-10,0,10],[-3,0,3]],dtype=np.float)/32.


Axx, Axy, Ayy = feature.structure_tensor(frame_sig, sigma=100)
l1, l2 = feature.structure_tensor_eigvals(Axx, Axy, Ayy)
l1_arr = np.array(l1)
l2_arr = np.array(l2)

#fig, ax = plt.subplots(3,2)
ax[0,0].imshow(frame_sig)
ax[1,0].imshow(l1_arr)
ax[2,0].imshow(l2_arr)
ax[0,1].imshow(Axx)
ax[1,1].imshow(Axy)
ax[2,1].imshow(Ayy)

#### Filtering ellipses
import pandas as pd

# from a list...
frame_params = pd.DataFrame({'x':x_list, 'y':y_list,
                             'a':a_list, 'b':b_list,
                             't':t_list, 'v':v_list,
                             'n':n_list})
frame_params = frame_params.astype({'n':np.int})



frame_params_bak = frame_params.copy()

#frame_params = frame_params_bak.copy()
#frame_params = pd.DataFrame(frame_params_bak, columns=['x','y','a','b','t','n'])

# given frame-params..

#frame_params = frame_params.astype({'x':np.int, 'y':np.int,
#                                    'a':np.int, 'b':np.int,
#                                    't':np.float,'n':np.int})

# redo coordinates so a is always larger than b, thetas are consistent
revs = frame_params['b'] > frame_params['a']
as_temp = frame_params.loc[revs,'a']
bs_temp = frame_params.loc[revs,'b']
frame_params.loc[revs,'a'] = bs_temp
frame_params.loc[revs,'b'] = as_temp
ts_temp = frame_params.loc[revs,'t']
ts_temp = (ts_temp + np.pi/2) % np.pi
frame_params.loc[revs,'t'] = ts_temp

# now make all thetas between 0 and pi
frame_params['t'] = (frame_params['t']+np.pi) % np.pi

# remove extreme x/y positions

z = np.where((frame_params['x']-ix)**2+(frame_params['y']-iy)**2 < rad**2)
z = z[0]
frame_params = frame_params.loc[z,:]

# remove extremely oblong ones
frame_params['e'] = frame_params['b']/frame_params['a']
frame_params = frame_params[frame_params.e>0.5]

# set n as index
#frame_params = frame_params.set_index('n')

# threshold based on mean value
thresh = filters.threshold_otsu(frame_params['v'])
frame_params = frame_params[frame_params.v > thresh]

#

fig, ax = plt.subplots(7,1)
for i, x in enumerate(['x','y','a','b','t', 'v', 'e']):
    ax[i].scatter(frame_params['n'], frame_params[x], s=0.5, alpha=0.2, c='k')

from sklearn.decomposition import FastICA
from scipy import signal

# Generate sample data
ica = FastICA()
S_ = ica.fit_transform(frame_params.loc[:,('x','y')])

fig, ax = plt.subplots(S_.shape[1],1)
for i in range(S_.shape[1]):
    ax[i].scatter(range(S_.shape[0]),S_[:,i], s=0.5)

# trace through and segregate
frame_params_ind = frame_params.set_index('n')


############# trying clustering
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

scaler = RobustScaler()
params_scaled = scaler.fit_transform(frame_params.loc[:,('a','b','x','y','v','e')])
params_scaled = np.column_stack((params_scaled, frame_params['n']))
params_scaled = pd.DataFrame(params_scaled, columns=['a','b','x','y','v','e','n'])

of = LocalOutlierFactor(n_jobs=7)
inliers = of.fit_predict(frame_params.loc[:,('x','y','e','v','n')])


#svr = SVR(verbose=True)
#svr.fit(frame_params['n'].reshape(-1,1), frame_params['a'])
#regress = RANSACRegressor()
#regress.fit(frame_params.loc[:,('n')].reshape(-1,1), frame_params['a'])

#grid_params = {'n_clusters':[2,3,4,5]}

#km= cluster.KMeans(n_jobs=7, copy_x=True)
#grids = GridSearchCV(km, param_grid=grid_params)
#grids.fit(frame_params.loc[:,('x','y','v','e')])

#grid_results = pd.DataFrame(grids.cv_results_)

#km = cluster.KMeans(n_jobs=7, n_clusters=3)
#clusts = km.fit_predict(frame_params.loc[:,('x','y','v','e')])




fig, ax = plt.subplots(7,1)
for i, x in enumerate(['x','y','a','b','t', 'v', 'e']):
    ax[i].scatter(frame_params['n'], frame_params[x], s=0.5, alpha=0.2, c=inliers)

params_filtered = frame_params[inliers==1]

fig, ax = plt.subplots(7,1)
for i, x in enumerate(['x','y','a','b','t', 'v', 'e']):
    ax[i].scatter(params_filtered['n'], params_filtered[x], s=0.5, alpha=0.2, c='k')

#params_filtered = params_filtered.set_index('n')
params_mean = params_filtered.groupby('n').mean()
#params_mean

fig, ax = plt.subplots(7,1)
for i, x in enumerate(['x','y','a','b','t', 'v', 'e']):
    ax[i].scatter(params_mean['n'], params_mean[x], s=0.5, alpha=0.2, c='k')

index = params_mean.index
index = np.linspace(0, frame_params['n'].max(), frame_params['n'].max()+1, dtype=np.int)
params_mean.reindex(index)
params_mean.interpolate('cubic')

fig, ax = plt.subplots(7,1)
for i, x in enumerate(['x','y','a','b','t', 'v', 'e']):
    ax[i].scatter(params_mean[x], s=0.5, alpha=0.2, c='k')

##########################

#fig, ax = plt.subplots(4,1)
thetas = np.linspace(0, np.pi*2, num=200, endpoint=False)

emod = measure.EllipseModel()

fn = '/home/lab/pupil_vids/nick3_track2.mp4'
#fourcc = cv2.VideoWriter_fourcc(*"X264")
writer = io.FFmpegWriter(fn)

ell_params = fitutils.clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list)
ell_params = basic_filter(ell_params, ix, iy, rad, e_thresh=0.6)
ell_out = fitutils.filter_outliers(df_fill, neighbors=1000)
ell_smooth = fitutils.smooth_estimates(ell_out, hl=5)

vid = cv2.VideoCapture(vid_fn)
#vid.set(cv2.CAP_PROP_POS_MSEC, 350000)
frame_counter = count()
cv2.namedWindow('run', flags=cv2.WINDOW_NORMAL)
for i in xrange(len(ell_smooth)):
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()
    if ret == False:
        break
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)


    n_frame = frame_counter.next()

    # try:
    #     aframe_params = ell_smooth.loc[n_frame,['x','y','a','b','t']]
    # except:
    #     frame_orig = imops.crop(frame_orig, roi)
    #     frame_orig = img_as_float(frame_orig)
    #     cv2.imshow('run', frame_orig)
    #     frame_orig = frame_orig * 255
    #     writer.writeFrame(frame_orig)
    #     continue

    # e_points = []
    # for i, p in ell_smooth[ell_smooth.n==n_frame].iterrows():
    #     points = emod.predict_xy(thetas, params=(p.x, p.y, p.a, p.b, p.t))
    #     e_points.append(points.astype(np.int))
    #     #points_up = points+1
    #     #points_down = points-1
    #     #points = np.concatenate((points, points_up, points_down), axis=0)
    # try:
    #     e_points = np.row_stack(e_points)
    # except ValueError:
    #     continue
    p = ell_smooth.iloc[n_frame]
    e_points = emod.predict_xy(thetas, params=(p.x, p.y, p.a, p.b, p.t))
    e_points = e_points.astype(np.int)





    frame_orig = imops.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)

    draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1]), (1, 0, 0))
    cv2.imshow('run', frame_orig)

    #frame_orig = frame_orig*255
    #frame_orig = frame_orig.astype(np.uint8)

    #writer.writeFrame(frame_orig)



#writer.close()
cv2.destroyAllWindows()

ret, frame_orig = vid.read()
frame = imops.preprocess_image(frame_orig, roi,
                               sig_cutoff=sig_cutoff,
                               sig_gain=sig_gain)
edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                  high_threshold=canny_high, low_threshold=canny_low)
ax.imshow(edges_params)


from PIL import ImageFilter, Image
frame_8 = frame.copy()
frame_8 = filters.gaussian(frame_8, sigma=2)
frame_8 = frame_8 * 255
frame_8 = frame_8.astype(np.uint8)
im = Image.fromarray(frame_8)
im1 = im.filter(ImageFilter.EDGE_ENHANCE)
im.show()
im1.show()

##################
fig, ax = plt.subplots(1,2)
for i in range(100):
    ret, frame_orig = vid.read()
    frame = imops.preprocess_image(frame_orig, roi,
                                   sig_cutoff=sig_cutoff,
                                   sig_gain=sig_gain)
    edges_params = scharr_canny(frame, sigma=canny_sig,
                                      high_threshold=canny_high, low_threshold=canny_low)
    edges_rep = repair_edges(edges_params, frame, sigma=canny_sig)
    edges_rep_img = np.zeros(edges_params.shape, dtype=np.int)
    for i, e in enumerate(edges_rep):
        edges_rep_img[e[:,0], e[:,1]] = i
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(edges_params)
    ax[1].imshow(edges_rep_img)
    plt.pause(1)



def run_shitty():
    vid_file = '/home/lab/pupil_vids/nick3.avi'
    # run once we get good params
    # remake video object to restart from frame 0
    vid = cv2.VideoCapture(vid_file)
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    roi = (725, 529, 523, 334)
    sig_cutoff = 0.7
    sig_gain = 10
    canny_sig = 4.07
    canny_high = 1.17
    canny_low = 0.3
    # pmod = model.Pupil_Model(ix, iy, rad)

    # cv2.namedWindow('run', flags=cv2.WINDOW_NORMAL)
    # fig, ax = plt.subplots(4,1)
    thetas = np.linspace(0, np.pi * 2, num=100, endpoint=False)
    frame_counter = count()

    # frame_params = np.ndarray(shape=(0, 7))
    x_list = []
    y_list = []
    a_list = []
    b_list = []
    t_list = []
    n_list = []
    v_list = []

    starttime = time()
    for i in range(100):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break

        ret, frame_orig = vid.read()
        if ret == False:
            break
        n_frame = frame_counter.next()
        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame_orig = imops.crop(frame_orig, roi)

        frame = imops.preprocess_image(frame_orig, params['roi'],
                                       sig_cutoff=params['sig_cutoff'],
                                       sig_gain=params['sig_gain'])

        edges = scharr_canny(frame, sigma=params['canny_sig'],
                                   high_threshold=params['canny_high'], low_threshold=params['canny_low'])

        edges_rep = repair_edges(edges, frame)

        ellipses = [imops.fit_ellipse(e) for e in edges_rep]
        # ell_pts = np.ndarray(shape=(0, 2))
        for e in ellipses:
            if not e:
                continue

            x_list.append(e.params[0])
            y_list.append(e.params[1])
            a_list.append(e.params[2])
            b_list.append(e.params[3])
            t_list.append(e.params[4])
            n_list.append(n_frame)
            # get mean darkness
            ell_mask_y, ell_mask_x = draw.ellipse(e.params[0], e.params[1], e.params[2], e.params[3],
                                                  shape=(frame.shape[1], frame.shape[0]),
                                                  rotation=e.params[4])

            v_list.append(np.mean(frame[ell_mask_x, ell_mask_y]))


import cProfile
edges_lots = []
frames = []
for i in range(200):
    ret, frame_orig = vid.read()
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_orig = imops.crop(frame_orig, roi)

    frame = imops.preprocess_image(frame_orig, params['roi'],
                                   sig_cutoff=params['sig_cutoff'],
                                   sig_gain=params['sig_gain'])
    frames.append(frame)

    # edges = imops.scharr_canny(frame, sigma=params['canny_sig'],
    #                            high_threshold=params['canny_high'], low_threshold=params['canny_low'])
    #
    # edges_lots.extend(repair_edges(edges, frame))

def order_many_edges():
    for e in edges_lots:
        order_points(e)

def scharr_many():
    for frame in frames:
        edges = scharr_canny(frame, sigma=params['canny_sig'],
                             high_threshold=params['canny_high'], low_threshold=params['canny_low'])




cProfile.runctx('run_shitty()', None, locals(), filename='/home/lab/stats6.txt')


####################
# timing scharr functions
cProfile.runctx('scharr_canny_old(frame, sigma, low_threshold=0.25, high_threshold=1.)', None, locals(), filename='/home/lab/scharr_old.txt')

import timeit
def run_new_shitty():
    vid_file = '/home/lab/pupil_vids/nick3.avi'
    vid = cv2.VideoCapture(vid_file)
    roi = (725, 529, 523, 334)
    sig_cutoff = 0.7
    sig_gain = 10
    canny_sig = 4.07
    canny_high = 1.17
    canny_low = 0.3

    for i in range(200):
        ret, frame_orig = vid.read()

        frame = imops.preprocess_image(frame_orig, roi,
                                       sig_cutoff=sig_cutoff,
                                       sig_gain=sig_gain)

        # canny edge detection & reshaping coords
        edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                          high_threshold=canny_high, low_threshold=canny_low)


cProfile.runctx('run_new_shitty()', None, locals(), filename='/home/lab/scharr_new2.txt')


########
one_col = np.ones((dists.shape[0],1))
first_col = dists[:,0].reshape(-1, 1)

gram = -0.5*(dists-np.matmul(one_col, first_col.T)-np.matmul(first_col, one_col.T))


def order_points(edge_points):
    dists = distance.squareform(distance.pdist(edge_points))

    inds = {i:i for i in range(len(edge_points))}

    backwards=False
    point = 0
    new_points = dq()
    new_points.append(edge_points[inds.pop(point),:])

    while True:
        close_enough = np.where(np.logical_and(dists[point,:]>0, dists[point,:]<3))[0]
        close_enough = close_enough[np.in1d(close_enough, inds.keys())]
        try:
            point = close_enough[np.argmin(dists[point,close_enough])]
        except ValueError:
            # either at one end or *the end*
            if not backwards:
                point = 0
                backwards = True
                continue
            else:
                break

        if not backwards:
            new_points.append(edge_points[inds.pop(point),:])
        else:
            new_points.appendleft(edge_points[inds.pop(point),:])

    return np.row_stack(new_points)



######################

img_close = img.copy()
for i, s in enumerate(closing_sizes):
    img_close_temp = morphology.closing(img_close, selem=morphology.disk(s))
    ax2.flatten()[i].imshow(img_close_temp)

#######################

from sklearn import mixture
import itertools

dpgmm = mixture.BayesianGaussianMixture(
    n_components=3, covariance_type='full', weight_concentration_prior=1e-2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(8),
    init_params="random", max_iter=100, random_state=2)
dpgmm.fit(ell_df_out)

def plot_samples(X, Y, n_components, index, title):
    fig, ax = plt.subplots(8,1)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        for j in range(8):
            ax[j].scatter(X[Y == i, 2], X[Y == i, j], .8, color=color)

    ax.xlim(-6., 4. * np.pi - 6.)
    ax.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

y_s = dpgmm.predict(ell_df_out)
def plot_params(params, labels):
    keys = params.keys()

    if 'n' in keys:
        fig, ax = plt.subplots(len(keys)-1, 1)
        keys = [k for k in keys if k != 'n']
        for i, x in enumerate(keys):
            ax[i].scatter(params['n'], params[x], s=0.5, alpha=0.2, c=labels)
            ax[i].set_ylabel(x)
    else:
        fig, ax = plt.subplots(len(keys), 1)

        for i, x in enumerate(keys):
            ax[i].scatter(params.index, params[x], s=0.5, alpha=0.2, c=labels)
            ax[i].set_ylabel(x)

edges_xy = [edge[['x','y']].get_values() for i, edge in frame_df.groupby('edge')['x','y']]
grads = [edge[['grad_x','grad_y']].get_values() for i, edge in frame_df.groupby('edge')]
angles = [edge[['angle']].get_values() for i, edge in frame_df.groupby('edge')]
def measure_affinity(edges_xy, grads):
    # given a dataframe with a multiindex, edge, points with x, y, grads, and angles

    # get edge points
    edge_points = np.row_stack([(e[0], e[-1]) for e in edges_xy])
    edge_grads = np.row_stack([(e[0], e[-1]) for e in grads])

    # get the  euclidean distances between points as a fraction of image size
    dist_metric = distance.squareform(distance.pdist(edge_points))
    grad_metric = distance.squareform(distance.pdist(edge_grads, "cosine"))

    ## 3. edge affinity
    # want edges that point at each other

    # get direction of edge points from line segments
    segs = [prasad_lines(e) for e in edges_xy]

    # 3d array of the edge points and the neighboring segment vertex
    edge_segs = np.row_stack([((e[0],e[1]),(e[-1],e[-2])) for e in segs])

    # subtract neighboring segment vertex from edge point to get vector pointing
    # in direction of edge
    # we also get the midpoints of the last segment,
    # because pointing towards the actual last point in the segment can be noisy
    edge_vects = normalize(edge_segs[:,0,:] - edge_segs[:,1,:])
    edge_mids = np.mean(edge_segs, axis=1)

    # cosine distance between direction of edge at endpoints and direction to other points
    # note this distance is asymmetrical, affinity from point in row to point in column
    affinity_metric = np.row_stack(
        [distance.cdist(edge_vects[i,:].reshape(1,2), edge_mids-edge_points[i,:], "cosine")
         for i in range(len(edge_segs))]
    )

    # get upper/lower indices manually so they match up right
    triu_indices = np.triu_indices(affinity_metric.shape[0], k= 1)
    tril_indices = triu_indices[::-1]

    # average top and bottom triangles - both edges should point at each other
    pair_affinity = np.mean((affinity_metric[tril_indices],
                             affinity_metric[triu_indices]), axis=0)
    affinity_metric[tril_indices] = pair_affinity
    affinity_metric[triu_indices] = pair_affinity

    # Clean up & combine merge metrics
    # want high values to be good, and for vals to be 0-1
    dist_metric = 1-((dist_metric-np.min(dist_metric))/(np.max(dist_metric)-np.min(dist_metric)))
    #dist_metric = 1-dist_metric
    grad_metric = 1-(grad_metric/2.) # cosine distance is 0-2
    affinity_metric = 1-(affinity_metric/2.)

    merge_score = dist_metric**2 * grad_metric**2 * affinity_metric**2
    merge_score[np.isnan(merge_score)] = 0.
    merge_score[triu_indices] = 0

    return merge_score

from itertools import combinations
def collapse_unique(angles):
    # get ends of angles
    edge_angles = np.row_stack([np.column_stack((np.min(e), np.max(e), np.median(e))) for e in angles])
    # find the ones that wrap around zero
    rev_inds = edge_angles[:,1]-edge_angles[:,0]>5.
    edge_angles[rev_inds,0:2] = np.fliplr(edge_angles[rev_inds,0:2])
    combo_mask = np.zeros((len(angles), len(angles)), dtype=np.bool)

    for edge_1, edge_2 in combinations(enumerate(edge_angles), 2):
        ind_1, angles_1 = edge_1
        ind_2, angles_2 = edge_2

        print(angles_1, angles_2)
        # check if overlap
        if angles_1[0] < angles_1[1]:
            np.logical_and(angles_1[0] >= angles_2, angles_2 >= angles_1[1])



#########################
import pandas as pd
import numpy as np
import json
import os
import runops

param_fn = '/home/lab/pupellipse/params/eyetrig_18-05-01-1652.json'
ell_df = pd.read_csv('/home/lab/pupellipse/data/EyeTrig.csv')
ell_df['gn'] = (ell_df.g-ell_df.g.min())/(ell_df.g.max()-ell_df.g.min())
ell_df['cn'] = (ell_df.c-ell_df.c.min())/(ell_df.c.max()-ell_df.c.min())
ell_df['en'] = (ell_df.e-ell_df.e.min())/(ell_df.e.max()-ell_df.e.min())
ell_frame = ell_df.groupby("n")
idxes = []
for name, group in ell_frame:
    # TODO: once V is fixed, also add variance, and then use V adjusted by area - a high V over a large area is better than a high V over a small area

    good_vals = group.gn * group.cn * group.en
    idxes.append(group.loc[good_vals.idxmax()][0])

only_good = ell_df.loc[np.isin(ell_df['Unnamed: 0'], idxes)]


########################3
# multiprocessing tests

from tqdm import trange
from itertools import count

def worker(input, output, params):
    for frame, n in iter(input.get, 'END'):
        result = runops.process_frame_all(frame, params, n)
        output.put(result)

def test(params, n_proc=7):
    vid = cv2.VideoCapture(params['files'][0])
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()


    for i in range(n_proc):
        mp.Process(target=worker, args=(task_queue, done_queue, params)).start()

    for i in trange(100):
        ret, frame = vid.read()
        n_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

        task_queue.put((frame, n_frame))

    results = []
    for i in range(100):
        results.append(done_queue.get())