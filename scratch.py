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


import imops
import model
import fitutils

import imops

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
ell_params = fitutils.basic_filter(ell_params, ix, iy, rad)
ell_out = fitutils.filter_outliers(ell_params, neighbors=500)
ell_smooth = fitutils.smooth_estimates(ell_out, hl=5)

vid = cv2.VideoCapture(vid_file)
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

    try:
        aframe_params = ell_smooth.loc[n_frame,['x','y','a','b','t']]
    except:
        frame_orig = imops.crop(frame_orig, roi)
        frame_orig = img_as_float(frame_orig)
        cv2.imshow('run', frame_orig)
        frame_orig = frame_orig * 255
        writer.writeFrame(frame_orig)
        continue


    points = emod.predict_xy(thetas, params=aframe_params)
    points_up = points+1
    points_down = points-1
    points = np.concatenate((points, points_up, points_down), axis=0)

    points = points.astype(np.int)


    frame_orig = imops.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)

    draw.set_color(frame_orig, (points[:, 1], points[:, 0]), (1, 0, 0))
    cv2.imshow('run', frame_orig)

    frame_orig = frame_orig*255
    #frame_orig = frame_orig.astype(np.uint8)

    writer.writeFrame(frame_orig)



writer.close()
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
    edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                      high_threshold=canny_high, low_threshold=canny_low)
    edges_rep = repair_edges(edges_params, frame, sigma=canny_sig)
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(edges_params)
    ax[1].imshow(edges_rep)
    plt.pause(5)



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
        # if n_frame % 10 == 0:
        #     now = time()
        #     fps = n_frame / (now - starttime)
        #     print('frame {} of {}, {} fps'.format(n_frame, total_frames, fps))

        frame = imops.preprocess_image(frame_orig, roi,
                                       sig_cutoff=sig_cutoff,
                                       sig_gain=sig_gain)

        # canny edge detection & reshaping coords
        edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                          high_threshold=canny_high, low_threshold=canny_low)

        edges_params = imops.repair_edges(edges_params, frame)

        try:
            labeled_edges = morphology.label(edges_params)
        except:
            continue
        uq_edges = np.unique(labeled_edges)
        uq_edges = uq_edges[uq_edges > 0]
        ellipses = [imops.fit_ellipse(labeled_edges, e) for e in uq_edges]
        ell_pts = np.ndarray(shape=(0, 2))
        for e in ellipses:
            if not e:
                continue
            points = e.predict_xy(thetas)
            points[points < 0] = 0
            points[points[:, 0] > labeled_edges.shape[1], 0] = labeled_edges.shape[1]
            points[points[:, 1] > labeled_edges.shape[0], 1] = labeled_edges.shape[0]

            ell_pts = np.concatenate((ell_pts, points), axis=0)
            ell_params = e.params
            x_list.append(e.params[0])
            y_list.append(e.params[1])
            a_list.append(e.params[2])
            b_list.append(e.params[3])
            t_list.append(e.params[4])
            n_list.append(n_frame)
            # get mean darkness
            ell_mask_y, ell_mask_x = draw.ellipse(ell_params[0], ell_params[1], ell_params[2], ell_params[3],
                                                  shape=(labeled_edges.shape[1], labeled_edges.shape[0]),
                                                  rotation=ell_params[4])

            v_list.append(np.mean(frame[ell_mask_x, ell_mask_y]))

import cProfile



cProfile.runctx('run_shitty()', None, locals(), filename='/home/lab/stats4.txt')