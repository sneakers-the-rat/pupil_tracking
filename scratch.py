from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import filters

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
