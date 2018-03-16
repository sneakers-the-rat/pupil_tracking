import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque
from pandas import ewma, ewmstd
from itertools import count
from sklearn.linear_model import LogisticRegression

def crop(im, roi):
    return im[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

def invert_color(im):
    if im.dtype == 'uint8':
        return 255-im
    elif im.dtype == float:
        return 1.-im

def kmeans_img(img, n_clusters, k_criteria):
    img_shape = img.shape
    img = img.reshape(-1, 1)
    img = np.float32(img)
    ret, label, center = cv2.kmeans(img, n_clusters, None, k_criteria, 16, cv2.KMEANS_RANDOM_CENTERS)
    res = center[label]
    res = res.reshape(img_shape)
    return res

def circle_mask(frame, ix, iy, rad):
    # get a boolean mask from circular parameters
    nx, ny = frame.shape
    pmask_x, pmask_y = np.ogrid[-iy:nx - iy, -ix:ny - ix]
    pmask = pmask_x ** 2 + pmask_y ** 2 <= rad ** 2

    return pmask


def infer_sigmoid(img, mask):
    X = img.reshape(-1, 1)
    y = mask.flatten()
    logistic = LogisticRegression()
    logistic.fit(X, y)
    return logistic

def logistic_image(img, logistic):
    # actually sigmoid adjust instead of skimage bs reverse parameterization
    img_flat = img.reshape(-1, 1)
    preds = logistic.predict_proba(img_flat)
    img_mult = np.multiply(img_flat.flatten(), preds[:,1])
    img_mult = img_mult.reshape(img.shape[0], img.shape[1])
    return img_mult


def draw_circle(event,x,y,flags,param):
    # Draw a circle on the frame to outline the pupil
    global ix,iy,drawing,rad,frame_pupil
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            rad = np.round(euclidean([ix,iy], [x,y])).astype(np.int)
            #cv2.circle(frame_pupil,(ix,iy),rad,(255,255,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #cv2.circle(frame_pupil,(ix,iy),rad,(255,255,255),-1)

def edges2xy(edges):
    edges_xy = np.where(edges)
    edges_xy = np.column_stack(edges_xy)

    # flip lr because coords are flipped for images
    return np.fliplr(edges_xy)


def topolar(img, order=5):
    max_radius = 0.5 * np.linalg.norm(img.shape)

    def transform(coords):
        theta = 2.0 * np.pi * coords[1] / (img.shape[1] - 1.)
        radius = max_radius * coords[0] / img.shape[0]
        i = 0.5 * img.shape[0] - radius * np.sin(theta)
        j = radius * np.cos(theta) + 0.5 * img.shape[1]
        return i, j

    polar = geometric_transform(img, transform, order=order, mode='nearest', prefilter=True)


    return polar


def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width),
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img

def preprocess_image(img, roi, gauss_sig=None, logistic=None, sig_cutoff=None, sig_gain=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop(img, roi)
    img = invert_color(img)

    img = exposure.equalize_hist(img)

    if gauss_sig:
        img = filters.gaussian(img, sigma=gauss_sig)

    if logistic:
        img = logistic_image(img, logistic)
    elif sig_cutoff and sig_gain:
        img = exposure.adjust_sigmoid(img, cutoff=sig_cutoff, gain=sig_gain)

    return img


def preprocess_image_old(img, roi, sig_cutoff=0.5, sig_gain=1, n_colors=12,
                     k_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
                     gauss_sig=0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop(img, roi)
    img = invert_color(img)

    img = exposure.equalize_hist(img)

    #img = exposure.adjust_sigmoid(img, cutoff=sig_cutoff, gain=sig_gain)

    # posterize to k colors
    #img = kmeans_img(img, n_colors, k_criteria)

    # blur
    #img = filters.gaussian(img, sigma=gauss_sig)

    return img

def process_edges(edges, x, y, rad):
    true_ellipse = np.array([x,y,rad])

    edges = morphology.label(edges)
    uq_edges = np.unique(edges)
    uq_edges = uq_edges[uq_edges>0]
    ellipses = [fit_ellipse(edges, e) for e in uq_edges]

    if len(ellipses) == 0:
        return False, 0, 0, 0

    # compute mean squared error between pupil & ellipse params
    errors = []
    for e in ellipses:
        try:
            ellipse_param = e.params[0:2]
            ellipse_param.append(np.max(e.params[2:4]))
            ellipse_param = np.array(ellipse_param)
            errors.append(np.mean((true_ellipse-ellipse_param)**2))
        except TypeError:
            # if ellipse couldn't fit, for example
            pass

    min_error_ind = errors.index(np.min(errors))
    min_error_ellipse = ellipses[min_error_ind]
    min_error_points = np.where(edges == min_error_ind+1)
    ret = True
    return ret, min_error_points, min_error_ellipse, edges

def fit_ellipse(edges, which_edge):
    edge_points = np.where(edges == which_edge)
    edge_points = np.column_stack((edge_points[1], edge_points[0]))
    ellipse = measure.EllipseModel()
    ellipse.estimate(edge_points)
    return ellipse

def nothing(x):
    pass