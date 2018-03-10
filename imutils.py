import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque
from pandas import ewma, ewmstd
from itertools import count

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

def preprocess_image(img, roi, sig_cutoff=0.5, sig_gain=1, n_colors=12,
                     k_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
                     gauss_sig=0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop(img, roi)
    img = invert_color(img)

    img = exposure.equalize_hist(img)

    img = exposure.adjust_sigmoid(img, cutoff=sig_cutoff, gain=sig_gain)

    # posterize to k colors
    #img = kmeans_img(img, n_colors, k_criteria)

    # blur
    img = filters.gaussian(img, sigma=gauss_sig)

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