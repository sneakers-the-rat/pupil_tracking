import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque as dq
from pandas import ewma, ewmstd
from itertools import count
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import scipy.ndimage as ndi
from scipy.ndimage import (gaussian_filter,
                           generate_binary_structure, binary_erosion, label)


# http://cdn.intechopen.com/pdfs/33559/InTech-Methods_for_ellipse_detection_from_edge_maps_of_real_images.pdf

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
    ret = ellipse.estimate(edge_points)
    if ret == True:
        #resid = np.mean(ellipse.residuals(edge_points)**2)
        #return ellipse, resid
        return ellipse

def nothing(x):
    pass

def edge_vectors(frame, sigma):
    """
    Get the eigenvectors/values of the structural tensor
    (1) https://arxiv.org/pdf/1402.5564.pdf
    (2) https://hal.archives-ouvertes.fr/hal-01037972/document

    The structure tensor A is defined as::
        A = [Axx Axy]
            [Axy Ayy]

    Where Axx is the array of squared gradients in the x direction,
    Axy is gradient in x * y for each pixel after convolution with gaussian kernel.

    We use the Scharr kernel to compute A for greater rotational invariance
    https://ac.els-cdn.com/S104732030190495X/1-s2.0-S104732030190495X-main.pdf?_tid=61aa5911-ea13-4939-8792-2ff929704990&acdnat=1521241976_f795a9cf5fde71276ea047f6a226de4c


    """
    #frame = np.flipud(frame)

    #grad_x = filters.sobel_h(frame)
    #grad_y = filters.sobel_v(frame)

    grad_x = filters.gaussian(cv2.Scharr(frame, ddepth=-1, dx=1, dy=0), sigma=sigma)
    grad_y = filters.gaussian(cv2.Scharr(frame, ddepth=-1, dx=0, dy=1), sigma=sigma)

    # Eigenvalues
    Axx = grad_x*grad_x
    Axy = grad_x*grad_y
    Ayy = grad_y*grad_y

    e1 = 0.5 * (Ayy + Axx - np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))
    e2 = 0.5 * (Ayy + Axx + np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))

    # norm the vectors
    grads = np.stack((grad_x, grad_y), axis=-1)
    grads = normalize(grads.reshape(-1,2), norm="l2", axis=1).reshape(grads.shape)
    grad_x, grad_y = grads[:,:,0], grads[:,:,1]

    # get angles 0-2pi
    angle = np.arccos(grad_x)
    angle[grad_y<=0] = np.arccos(-grad_x[grad_y<=0])+np.pi

    # edges have eigenvalues with high e2 and low e1, so
    edge_scale = e2-e1

    return grad_x, grad_y, angle, edge_scale

def scharr_canny(image, sigma, low_threshold, high_threshold):
    # skimage's canny but we get scharr grads instead of sobel,
    # and use the eigenvalues of the structure tensor rather than the hypotenuse

    isobel = filters.gaussian(cv2.Scharr(image, ddepth=-1, dx=0, dy=1), sigma=sigma)
    jsobel = filters.gaussian(cv2.Scharr(image, ddepth=-1, dx=1, dy=0), sigma=sigma)
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)

    Axx = jsobel*jsobel
    Axy = jsobel*isobel
    Ayy = isobel*isobel

    e1 = 0.5 * (Ayy + Axx - np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))
    e2 = 0.5 * (Ayy + Axx + np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))

    magnitude = exposure.adjust_sigmoid(e2-e1, cutoff=0.5, gain=2)

    # magnitude = np.hypot(isobel, jsobel)
    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    mask = np.ones(image.shape)
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    #----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus


    #
    #---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask


def repair_edges(edges, im_grad):
    # connect contiguous edges, disconnect edges w/ sharp angles
    # expect a binary edge image, like from feature.canny
    # im_grad should be complex (array of vectors)

    label_edges = morphology.label(edges)

    uq_edges = np.unique(label_edges)
    uq_edges = uq_edges[uq_edges>0]

    # first, go through and break edges at corners
    for e in uq_edges:



def order_points(edge_points):
    # convert edge masks to ordered x-y coords

    if isinstance(edge_points, list):
        edge_points = np.array(edge_points)

    if edge_points.shape[1] > 2:
        # starting w/ imagelike thing, get the points
        edge_points = np.column_stack(np.where(edge_points))

    dists = distance.squareform(distance.pdist(edge_points))
    # make binary connectedness graph, max dist is ~1.4 for diagonal pix
    # convert to 3 and 2 so singly-connected points are always <4
    dists[dists > 1.5] = 0
    dists[dists >1]    = 3
    dists[dists == 1]  = 2

    # check if we have easy edges
    dists_sum = np.sum(dists, axis=1)

    ends = np.where(dists_sum<4)[0]
    if len(ends)>0:
        pt_i = ends[0]
        first_i = ends[0]
        got_end = True
    else:
        # otherwise just start at the beginning
        pt_i = 0
        first_i = 0
        got_end = False

    # walk through our dist graph, gathering points as we go
    inds = range(len(edge_points))
    new_pts = dq()
    forwards = True
    # this confusing bundle will get reused a bit...
    # we are making a new list of points, and don't want to double-count points
    # but we can't pop from edge_points directly, because then the indices from the
    # dist mat will be wrong. Instead we make a list of all the indices and pop
    # from that. But since popping will change the index/value parity, we
    # have to double index inds.pop(inds.index(etc.))

    new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
    while True:
        # get dict of connected points and distances
        # filtered by whether the index hasn't been added yet
        connected_pts = {k: dists[pt_i,k] for k in np.where(dists[pt_i,:])[0] if k in inds}

        # if we get nothing, we're either done or we have to go back to the first pt
        if len(connected_pts) == 0:
            if got_end:
                # still have points left, go back to first and go backwards
                pt_i = first_i
                forwards = False
                got_end = False
                continue
            else:
                # got to the end lets get outta here
                break

        # find point with min distance (take horiz/vert points before diags)
        pt_i = min(connected_pts, key=connected_pts.get)
        if forwards:
            new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
        else:
            new_pts.appendleft(edge_points[inds.pop(inds.index(pt_i))])

    return np.array(new_pts)