import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque as dq
from pandas import ewma, ewmstd
from itertools import count, cycle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import scipy.ndimage as ndi
from copy import copy
from scipy.ndimage import (gaussian_filter,
                           generate_binary_structure, binary_erosion, label)
from matplotlib import pyplot as plt
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

def edges2xy(edges, which_edge=None, sort=True):
    if not isinstance(which_edge, int):
        edges_xy = np.where(edges)
    else:
        edges_xy = np.where(edges==int(which_edge))

    edges_xy = np.column_stack(edges_xy)

    # reorder so points are in spatial order (rather than axis=0 order)
    if sort:
        edges_xy = order_points(edges_xy)

    return edges_xy




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

def edge_vectors(frame, sigma=1, return_angles = False):
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
    grad_x = filters.gaussian(cv2.Scharr(frame, ddepth=-1, dx=1, dy=0), sigma=sigma)
    grad_y = filters.gaussian(cv2.Scharr(frame, ddepth=-1, dx=0, dy=1), sigma=sigma)

    # Eigenvalues
    Axx = grad_x*grad_x
    Axy = grad_x*grad_y
    Ayy = grad_y*grad_y

    e1 = 0.5 * (Ayy + Axx - np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))
    e2 = 0.5 * (Ayy + Axx + np.sqrt((Ayy - Axx) ** 2 + 4 * (Axy ** 2)))
    # edges have eigenvalues with high e2 and low e1, so
    edge_scale = e2 - e1

    # norm the vectors
    grads = np.stack((grad_x, grad_y), axis=-1)
    grads = normalize(grads.reshape(-1,2), norm="l2", axis=1).reshape(grads.shape)
    grad_x, grad_y = grads[:,:,0], grads[:,:,1]

    if return_angles:
        # get angles 0-2pi
        angle = np.arccos(grad_x)
        angle[grad_y<=0] = np.arccos(-grad_x[grad_y<=0])+np.pi

        return grad_x, grad_y, edge_scale, angle

    else:
        return grad_x, grad_y, edge_scale

def scharr_canny(image, sigma, low_threshold=0.2, high_threshold=0.5):
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

    magnitude = e2-e1

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

    # skeletonize to reduce thick pixels we mighta missed
    output_mask = morphology.skeletonize(output_mask)

    return output_mask


def repair_edges(edges, frame, sigma=3):
    # connect contiguous edges, disconnect edges w/ sharp angles
    # expect a binary edge image, like from feature.canny
    # im_grad should be complex (array of vectors)

    # super inefficient rn, will eventually just work with xy's directly but...
    edges = edges.copy()

    label_edges = morphology.label(edges)

    uq_edges, counts = np.unique(label_edges, return_counts = True)
    uq_edges, counts = uq_edges[uq_edges>0], counts[1:]

    if len(uq_edges) == 0:
        return

    # delete tiny edges
    uq_edges = uq_edges[counts > 30]
    edges[~np.isin(label_edges, uq_edges)] = 0



    ##################################
    # first, go through and break edges at corners and inflection points
    edges = delete_corners(edges=edges, label_edges=label_edges, uq_edges=uq_edges)

    # delete tiny edges again
    edges = remove_tiny_edges(edges)

    # filter points that are convex around darkness
    #edges = positive_convexity(edges, frame)

    # one last time
    #edges = remove_tiny_edges(edges)

    # reconnect edges based on position, image gradient, and edge affinity
    edges = merge_curves(edges, frame, sigma=sigma)

    # final round of breaking curves if we introduced any bad ones in our merge
    #edges = delete_corners(edges=edges)

    return edges

def delete_corners(edges=None,label_edges=None, uq_edges=None):
    if (edges is None) and (label_edges is None):
        Exception('Need either edges or labeled edges!')

    if (edges is not None) and (label_edges is None):
        label_edges =morphology.label(edges)
        uq_edges = np.unique(label_edges)
        uq_edges = uq_edges[uq_edges > 0]

    if (label_edges is not None) and (uq_edges is None):
        edges = label_edges>0
        uq_edges = np.unique(label_edges)
        uq_edges = uq_edges[uq_edges > 0]

    if edges is None:
        edges = label_edges > 0

    edges_xy = [edges2xy(label_edges, e) for e in uq_edges]
    delete_mask = np.zeros(edges.shape, dtype=np.bool)
    num = 0
    for edge_xy in edges_xy:
        num +=1
        delete_points = break_corners(edge_xy)

        # if delete_points gets returned as False, delete this whole segment
        if delete_points == False:
            edges[edge_xy[:,0], edge_xy[:,1]] = 0
        elif len(delete_points)>0:
            for p in delete_points:
                delete_mask[p[0], p[1]] = True

    # do a dilation w/ square to get all the neighboring points too
    # use 5-diam square so reconnecting doesn't make T's
    delete_mask = morphology.binary_dilation(delete_mask, selem=morphology.square(5))
    edges[delete_mask] = 0
    return edges

def remove_tiny_edges(edges, thresh=30):
    label_edges = morphology.label(edges)
    uq_edges, counts = np.unique(label_edges, return_counts=True)
    uq_edges, counts = uq_edges[uq_edges > 0], counts[1:]
    uq_edges = uq_edges[counts > thresh]
    edges[~np.isin(label_edges, uq_edges)] = 0
    return edges

def merge_curves(edges, frame, sigma=3, threshold=0.5):
    '''
    Given a binary array of edges and a grayscale image,
    generate three metrics of continuity for corner points (edges of edges):
        - euclidean distance
        - cosine distance between image gradient
        - cosine distance between edge direction & direction to other points
    merge those above a given threshold

    :param edges:
    :param frame:
    :param sigma:
    :return:
    '''

    # get edge points
    label_edges = morphology.label(edges)
    uq_edges = np.unique(label_edges)
    uq_edges = uq_edges[uq_edges>0]
    edges_xy = [edges2xy(label_edges, e) for e in uq_edges]
    edge_points = np.row_stack([(e[0], e[-1]) for e in edges_xy])

    # image gradients
    grad_x, grad_y, edge_scale = edge_vectors(frame, sigma=sigma)

    ###################
    # Continuity metrics
    ###################
    ## 1. distance

    # get the  euclidean distances between points as a fraction of image size
    edge_dists = distance.squareform(distance.pdist(edge_points))/np.max(frame.shape)

    ## 2. gradient similarity
    # and the dot product of the image gradient
    edge_grads = np.column_stack((grad_x[edge_points[:,0], edge_points[:,1]],
                                  grad_y[edge_points[:,0], edge_points[:,1]]))
    edge_grads = distance.squareform(distance.pdist(edge_grads, "cosine"))

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
    edge_affinity = np.row_stack(
        [distance.cdist(edge_vects[i,:].reshape(1,2), edge_mids-edge_points[i,:], "cosine")
         for i in range(len(edge_segs))]
    )

    # get upper/lower indices manually so they match up right
    triu_indices = np.triu_indices(edge_affinity.shape[0], k= 1)
    tril_indices = triu_indices[::-1]

    # average top and bottom triangles - both edges should point at each other
    pair_affinity = np.mean((edge_affinity[tril_indices],
                             edge_affinity[triu_indices]), axis=0)
    edge_affinity[tril_indices] = pair_affinity
    edge_affinity[triu_indices] = pair_affinity

    ########################################
    # Clean up & combine merge metrics
    # want high values to be good, and for vals to be 0-1
    edge_dists = 1-((edge_dists-np.min(edge_dists))/(np.max(edge_dists)-np.min(edge_dists)))
    #edge_dists = 1-edge_dists
    edge_grads = 1-(edge_grads/2.) # cosine distance is 0-2
    edge_affinity = 1-(edge_affinity/2.)

    merge_score = edge_dists**6 * edge_grads**2 * edge_affinity**2
    merge_score[np.isnan(merge_score)] = 0.

    # zero nonmax joins (only want to join each end one time at most..
    # only keep scores if mutually max'd (avoids 3-way joins)
    col_max = np.argmax(merge_score, axis=0)
    dupe_mask = np.ones_like(merge_score, dtype=np.bool)
    dupe_mask[col_max[col_max], col_max] = 0
    merge_score[dupe_mask] = 0.

    # now only need triangle to avoid dupes
    merge_score[triu_indices] = 0.

    # get good edges to merge,
    # empirically threshold of 0.5 seems to get the visually salient ones
    merge_points = np.column_stack(np.where(merge_score>threshold))

    # filter points that are on the same curve...
    merge_points = merge_points[(merge_points/2)[:,0] != (merge_points/2)[:,1],:]

    # and draw them on the edge image
    for i in range(len(merge_points)):
        points = np.row_stack((edge_points[merge_points[i,0],:],
                              edge_points[merge_points[i,1],:]))
        join_points = line_mask(points)
        edges[join_points[:,0], join_points[:,1]] = 1

    return edges

def positive_convexity(edges, frame, brightness=True):
    # filter edges that are convex around darkness (or brightness

    # relabel and delete small edges
    label_edges = morphology.label(edges)
    uq_edges = np.unique(label_edges)
    uq_edges = uq_edges[uq_edges>0]


    for e in uq_edges:
        one_edge = label_edges.copy()
        one_edge[one_edge != e] = 0
        one_edge[one_edge == e] = 1

        hull = morphology.convex_hull_image(one_edge)

        inner_points = np.logical_xor(hull, morphology.binary_erosion(hull, morphology.disk(5)))
        outer_points = np.logical_xor(hull, morphology.binary_dilation(hull, morphology.disk(5)))

        inner_val = np.median(frame[inner_points])
        outer_val = np.median(frame[outer_points])

        if brightness:
            if inner_val < outer_val:
                edge_xy = edges2xy(label_edges, e)
                edges[edge_xy[:,0], edge_xy[:,1]] = 0
        else:
            if outer_val < inner_val:
                edge_xy = edges2xy(label_edges, e)
                edges[edge_xy[:,0], edge_xy[:,1]] = 0

    return edges

def break_corners(edge_xy, return_segments=False, corner_thresh=.3):
    # pass in an ordered list of x/y coords,
    # using prasad's line estimation we break edges at sharp turns/inflection points
    # returns an array of coords of corners/inflection points to
    # modify the edge image
    # Sharp turn threshold is set to ~65 degrees (1.1 rads) by default.

    # make prasad segments
    # keep both point and segment representations,
    # we want to work with segments, but we ultimately need to
    # figure out which point to delete
    #
    # there will be n points, n-1 segments, and n-2 angles.
    # so splitting the edge at point 0<i<n (we don't split endpoints dummy!)
    # will depend on angle i-1

    segs_points = prasad_lines(edge_xy)
    segs = nodes2segs(segs_points)

    # get angles between segments
    angles = [] # dot product between vectors, tells sharp angles
    inflection = [] # dot product between n+1 vector and (-y, x) of n vector
    for i in range(len(segs)-1):
        # make segments unit vectors
        seg1, seg2 = segs[i], segs[i+1]
        seg_n = normalize([[seg1[1][0]-seg1[0][0], seg1[1][1]-seg1[0][1]],
                              [seg2[1][0]-seg2[0][0], seg2[1][1]-seg2[0][1]]])

        # dot product of vector 2 and perp. vector to v1 (x,y)T = (-y,x)


        inflection.append(-seg_n[0,1]*seg_n[1,0]+seg_n[0,0]*seg_n[1,1])
        angles.append(seg_n[0, 0] * seg_n[1, 0] + seg_n[0, 1] * seg_n[1, 1])

    # first thing's first, if we get back a 2 pt segment
    # recommend to delete the whole segment by returning False

    if len(angles) == 0:
        return False


    # otherwise go through and find sharp angles and inflection points
    # boolean mask for deletion of points
    delete_pts = np.zeros(len(segs_points), dtype=np.bool)
    angles = np.array(angles)
    inflection = np.array(inflection)

    # look for sharp edges
    delete_pts[np.where(angles < corner_thresh)[0] + 1] = True

    # and inflection points
    # sorry for how this works so sucky, head unclear.
    signs = np.zeros(len(inflection), dtype=np.bool)
    signs[np.where(inflection >= 0)] = True
    last_deleted = False
    for i in range(len(angles)):
        if last_deleted:
            # don't want to double-punish inflection points,
            # when we cut, we make a new edge, so it doesnt make sense to
            # also cut the next sign for being different because it has
            # nothing to be different from in its new life
            last_deleted =False
            continue

        if i>0:
            if signs[i-1] != signs[i]:
                delete_pts[i+1] =True
                last_deleted=True


    # return the inds of the points to delete
    return [p for i, p in enumerate(segs_points) if delete_pts[i]]


def nodes2segs(segs):
    # convert a list of line segments specified by their x/y nodes
    # to x/y coords for each segment
    # ie.
    #    [[1, 1], [2,2], [3,3]]
    # becomes
    #    [[[1,1], [2,2]],
    #     [[2,2], [3,3]]]

    # for an nx4 array:
    # segs_out = np.column_stack((segs[:-1], segs[1:]))

    # for a list of lists:
    segs_out = []
    for pt1, pt2 in zip(segs[:-1], segs[1:]):
        segs_out.append([pt1, pt2])

    return segs_out

def plot_edge(labeled_edges, edge):
    # given an image of labeled edges (m x n array of integer labels)
    # scatterplot one edge in particular
    pass


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
        connected_pts = [k for k in np.where(dists[pt_i, :])[0] if k in inds]

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
        pt_i = connected_pts[np.argmax([dists[pt_i, k] for k in connected_pts])]
        if forwards:
            new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
        else:
            new_pts.appendleft(edge_points[inds.pop(inds.index(pt_i))])

    return np.array(new_pts)

def order_points_new(edge_points):
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
        #connected_pts = {k: dists[pt_i,k] for k in np.where(dists[pt_i,:])[0] if k in inds}
        connected_pts = [k for k in np.where(dists[pt_i, :])[0] if k in inds]

        # if we get one, cool. just append
        if len(connected_pts) == 1:
            #pt_i = connected_pts.keys()[0]
            pt_i = connected_pts[0]
            if forwards:
                new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
            else:
                new_pts.appendleft(edge_points[inds.pop(inds.index(pt_i))])

        # if we get more than one, find the longest one
        if len(connected_pts) > 1:
            chains = {}
            chain_inds = {}
            last_pts = {}
            # recursively call walk_points
            for k in connected_pts:
                chains[k], chain_inds[k], last_pts[k] = walk_points(dists, edge_points, k, inds)
            longest_chain = max(chains.items(), key=lambda x: len(x[1]))[0]
            if forwards:
                new_pts.extend(chains[longest_chain])
            else:
                new_pts.extendleft(chains[longest_chain])
            inds = chain_inds[longest_chain]
            pt_i = last_pts[longest_chain]

            if (len(inds)>0) and not got_end:
                # still have the other side to doooooo
                pt_i = first_i
                forwards = False
                got_end = False
                continue
            else:
                break

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
        #pt_i = min(connected_pts, key=connected_pts.get)


    return np.array(new_pts)

def walk_points(dists,edge_points, pt_i, inds):
    pt_i = copy(pt_i)
    inds = copy(inds)
    new_pts = []
    new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
    while True:
        # get dict of connected points and distances
        # filtered by whether the index hasn't been added yet
        try:
            connected_pts = [k for k in np.where(dists[pt_i,:])[0] if k in inds]
        except ValueError:
            connected_pts = [k for k in np.where(dists[pt_i, :]) if k in inds]

        # if we get nothing, we're done
        if len(connected_pts) == 0:
            break

        # find point with min distance (take horiz/vert points before diags)
        if len(connected_pts) == 1:
            pt_i = connected_pts[0]
            new_pts.append(edge_points[inds.pop(inds.index(pt_i))])

        if len(connected_pts) > 1:
            chains = {}
            chain_inds = {}
            last_pts = {}
            for k in connected_pts:
                chains[k], chain_inds[k], last_pts[k] = walk_points(dists, edge_points, k, inds)
            longest_chain = max(chains.items(), key=lambda x: len(x[1]))[0]
            new_pts.extend(chains[longest_chain])
            inds = chain_inds[longest_chain]
            pt_i = last_pts[longest_chain]

    return new_pts, inds, pt_i



def prasad_lines(edge, return_metrics=False):
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

    if return_metrics:
        precision = []
        reliability = []


    while first<last:

        mdev_results = prasad_maxlinedev(x[first:last+1], y[first:last+1], return_metrics=return_metrics)

        while mdev_results['d_max'] > mdev_results['del_tol_max']:
            if mdev_results['index_d_max']+first == last:
                last = len(x)-1
                break
            else:
                last = mdev_results['index_d_max']+first

            if (last == first+1) or (last==first):
                last = len(x)-1
                break

            try:
                mdev_results = prasad_maxlinedev(x[first:last+1], y[first:last+1], return_metrics=return_metrics)
            except IndexError:
                break

        seglist.append([x[last], y[last]])
        if return_metrics:
            precision.append(mdev_results['precision'])
            reliability.append(mdev_results['reliability'])

        first = last
        last = len(x)-1

    if return_metrics:
        return seglist, precision, reliability
    else:
        return seglist



def prasad_maxlinedev(x, y, return_metrics=False):
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

    s_mat = np.column_stack((x-x[first], y-y[first])) ** 2
    s_max = np.max(np.sqrt(np.sum(s_mat, axis=1)))
    del_phi_max = prasad_digital_error(s_max)
    results['del_tol_max'] = np.tan((del_phi_max *s_max))

    if return_metrics:
        results['precision'] = np.linalg.norm(dev, ord=2) / np.sqrt(float(last))
        results['reliability'] = np.sum(dev) / s_max

    return results

def prasad_digital_error(ss):
    # all credit to http://ieeexplore.ieee.org/document/6166585/
    # adapted from MATLAB scripts here: https://docs.google.com/open?id=0B10RxHxW3I92dG9SU0pNMV84alk

    phi = np.arange(0, np.pi*2, np.pi / 360)

    #s, phii = np.meshgrid(ss, phi)

    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    sin_plus_cos = sin_p+cos_p
    sin_minus_cos = sin_p-cos_p

    term1 = []
    term1.append(np.abs(cos_p))
    term1.append(np.abs(sin_p))
    term1.append(np.abs(sin_plus_cos))
    term1.append(np.abs(sin_minus_cos))

    tt2 = []
    tt2.append(sin_p/ss)
    tt2.append(cos_p/ ss)
    tt2.append(sin_minus_cos/ ss)
    tt2.append(sin_plus_cos/ ss)
    tt2.extend([-tt2[0], -tt2[1], -tt2[2], -tt2[3]])

    term2 = []
    for t2_item in tt2:
        term2.append(ss * (1 - t2_item + t2_item ** 2))

    case_value = []
    for c_i in range(8):
        case_value.append((1/ ss ** 2) * term1[c_i%4] * term2[c_i])

    return np.max(case_value)


from numpy import where, dstack, diff, meshgrid

def find_intersections(A, B):

    # min, max and all for arrays
    amin = lambda x1, x2: where(x1<x2, x1, x2)
    amax = lambda x1, x2: where(x1>x2, x1, x2)
    aall = lambda abools: dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(diff(line, axis=0))

    x11, x21 = meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    return xi[aall(xconds)], yi[aall(yconds)]



# do it
#each vector a line
# inds = np.indices(grad_x.shape)
# inds_y = inds[0].flatten()
# inds_x = inds[1].flatten()
# inds = np.column_stack((inds_x, inds_y))

# ends_y = inds_y+500*np.sin(angle.flatten())
# ends_x = inds_x+500*np.cos(angle.flatten())
# ends = np.column_stack((ends_x, ends_y))

# perms_inds = inds.copy()
# perms_ends = ends.copy()
# np.random.shuffle(perms_inds)
# np.random.shuffle(perms_ends)

# intersections = line_intersect(inds, ends, perms_inds, perms_ends)

def line_intersect(a1, a2, b1, b2):
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


def line_mask(ends):
    d = np.diff(ends, axis=0)[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)
    return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD
