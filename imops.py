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
import pandas as pd
# http://cdn.intechopen.com/pdfs/33559/InTech-Methods_for_ellipse_detection_from_edge_maps_of_real_images.pdf

def crop(im, roi):
    return im[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]


def invert_color(im):
    if im.dtype == 'uint8':
        return 255-im
    elif im.dtype == float:
        return 1.-im


def circle_mask(frame, ix, iy, rad):
    # get a boolean mask from circular parameters
    nx, ny = frame.shape
    pmask_x, pmask_y = np.ogrid[-iy:nx - iy, -ix:ny - ix]
    pmask = pmask_x ** 2 + pmask_y ** 2 <= rad ** 2

    return pmask


def edges2xy(edges, which_edge=None, order=True):
    if not isinstance(which_edge, int):
        edges_xy = np.where(edges)
    else:
        edges_xy = np.where(edges==int(which_edge))

    edges_xy = np.column_stack(edges_xy)

    # reorder so points are in spatial order (rather than axis=0 order)
    if order:
        edges_xy = order_points(edges_xy)

    return edges_xy


def preprocess_image(img, roi = None, gauss_sig=None, sig_cutoff=None, sig_gain=None, closing=3):
    if len(img.shape)>2:
        # TODO: this is what i'm talking about -- respect the --gray cmd line param.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = crop(img, roi)
    elif roi is not None:
        img = crop(img, roi)
    # otherwise we've got a recolored/cropped image already

    img = invert_color(img)

    img = exposure.equalize_hist(img)


    if gauss_sig:
        img = filters.gaussian(img, sigma=gauss_sig)

    if sig_cutoff and sig_gain:
        img = exposure.adjust_sigmoid(img, cutoff=sig_cutoff, gain=sig_gain)

    img = morphology.closing(img, selem=morphology.disk(closing))

    return img


def fit_ellipse(edges, which_edge=1):
    if edges.shape[1]>2:
        # imagelike
        edge_points = np.where(edges == which_edge)
        edge_points = np.column_stack((edge_points[1], edge_points[0]))
    else:
        edge_points = edges
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
    grad_x = cv2.GaussianBlur(cv2.Scharr(frame, ddepth=-1, dx=1, dy=0), ksize=(0,0), sigmaX=sigma)
    grad_y = cv2.GaussianBlur(cv2.Scharr(frame, ddepth=-1, dx=0, dy=1), ksize=(0,0), sigmaX=sigma)

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


def scharr_canny(image, sigma, low_threshold=0.2, high_threshold=0.5, grads = None):
    # skimage's canny but we get scharr grads instead of sobel,
    # and use the eigenvalues of the structure tensor rather than the hypotenuse
    # we can be passed a precomputed set of image gradients if we haven't already gotten them

    if grads is not None:
        isobel = grads['grad_y']
        jsobel = grads['grad_x']
        abs_isobel = np.abs(isobel)
        abs_jsobel = np.abs(jsobel)
        magnitude = grads['edge_mag']
    else:

        isobel = cv2.GaussianBlur(cv2.Scharr(image, ddepth=-1, dx=0, dy=1), ksize=(0,0), sigmaX=sigma)
        jsobel = cv2.GaussianBlur(cv2.Scharr(image, ddepth=-1, dx=1, dy=0), ksize=(0,0), sigmaX=sigma)


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
    #mask = np.ones(image.shape)
    #s = generate_binary_structure(2, 2)
    #eroded_mask = binary_erosion(mask, s, border_value=0)
    #eroded_mask = eroded_mask & (magnitude > 0)
    eroded_mask = np.zeros(image.shape, dtype=np.bool)
    eroded_mask[1:-1, 1:-1] = True


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
    #output_mask = morphology.skeletonize(output_mask)

    return output_mask

def parameterize_edges(edges, grad_x, grad_y, angles, small_thresh=20):
    # reduce binary 2d edge image to parameters
    edges = morphology.label(edges)

    uq_edges, counts = np.unique(edges, return_counts = True)
    uq_edges, counts = uq_edges[uq_edges>0], counts[1:]

    if len(uq_edges) == 0:
        return

    # get x/y representation so we don't have to keep passing a buncha booleans
    edges_xy = [edges2xy(edges, e) for e in uq_edges]

    # delete tiny edges
    edges_xy = [e for e in edges_xy if len(e)>small_thresh]
    if len(edges_xy)==0:
        return []

    # break edges at corners and inflection points
    edges_xy = break_corners(edges_xy)

    # delete tiny edges
    edges_xy = [e for e in edges_xy if len(e)>small_thresh]
    if len(edges_xy)==0:
        return []

    # melt list of edge xy points to dataframe
    edges_df = pd.DataFrame.from_records(edges_xy).T.melt(var_name="edge").dropna()
    xy = np.row_stack(edges_df.value)
    edges_df.drop('value', axis=1,inplace=True)
    edges_df['x'], edges_df['y'] = xy[:,0], xy[:,1]

    # add gradients and angles
    edges_df['grad_x'] = grad_x[edges_df['x'], edges_df['y']]
    edges_df['grad_y'] = grad_y[edges_df['x'], edges_df['y']]
    edges_df['angle'] = angles[edges_df['x'], edges_df['y']]
    return edges_df







def repair_edges(edges, frame, sigma=3, small_thresh=20, grads=None):
    # connect contiguous edges, disconnect edges w/ sharp angles
    # expect a binary edge image, like from feature.canny
    # im_grad should be complex (array of vectors)

    # super inefficient rn, will eventually just work with xy's directly but...
    #edges = edges.copy()

    label_edges = morphology.label(edges)

    uq_edges, counts = np.unique(label_edges, return_counts = True)
    uq_edges, counts = uq_edges[uq_edges>0], counts[1:]

    if len(uq_edges) == 0:
        return

    # get x/y representation so we don't have to keep passing a buncha booleans
    edges_xy = [edges2xy(label_edges, e) for e in uq_edges]

    ##################################
    # repair

    # delete tiny edges
    edges_xy = [e for e in edges_xy if len(e)>small_thresh]
    if len(edges_xy)==0:
        return []

    # break edges at corners and inflection points
    edges_xy = break_corners(edges_xy)

    # delete tiny edges again
    edges_xy = [e for e in edges_xy if len(e) > small_thresh]
    if len(edges_xy) == 0:
        return []

    # # filter points that are convex around darkness
    # edges_xy = positive_convexity(edges_xy, frame)
    # #
    # # # one last time
    # edges_xy = [e for e in edges_xy if len(e) > small_thresh]
    # if len(edges_xy) == 0:
    #     return []

    # reconnect edges based on position, image gradient, and edge affinity
    if grads is not None:
        edges_xy = merge_curves(edges_xy, frame, sigma=sigma, grads=grads)
    else:
        edges_xy = merge_curves(edges_xy, frame, sigma=sigma, grads=grads)

    # final round of breaking curves if we introduced any bad ones in our merge
    edges_xy = break_corners(edges_xy)

    return edges_xy

def break_corners(edges_xy):
    return_edges = []

    #delete_mask = np.zeros(edges.shape, dtype=np.bool)
    for edge_xy in edges_xy:
        return_edges.extend(break_at_corners(edge_xy))

    return return_edges


def break_at_corners(edge_xy, return_segments=False, corner_thresh=.6, delete_straight=False):
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

    if len(angles) == 0 and delete_straight:
        return []


    # otherwise go through and find sharp angles and inflection points
    # boolean mask for deletion of points
    angles = np.array(angles)
    inflection = np.array(inflection)

    # look for sharp edges
    split_pts = np.zeros(len(segs_points), dtype=np.bool)
    split_pts[np.where(angles < corner_thresh)[0] + 1] = True

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
                split_pts[i+1] =True
                last_deleted=True

    # return a list of split edges
    return_segs = []
    split_inds = np.where(split_pts)[0]
    if np.alen(split_inds)==0:
        return_segs.append(edge_xy)
    else:
        for split in np.where(split_pts)[0]:
            split_at = np.array(segs_points[split])
            split_ind = np.where((edge_xy == split_at).all(axis=1))[0][0]
            return_segs.append(edge_xy[:split_ind-2,:])
            edge_xy = edge_xy[split_ind+2:,:]
        return_segs.append(edge_xy)


    return return_segs


def remove_tiny_edges(edges, thresh=30):
    label_edges = morphology.label(edges)
    uq_edges, counts = np.unique(label_edges, return_counts=True)
    uq_edges, counts = uq_edges[uq_edges > 0], counts[1:]
    uq_edges = uq_edges[counts > thresh]
    edges[~np.isin(label_edges, uq_edges)] = 0
    return edges

def merge_curves(edges_xy, frame, sigma=3, threshold=0.5, keep_originals=True, only_max=False,
                 grads=False):
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
    edge_points = np.row_stack([(e[0], e[-1]) for e in edges_xy])

    # image gradients
    if not grads:
        grad_x, grad_y, edge_scale = edge_vectors(frame, sigma=sigma)
    else:
        grad_x, grad_y, edge_scale = grads['grad_x'], grads['grad_y'], grads['edge_mag']

    ###################
    # Continuity metrics
    ###################
    ## 1. distance

    # get the  euclidean distances between points as a fraction of image size
    edge_dists = distance.squareform(distance.pdist(edge_points))/np.min(frame.shape)

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

    merge_score = edge_dists * edge_grads**2 * edge_affinity**2
    merge_score[np.isnan(merge_score)] = 0.

    # zero nonmax joins (only want to join each end one time at most..
    # only keep scores if mutually max'd (avoids 3-way joins)
    if only_max:
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

    # make the merged segments, pop the originals and combine
    newsegs = []
    for p in merge_points:
        seg1 = edges_xy[p[0]//2]
        seg2 = edges_xy[p[1]//2]

        # stack correctly!
        # if we have an odd and an even, just append
        if p[0]%2<1 and p[1]%2>0:
            # ie. p[0] is at the beginning of the array, p[1] at the end
            newseg = np.row_stack((seg2, seg1))
        elif p[0]%2>0 and p[1]%2<1:
            # ie. the opposite
            newseg = np.row_stack((seg1, seg2))
        elif p[0]%2<1 and p[1]%2<1:
            # both are at the beginning, flip one and append to the start
            newseg = np.row_stack((np.flipud(seg1), seg2))
        elif p[0]%2>0 and p[1]%2>0:
            # only one more possibility...
            newseg = np.row_stack((seg1, np.flipud(seg2)))
        else:
            Exception("What the hell happened here? {}".format(p))

        newsegs.append(newseg)

    # now we can remove the original segs and append
    if keep_originals == False:
        edges_xy = [e for i, e in enumerate(edges_xy) if i not in merge_points.flatten()//2]

    edges_xy.extend(newsegs)

    # TODO: Return merge score as well
    return edges_xy

def positive_convexity(edges, frame, brightness=True):
    # filter edges that are convex around darkness (or brightness

    keep_inds = []
    for i, e in enumerate(edges):
        one_edge = np.zeros(frame.shape, dtype=np.bool)
        one_edge[e[:,0], e[:,1]] = True

        hull = morphology.convex_hull_image(one_edge)

        inner_points = np.logical_xor(hull, morphology.binary_erosion(hull, morphology.disk(5)))
        outer_points = np.logical_xor(hull, morphology.binary_dilation(hull, morphology.disk(5)))

        inner_val = np.median(frame[inner_points])
        outer_val = np.median(frame[outer_points])

        if brightness:
            if inner_val > outer_val:
                keep_inds.append(i)
        else:
            if outer_val < inner_val:
                keep_inds.append(i)

    edges = [edges[i] for i in keep_inds]

    return edges


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
    dists = distance.squareform(distance.pdist(edge_points))

    inds = {i: i for i in range(len(edge_points))}

    backwards = False
    found=False
    point = 0
    new_points = dq()
    new_points.append(edge_points[inds.pop(point), :])

    while True:
        close_enough = np.where(np.logical_and(dists[point, :] > 0, dists[point, :] < 3))[0]
        #close_enough = close_enough[np.isin(close_enough, inds.keys(), assume_unique=True)]
        close_enough = close_enough[np.argsort(dists[point,close_enough])]

        if np.alen(close_enough)==0:
            # either at one end or *the end*
            if not backwards:
                point = 0
                backwards = True
                continue
            else:
                break

        for p in close_enough:
            try:
                point = inds.pop(p)
                found=True
                break
            except KeyError:
                continue
        # point = close_enough[np.argmin(dists[point,close_enough])]

        # for p in close_enough:
        #     try:
        #         point=inds[p]
        #         point=p
        #         found=True
        #     except KeyError:
        #         # sorta the point.. keep trying until we find a key that hasn't been popped
        #         continue

        if not found:
            # didn't find a point, same as previous check
            if not backwards:
                point = 0
                backwards = True
                continue
            else:
                break



        if not backwards:
            #new_points.append(edge_points[inds.pop(point), :])
            new_points.append(edge_points[point, :])
        else:
            #new_points.appendleft(edge_points[inds.pop(point), :])
            new_points.appendleft(edge_points[point, :])

        found=False

    return np.row_stack(new_points)



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

    # constants are hardcoded for faster operation

    # phi = np.arange(0, np.pi*2, np.pi / 180)

    #s, phii = np.meshgrid(ss, phi)

    #sin_p = np.sin(phi)
    #cos_p = np.cos(phi)
    #sin_plus_cos = sin_p+cos_p
    #sin_minus_cos = sin_p-cos_p

    #term1 = []
    #term1.append(np.abs(cos_p))
    #term1.append(np.abs(sin_p))
    #term1.append(np.abs(sin_plus_cos))
    #term1.append(np.abs(sin_minus_cos))

    term1 = np.array([[1.00000000e+00, 9.99847695e-01, 9.99390827e-01, 9.98629535e-01,9.97564050e-01, 9.96194698e-01, 9.94521895e-01, 9.92546152e-01,9.90268069e-01, 9.87688341e-01, 9.84807753e-01, 9.81627183e-01,9.78147601e-01, 9.74370065e-01, 9.70295726e-01, 9.65925826e-01,9.61261696e-01, 9.56304756e-01, 9.51056516e-01, 9.45518576e-01,9.39692621e-01, 9.33580426e-01, 9.27183855e-01, 9.20504853e-01,9.13545458e-01, 9.06307787e-01, 8.98794046e-01, 8.91006524e-01,8.82947593e-01, 8.74619707e-01, 8.66025404e-01, 8.57167301e-01,8.48048096e-01, 8.38670568e-01, 8.29037573e-01, 8.19152044e-01,8.09016994e-01, 7.98635510e-01, 7.88010754e-01, 7.77145961e-01,7.66044443e-01, 7.54709580e-01, 7.43144825e-01, 7.31353702e-01,7.19339800e-01, 7.07106781e-01, 6.94658370e-01, 6.81998360e-01,6.69130606e-01, 6.56059029e-01, 6.42787610e-01, 6.29320391e-01,6.15661475e-01, 6.01815023e-01, 5.87785252e-01, 5.73576436e-01,5.59192903e-01, 5.44639035e-01, 5.29919264e-01, 5.15038075e-01,5.00000000e-01, 4.84809620e-01, 4.69471563e-01, 4.53990500e-01,4.38371147e-01, 4.22618262e-01, 4.06736643e-01, 3.90731128e-01,3.74606593e-01, 3.58367950e-01, 3.42020143e-01, 3.25568154e-01,3.09016994e-01, 2.92371705e-01, 2.75637356e-01, 2.58819045e-01,2.41921896e-01, 2.24951054e-01, 2.07911691e-01, 1.90808995e-01,1.73648178e-01, 1.56434465e-01, 1.39173101e-01, 1.21869343e-01,1.04528463e-01, 8.71557427e-02, 6.97564737e-02, 5.23359562e-02,3.48994967e-02, 1.74524064e-02, 6.12323400e-17, 1.74524064e-02,3.48994967e-02, 5.23359562e-02, 6.97564737e-02, 8.71557427e-02,1.04528463e-01, 1.21869343e-01, 1.39173101e-01, 1.56434465e-01,1.73648178e-01, 1.90808995e-01, 2.07911691e-01, 2.24951054e-01,2.41921896e-01, 2.58819045e-01, 2.75637356e-01, 2.92371705e-01,3.09016994e-01, 3.25568154e-01, 3.42020143e-01, 3.58367950e-01,3.74606593e-01, 3.90731128e-01, 4.06736643e-01, 4.22618262e-01,4.38371147e-01, 4.53990500e-01, 4.69471563e-01, 4.84809620e-01,5.00000000e-01, 5.15038075e-01, 5.29919264e-01, 5.44639035e-01,5.59192903e-01, 5.73576436e-01, 5.87785252e-01, 6.01815023e-01,6.15661475e-01, 6.29320391e-01, 6.42787610e-01, 6.56059029e-01,6.69130606e-01, 6.81998360e-01, 6.94658370e-01, 7.07106781e-01,7.19339800e-01, 7.31353702e-01, 7.43144825e-01, 7.54709580e-01,7.66044443e-01, 7.77145961e-01, 7.88010754e-01, 7.98635510e-01,8.09016994e-01, 8.19152044e-01, 8.29037573e-01, 8.38670568e-01,8.48048096e-01, 8.57167301e-01, 8.66025404e-01, 8.74619707e-01,8.82947593e-01, 8.91006524e-01, 8.98794046e-01, 9.06307787e-01,9.13545458e-01, 9.20504853e-01, 9.27183855e-01, 9.33580426e-01,9.39692621e-01, 9.45518576e-01, 9.51056516e-01, 9.56304756e-01,9.61261696e-01, 9.65925826e-01, 9.70295726e-01, 9.74370065e-01,9.78147601e-01, 9.81627183e-01, 9.84807753e-01, 9.87688341e-01,9.90268069e-01, 9.92546152e-01, 9.94521895e-01, 9.96194698e-01,9.97564050e-01, 9.98629535e-01, 9.99390827e-01, 9.99847695e-01,1.00000000e+00, 9.99847695e-01, 9.99390827e-01, 9.98629535e-01,9.97564050e-01, 9.96194698e-01, 9.94521895e-01, 9.92546152e-01,9.90268069e-01, 9.87688341e-01, 9.84807753e-01, 9.81627183e-01,9.78147601e-01, 9.74370065e-01, 9.70295726e-01, 9.65925826e-01,9.61261696e-01, 9.56304756e-01, 9.51056516e-01, 9.45518576e-01,9.39692621e-01, 9.33580426e-01, 9.27183855e-01, 9.20504853e-01,9.13545458e-01, 9.06307787e-01, 8.98794046e-01, 8.91006524e-01,8.82947593e-01, 8.74619707e-01, 8.66025404e-01, 8.57167301e-01,8.48048096e-01, 8.38670568e-01, 8.29037573e-01, 8.19152044e-01,8.09016994e-01, 7.98635510e-01, 7.88010754e-01, 7.77145961e-01,7.66044443e-01, 7.54709580e-01, 7.43144825e-01, 7.31353702e-01,7.19339800e-01, 7.07106781e-01, 6.94658370e-01, 6.81998360e-01,6.69130606e-01, 6.56059029e-01, 6.42787610e-01, 6.29320391e-01,6.15661475e-01, 6.01815023e-01, 5.87785252e-01, 5.73576436e-01,5.59192903e-01, 5.44639035e-01, 5.29919264e-01, 5.15038075e-01,5.00000000e-01, 4.84809620e-01, 4.69471563e-01, 4.53990500e-01,4.38371147e-01, 4.22618262e-01, 4.06736643e-01, 3.90731128e-01,3.74606593e-01, 3.58367950e-01, 3.42020143e-01, 3.25568154e-01,3.09016994e-01, 2.92371705e-01, 2.75637356e-01, 2.58819045e-01,2.41921896e-01, 2.24951054e-01, 2.07911691e-01, 1.90808995e-01,1.73648178e-01, 1.56434465e-01, 1.39173101e-01, 1.21869343e-01,1.04528463e-01, 8.71557427e-02, 6.97564737e-02, 5.23359562e-02,3.48994967e-02, 1.74524064e-02, 1.83697020e-16, 1.74524064e-02,3.48994967e-02, 5.23359562e-02, 6.97564737e-02, 8.71557427e-02,1.04528463e-01, 1.21869343e-01, 1.39173101e-01, 1.56434465e-01,1.73648178e-01, 1.90808995e-01, 2.07911691e-01, 2.24951054e-01,2.41921896e-01, 2.58819045e-01, 2.75637356e-01, 2.92371705e-01,3.09016994e-01, 3.25568154e-01, 3.42020143e-01, 3.58367950e-01,3.74606593e-01, 3.90731128e-01, 4.06736643e-01, 4.22618262e-01,4.38371147e-01, 4.53990500e-01, 4.69471563e-01, 4.84809620e-01,5.00000000e-01, 5.15038075e-01, 5.29919264e-01, 5.44639035e-01,5.59192903e-01, 5.73576436e-01, 5.87785252e-01, 6.01815023e-01,6.15661475e-01, 6.29320391e-01, 6.42787610e-01, 6.56059029e-01,6.69130606e-01, 6.81998360e-01, 6.94658370e-01, 7.07106781e-01,7.19339800e-01, 7.31353702e-01, 7.43144825e-01, 7.54709580e-01,7.66044443e-01, 7.77145961e-01, 7.88010754e-01, 7.98635510e-01,8.09016994e-01, 8.19152044e-01, 8.29037573e-01, 8.38670568e-01,8.48048096e-01, 8.57167301e-01, 8.66025404e-01, 8.74619707e-01,8.82947593e-01, 8.91006524e-01, 8.98794046e-01, 9.06307787e-01,9.13545458e-01, 9.20504853e-01, 9.27183855e-01, 9.33580426e-01,9.39692621e-01, 9.45518576e-01, 9.51056516e-01, 9.56304756e-01,9.61261696e-01, 9.65925826e-01, 9.70295726e-01, 9.74370065e-01,9.78147601e-01, 9.81627183e-01, 9.84807753e-01, 9.87688341e-01,9.90268069e-01, 9.92546152e-01, 9.94521895e-01, 9.96194698e-01,9.97564050e-01, 9.98629535e-01, 9.99390827e-01, 9.99847695e-01],
                      [0.00000000e+00, 1.74524064e-02, 3.48994967e-02, 5.23359562e-02,6.97564737e-02, 8.71557427e-02, 1.04528463e-01, 1.21869343e-01,1.39173101e-01, 1.56434465e-01, 1.73648178e-01, 1.90808995e-01,2.07911691e-01, 2.24951054e-01, 2.41921896e-01, 2.58819045e-01,2.75637356e-01, 2.92371705e-01, 3.09016994e-01, 3.25568154e-01,3.42020143e-01, 3.58367950e-01, 3.74606593e-01, 3.90731128e-01,4.06736643e-01, 4.22618262e-01, 4.38371147e-01, 4.53990500e-01,4.69471563e-01, 4.84809620e-01, 5.00000000e-01, 5.15038075e-01,5.29919264e-01, 5.44639035e-01, 5.59192903e-01, 5.73576436e-01,5.87785252e-01, 6.01815023e-01, 6.15661475e-01, 6.29320391e-01,6.42787610e-01, 6.56059029e-01, 6.69130606e-01, 6.81998360e-01,6.94658370e-01, 7.07106781e-01, 7.19339800e-01, 7.31353702e-01,7.43144825e-01, 7.54709580e-01, 7.66044443e-01, 7.77145961e-01,7.88010754e-01, 7.98635510e-01, 8.09016994e-01, 8.19152044e-01,8.29037573e-01, 8.38670568e-01, 8.48048096e-01, 8.57167301e-01,8.66025404e-01, 8.74619707e-01, 8.82947593e-01, 8.91006524e-01,8.98794046e-01, 9.06307787e-01, 9.13545458e-01, 9.20504853e-01,9.27183855e-01, 9.33580426e-01, 9.39692621e-01, 9.45518576e-01,9.51056516e-01, 9.56304756e-01, 9.61261696e-01, 9.65925826e-01,9.70295726e-01, 9.74370065e-01, 9.78147601e-01, 9.81627183e-01,9.84807753e-01, 9.87688341e-01, 9.90268069e-01, 9.92546152e-01,9.94521895e-01, 9.96194698e-01, 9.97564050e-01, 9.98629535e-01,9.99390827e-01, 9.99847695e-01, 1.00000000e+00, 9.99847695e-01,9.99390827e-01, 9.98629535e-01, 9.97564050e-01, 9.96194698e-01,9.94521895e-01, 9.92546152e-01, 9.90268069e-01, 9.87688341e-01,9.84807753e-01, 9.81627183e-01, 9.78147601e-01, 9.74370065e-01,9.70295726e-01, 9.65925826e-01, 9.61261696e-01, 9.56304756e-01,9.51056516e-01, 9.45518576e-01, 9.39692621e-01, 9.33580426e-01,9.27183855e-01, 9.20504853e-01, 9.13545458e-01, 9.06307787e-01,8.98794046e-01, 8.91006524e-01, 8.82947593e-01, 8.74619707e-01,8.66025404e-01, 8.57167301e-01, 8.48048096e-01, 8.38670568e-01,8.29037573e-01, 8.19152044e-01, 8.09016994e-01, 7.98635510e-01,7.88010754e-01, 7.77145961e-01, 7.66044443e-01, 7.54709580e-01,7.43144825e-01, 7.31353702e-01, 7.19339800e-01, 7.07106781e-01,6.94658370e-01, 6.81998360e-01, 6.69130606e-01, 6.56059029e-01,6.42787610e-01, 6.29320391e-01, 6.15661475e-01, 6.01815023e-01,5.87785252e-01, 5.73576436e-01, 5.59192903e-01, 5.44639035e-01,5.29919264e-01, 5.15038075e-01, 5.00000000e-01, 4.84809620e-01,4.69471563e-01, 4.53990500e-01, 4.38371147e-01, 4.22618262e-01,4.06736643e-01, 3.90731128e-01, 3.74606593e-01, 3.58367950e-01,3.42020143e-01, 3.25568154e-01, 3.09016994e-01, 2.92371705e-01,2.75637356e-01, 2.58819045e-01, 2.41921896e-01, 2.24951054e-01,2.07911691e-01, 1.90808995e-01, 1.73648178e-01, 1.56434465e-01,1.39173101e-01, 1.21869343e-01, 1.04528463e-01, 8.71557427e-02,6.97564737e-02, 5.23359562e-02, 3.48994967e-02, 1.74524064e-02,1.22464680e-16, 1.74524064e-02, 3.48994967e-02, 5.23359562e-02,6.97564737e-02, 8.71557427e-02, 1.04528463e-01, 1.21869343e-01,1.39173101e-01, 1.56434465e-01, 1.73648178e-01, 1.90808995e-01,2.07911691e-01, 2.24951054e-01, 2.41921896e-01, 2.58819045e-01,2.75637356e-01, 2.92371705e-01, 3.09016994e-01, 3.25568154e-01,3.42020143e-01, 3.58367950e-01, 3.74606593e-01, 3.90731128e-01,4.06736643e-01, 4.22618262e-01, 4.38371147e-01, 4.53990500e-01,4.69471563e-01, 4.84809620e-01, 5.00000000e-01, 5.15038075e-01,5.29919264e-01, 5.44639035e-01, 5.59192903e-01, 5.73576436e-01,5.87785252e-01, 6.01815023e-01, 6.15661475e-01, 6.29320391e-01,6.42787610e-01, 6.56059029e-01, 6.69130606e-01, 6.81998360e-01,6.94658370e-01, 7.07106781e-01, 7.19339800e-01, 7.31353702e-01,7.43144825e-01, 7.54709580e-01, 7.66044443e-01, 7.77145961e-01,7.88010754e-01, 7.98635510e-01, 8.09016994e-01, 8.19152044e-01,8.29037573e-01, 8.38670568e-01, 8.48048096e-01, 8.57167301e-01,8.66025404e-01, 8.74619707e-01, 8.82947593e-01, 8.91006524e-01,8.98794046e-01, 9.06307787e-01, 9.13545458e-01, 9.20504853e-01,9.27183855e-01, 9.33580426e-01, 9.39692621e-01, 9.45518576e-01,9.51056516e-01, 9.56304756e-01, 9.61261696e-01, 9.65925826e-01,9.70295726e-01, 9.74370065e-01, 9.78147601e-01, 9.81627183e-01,9.84807753e-01, 9.87688341e-01, 9.90268069e-01, 9.92546152e-01,9.94521895e-01, 9.96194698e-01, 9.97564050e-01, 9.98629535e-01,9.99390827e-01, 9.99847695e-01, 1.00000000e+00, 9.99847695e-01,9.99390827e-01, 9.98629535e-01, 9.97564050e-01, 9.96194698e-01,9.94521895e-01, 9.92546152e-01, 9.90268069e-01, 9.87688341e-01,9.84807753e-01, 9.81627183e-01, 9.78147601e-01, 9.74370065e-01,9.70295726e-01, 9.65925826e-01, 9.61261696e-01, 9.56304756e-01,9.51056516e-01, 9.45518576e-01, 9.39692621e-01, 9.33580426e-01,9.27183855e-01, 9.20504853e-01, 9.13545458e-01, 9.06307787e-01,8.98794046e-01, 8.91006524e-01, 8.82947593e-01, 8.74619707e-01,8.66025404e-01, 8.57167301e-01, 8.48048096e-01, 8.38670568e-01,8.29037573e-01, 8.19152044e-01, 8.09016994e-01, 7.98635510e-01,7.88010754e-01, 7.77145961e-01, 7.66044443e-01, 7.54709580e-01,7.43144825e-01, 7.31353702e-01, 7.19339800e-01, 7.07106781e-01,6.94658370e-01, 6.81998360e-01, 6.69130606e-01, 6.56059029e-01,6.42787610e-01, 6.29320391e-01, 6.15661475e-01, 6.01815023e-01,5.87785252e-01, 5.73576436e-01, 5.59192903e-01, 5.44639035e-01,5.29919264e-01, 5.15038075e-01, 5.00000000e-01, 4.84809620e-01,4.69471563e-01, 4.53990500e-01, 4.38371147e-01, 4.22618262e-01,4.06736643e-01, 3.90731128e-01, 3.74606593e-01, 3.58367950e-01,3.42020143e-01, 3.25568154e-01, 3.09016994e-01, 2.92371705e-01,2.75637356e-01, 2.58819045e-01, 2.41921896e-01, 2.24951054e-01,2.07911691e-01, 1.90808995e-01, 1.73648178e-01, 1.56434465e-01,1.39173101e-01, 1.21869343e-01, 1.04528463e-01, 8.71557427e-02,6.97564737e-02, 5.23359562e-02, 3.48994967e-02, 1.74524064e-02],
                      [1.00000000e+00, 1.01730010e+00, 1.03429032e+00, 1.05096549e+00,1.06732052e+00, 1.08335044e+00, 1.09905036e+00, 1.11441550e+00,1.12944117e+00, 1.14412281e+00, 1.15845593e+00, 1.17243618e+00,1.18605929e+00, 1.19932112e+00, 1.21221762e+00, 1.22474487e+00,1.23689905e+00, 1.24867646e+00, 1.26007351e+00, 1.27108673e+00,1.28171276e+00, 1.29194838e+00, 1.30179045e+00, 1.31123598e+00,1.32028210e+00, 1.32892605e+00, 1.33716519e+00, 1.34499702e+00,1.35241916e+00, 1.35942933e+00, 1.36602540e+00, 1.37220538e+00,1.37796736e+00, 1.38330960e+00, 1.38823048e+00, 1.39272848e+00,1.39680225e+00, 1.40045053e+00, 1.40367223e+00, 1.40646635e+00,1.40883205e+00, 1.41076861e+00, 1.41227543e+00, 1.41335206e+00,1.41399817e+00, 1.41421356e+00, 1.41399817e+00, 1.41335206e+00,1.41227543e+00, 1.41076861e+00, 1.40883205e+00, 1.40646635e+00,1.40367223e+00, 1.40045053e+00, 1.39680225e+00, 1.39272848e+00,1.38823048e+00, 1.38330960e+00, 1.37796736e+00, 1.37220538e+00,1.36602540e+00, 1.35942933e+00, 1.35241916e+00, 1.34499702e+00,1.33716519e+00, 1.32892605e+00, 1.32028210e+00, 1.31123598e+00,1.30179045e+00, 1.29194838e+00, 1.28171276e+00, 1.27108673e+00,1.26007351e+00, 1.24867646e+00, 1.23689905e+00, 1.22474487e+00,1.21221762e+00, 1.19932112e+00, 1.18605929e+00, 1.17243618e+00,1.15845593e+00, 1.14412281e+00, 1.12944117e+00, 1.11441550e+00,1.09905036e+00, 1.08335044e+00, 1.06732052e+00, 1.05096549e+00,1.03429032e+00, 1.01730010e+00, 1.00000000e+00, 9.82395289e-01,9.64491330e-01, 9.46293579e-01, 9.27807577e-01, 9.09038955e-01,8.89993432e-01, 8.70676808e-01, 8.51094968e-01, 8.31253876e-01,8.11159575e-01, 7.90818188e-01, 7.70235910e-01, 7.49419010e-01,7.28373831e-01, 7.07106781e-01, 6.85624340e-01, 6.63933051e-01,6.42039522e-01, 6.19950421e-01, 5.97672477e-01, 5.75212477e-01,5.52577261e-01, 5.29773725e-01, 5.06808815e-01, 4.83689525e-01,4.60422900e-01, 4.37016024e-01, 4.13476030e-01, 3.89810087e-01,3.66025404e-01, 3.42129226e-01, 3.18128832e-01, 2.94031533e-01,2.69844669e-01, 2.45575608e-01, 2.21231742e-01, 1.96820487e-01,1.72349278e-01, 1.47825570e-01, 1.23256833e-01, 9.86505512e-02,7.40142191e-02, 4.93553416e-02, 2.46814299e-02, 1.11022302e-16,2.46814299e-02, 4.93553416e-02, 7.40142191e-02, 9.86505512e-02,1.23256833e-01, 1.47825570e-01, 1.72349278e-01, 1.96820487e-01,2.21231742e-01, 2.45575608e-01, 2.69844669e-01, 2.94031533e-01,3.18128832e-01, 3.42129226e-01, 3.66025404e-01, 3.89810087e-01,4.13476030e-01, 4.37016024e-01, 4.60422900e-01, 4.83689525e-01,5.06808815e-01, 5.29773725e-01, 5.52577261e-01, 5.75212477e-01,5.97672477e-01, 6.19950421e-01, 6.42039522e-01, 6.63933051e-01,6.85624340e-01, 7.07106781e-01, 7.28373831e-01, 7.49419010e-01,7.70235910e-01, 7.90818188e-01, 8.11159575e-01, 8.31253876e-01,8.51094968e-01, 8.70676808e-01, 8.89993432e-01, 9.09038955e-01,9.27807577e-01, 9.46293579e-01, 9.64491330e-01, 9.82395289e-01,1.00000000e+00, 1.01730010e+00, 1.03429032e+00, 1.05096549e+00,1.06732052e+00, 1.08335044e+00, 1.09905036e+00, 1.11441550e+00,1.12944117e+00, 1.14412281e+00, 1.15845593e+00, 1.17243618e+00,1.18605929e+00, 1.19932112e+00, 1.21221762e+00, 1.22474487e+00,1.23689905e+00, 1.24867646e+00, 1.26007351e+00, 1.27108673e+00,1.28171276e+00, 1.29194838e+00, 1.30179045e+00, 1.31123598e+00,1.32028210e+00, 1.32892605e+00, 1.33716519e+00, 1.34499702e+00,1.35241916e+00, 1.35942933e+00, 1.36602540e+00, 1.37220538e+00,1.37796736e+00, 1.38330960e+00, 1.38823048e+00, 1.39272848e+00,1.39680225e+00, 1.40045053e+00, 1.40367223e+00, 1.40646635e+00,1.40883205e+00, 1.41076861e+00, 1.41227543e+00, 1.41335206e+00,1.41399817e+00, 1.41421356e+00, 1.41399817e+00, 1.41335206e+00,1.41227543e+00, 1.41076861e+00, 1.40883205e+00, 1.40646635e+00,1.40367223e+00, 1.40045053e+00, 1.39680225e+00, 1.39272848e+00,1.38823048e+00, 1.38330960e+00, 1.37796736e+00, 1.37220538e+00,1.36602540e+00, 1.35942933e+00, 1.35241916e+00, 1.34499702e+00,1.33716519e+00, 1.32892605e+00, 1.32028210e+00, 1.31123598e+00,1.30179045e+00, 1.29194838e+00, 1.28171276e+00, 1.27108673e+00,1.26007351e+00, 1.24867646e+00, 1.23689905e+00, 1.22474487e+00,1.21221762e+00, 1.19932112e+00, 1.18605929e+00, 1.17243618e+00,1.15845593e+00, 1.14412281e+00, 1.12944117e+00, 1.11441550e+00,1.09905036e+00, 1.08335044e+00, 1.06732052e+00, 1.05096549e+00,1.03429032e+00, 1.01730010e+00, 1.00000000e+00, 9.82395289e-01,9.64491330e-01, 9.46293579e-01, 9.27807577e-01, 9.09038955e-01,8.89993432e-01, 8.70676808e-01, 8.51094968e-01, 8.31253876e-01,8.11159575e-01, 7.90818188e-01, 7.70235910e-01, 7.49419010e-01,7.28373831e-01, 7.07106781e-01, 6.85624340e-01, 6.63933051e-01,6.42039522e-01, 6.19950421e-01, 5.97672477e-01, 5.75212477e-01,5.52577261e-01, 5.29773725e-01, 5.06808815e-01, 4.83689525e-01,4.60422900e-01, 4.37016024e-01, 4.13476030e-01, 3.89810087e-01,3.66025404e-01, 3.42129226e-01, 3.18128832e-01, 2.94031533e-01,2.69844669e-01, 2.45575608e-01, 2.21231742e-01, 1.96820487e-01,1.72349278e-01, 1.47825570e-01, 1.23256833e-01, 9.86505512e-02,7.40142191e-02, 4.93553416e-02, 2.46814299e-02, 3.33066907e-16,2.46814299e-02, 4.93553416e-02, 7.40142191e-02, 9.86505512e-02,1.23256833e-01, 1.47825570e-01, 1.72349278e-01, 1.96820487e-01,2.21231742e-01, 2.45575608e-01, 2.69844669e-01, 2.94031533e-01,3.18128832e-01, 3.42129226e-01, 3.66025404e-01, 3.89810087e-01,4.13476030e-01, 4.37016024e-01, 4.60422900e-01, 4.83689525e-01,5.06808815e-01, 5.29773725e-01, 5.52577261e-01, 5.75212477e-01,5.97672477e-01, 6.19950421e-01, 6.42039522e-01, 6.63933051e-01,6.85624340e-01, 7.07106781e-01, 7.28373831e-01, 7.49419010e-01,7.70235910e-01, 7.90818188e-01, 8.11159575e-01, 8.31253876e-01,8.51094968e-01, 8.70676808e-01, 8.89993432e-01, 9.09038955e-01,9.27807577e-01, 9.46293579e-01, 9.64491330e-01, 9.82395289e-01],
                      [1.00000000e+00, 9.82395289e-01, 9.64491330e-01, 9.46293579e-01,9.27807577e-01, 9.09038955e-01, 8.89993432e-01, 8.70676808e-01,8.51094968e-01, 8.31253876e-01, 8.11159575e-01, 7.90818188e-01,7.70235910e-01, 7.49419010e-01, 7.28373831e-01, 7.07106781e-01,6.85624340e-01, 6.63933051e-01, 6.42039522e-01, 6.19950421e-01,5.97672477e-01, 5.75212477e-01, 5.52577261e-01, 5.29773725e-01,5.06808815e-01, 4.83689525e-01, 4.60422900e-01, 4.37016024e-01,4.13476030e-01, 3.89810087e-01, 3.66025404e-01, 3.42129226e-01,3.18128832e-01, 2.94031533e-01, 2.69844669e-01, 2.45575608e-01,2.21231742e-01, 1.96820487e-01, 1.72349278e-01, 1.47825570e-01,1.23256833e-01, 9.86505512e-02, 7.40142191e-02, 4.93553416e-02,2.46814299e-02, 1.11022302e-16, 2.46814299e-02, 4.93553416e-02,7.40142191e-02, 9.86505512e-02, 1.23256833e-01, 1.47825570e-01,1.72349278e-01, 1.96820487e-01, 2.21231742e-01, 2.45575608e-01,2.69844669e-01, 2.94031533e-01, 3.18128832e-01, 3.42129226e-01,3.66025404e-01, 3.89810087e-01, 4.13476030e-01, 4.37016024e-01,4.60422900e-01, 4.83689525e-01, 5.06808815e-01, 5.29773725e-01,5.52577261e-01, 5.75212477e-01, 5.97672477e-01, 6.19950421e-01,6.42039522e-01, 6.63933051e-01, 6.85624340e-01, 7.07106781e-01,7.28373831e-01, 7.49419010e-01, 7.70235910e-01, 7.90818188e-01,8.11159575e-01, 8.31253876e-01, 8.51094968e-01, 8.70676808e-01,8.89993432e-01, 9.09038955e-01, 9.27807577e-01, 9.46293579e-01,9.64491330e-01, 9.82395289e-01, 1.00000000e+00, 1.01730010e+00,1.03429032e+00, 1.05096549e+00, 1.06732052e+00, 1.08335044e+00,1.09905036e+00, 1.11441550e+00, 1.12944117e+00, 1.14412281e+00,1.15845593e+00, 1.17243618e+00, 1.18605929e+00, 1.19932112e+00,1.21221762e+00, 1.22474487e+00, 1.23689905e+00, 1.24867646e+00,1.26007351e+00, 1.27108673e+00, 1.28171276e+00, 1.29194838e+00,1.30179045e+00, 1.31123598e+00, 1.32028210e+00, 1.32892605e+00,1.33716519e+00, 1.34499702e+00, 1.35241916e+00, 1.35942933e+00,1.36602540e+00, 1.37220538e+00, 1.37796736e+00, 1.38330960e+00,1.38823048e+00, 1.39272848e+00, 1.39680225e+00, 1.40045053e+00,1.40367223e+00, 1.40646635e+00, 1.40883205e+00, 1.41076861e+00,1.41227543e+00, 1.41335206e+00, 1.41399817e+00, 1.41421356e+00,1.41399817e+00, 1.41335206e+00, 1.41227543e+00, 1.41076861e+00,1.40883205e+00, 1.40646635e+00, 1.40367223e+00, 1.40045053e+00,1.39680225e+00, 1.39272848e+00, 1.38823048e+00, 1.38330960e+00,1.37796736e+00, 1.37220538e+00, 1.36602540e+00, 1.35942933e+00,1.35241916e+00, 1.34499702e+00, 1.33716519e+00, 1.32892605e+00,1.32028210e+00, 1.31123598e+00, 1.30179045e+00, 1.29194838e+00,1.28171276e+00, 1.27108673e+00, 1.26007351e+00, 1.24867646e+00,1.23689905e+00, 1.22474487e+00, 1.21221762e+00, 1.19932112e+00,1.18605929e+00, 1.17243618e+00, 1.15845593e+00, 1.14412281e+00,1.12944117e+00, 1.11441550e+00, 1.09905036e+00, 1.08335044e+00,1.06732052e+00, 1.05096549e+00, 1.03429032e+00, 1.01730010e+00,1.00000000e+00, 9.82395289e-01, 9.64491330e-01, 9.46293579e-01,9.27807577e-01, 9.09038955e-01, 8.89993432e-01, 8.70676808e-01,8.51094968e-01, 8.31253876e-01, 8.11159575e-01, 7.90818188e-01,7.70235910e-01, 7.49419010e-01, 7.28373831e-01, 7.07106781e-01,6.85624340e-01, 6.63933051e-01, 6.42039522e-01, 6.19950421e-01,5.97672477e-01, 5.75212477e-01, 5.52577261e-01, 5.29773725e-01,5.06808815e-01, 4.83689525e-01, 4.60422900e-01, 4.37016024e-01,4.13476030e-01, 3.89810087e-01, 3.66025404e-01, 3.42129226e-01,3.18128832e-01, 2.94031533e-01, 2.69844669e-01, 2.45575608e-01,2.21231742e-01, 1.96820487e-01, 1.72349278e-01, 1.47825570e-01,1.23256833e-01, 9.86505512e-02, 7.40142191e-02, 4.93553416e-02,2.46814299e-02, 2.22044605e-16, 2.46814299e-02, 4.93553416e-02,7.40142191e-02, 9.86505512e-02, 1.23256833e-01, 1.47825570e-01,1.72349278e-01, 1.96820487e-01, 2.21231742e-01, 2.45575608e-01,2.69844669e-01, 2.94031533e-01, 3.18128832e-01, 3.42129226e-01,3.66025404e-01, 3.89810087e-01, 4.13476030e-01, 4.37016024e-01,4.60422900e-01, 4.83689525e-01, 5.06808815e-01, 5.29773725e-01,5.52577261e-01, 5.75212477e-01, 5.97672477e-01, 6.19950421e-01,6.42039522e-01, 6.63933051e-01, 6.85624340e-01, 7.07106781e-01,7.28373831e-01, 7.49419010e-01, 7.70235910e-01, 7.90818188e-01,8.11159575e-01, 8.31253876e-01, 8.51094968e-01, 8.70676808e-01,8.89993432e-01, 9.09038955e-01, 9.27807577e-01, 9.46293579e-01,9.64491330e-01, 9.82395289e-01, 1.00000000e+00, 1.01730010e+00,1.03429032e+00, 1.05096549e+00, 1.06732052e+00, 1.08335044e+00,1.09905036e+00, 1.11441550e+00, 1.12944117e+00, 1.14412281e+00,1.15845593e+00, 1.17243618e+00, 1.18605929e+00, 1.19932112e+00,1.21221762e+00, 1.22474487e+00, 1.23689905e+00, 1.24867646e+00,1.26007351e+00, 1.27108673e+00, 1.28171276e+00, 1.29194838e+00,1.30179045e+00, 1.31123598e+00, 1.32028210e+00, 1.32892605e+00,1.33716519e+00, 1.34499702e+00, 1.35241916e+00, 1.35942933e+00,1.36602540e+00, 1.37220538e+00, 1.37796736e+00, 1.38330960e+00,1.38823048e+00, 1.39272848e+00, 1.39680225e+00, 1.40045053e+00,1.40367223e+00, 1.40646635e+00, 1.40883205e+00, 1.41076861e+00,1.41227543e+00, 1.41335206e+00, 1.41399817e+00, 1.41421356e+00,1.41399817e+00, 1.41335206e+00, 1.41227543e+00, 1.41076861e+00,1.40883205e+00, 1.40646635e+00, 1.40367223e+00, 1.40045053e+00,1.39680225e+00, 1.39272848e+00, 1.38823048e+00, 1.38330960e+00,1.37796736e+00, 1.37220538e+00, 1.36602540e+00, 1.35942933e+00,1.35241916e+00, 1.34499702e+00, 1.33716519e+00, 1.32892605e+00,1.32028210e+00, 1.31123598e+00, 1.30179045e+00, 1.29194838e+00,1.28171276e+00, 1.27108673e+00, 1.26007351e+00, 1.24867646e+00,1.23689905e+00, 1.22474487e+00, 1.21221762e+00, 1.19932112e+00,1.18605929e+00, 1.17243618e+00, 1.15845593e+00, 1.14412281e+00,1.12944117e+00, 1.11441550e+00, 1.09905036e+00, 1.08335044e+00,1.06732052e+00, 1.05096549e+00, 1.03429032e+00, 1.01730010e+00]
                     ])
    term1 = np.row_stack((term1, term1))

    #tt2 = []
    #tt2.append(sin_p/ss)
    #tt2.append(cos_p/ ss)
    #tt2.append(sin_minus_cos/ ss)
    #tt2.append(sin_plus_cos/ ss)
    #tt2.extend([-tt2[0], -tt2[1], -tt2[2], -tt2[3]])

    tt2 = np.array([
        [ 0.00000000e+00,  1.74524064e-02,  3.48994967e-02,  5.23359562e-02, 6.97564737e-02,  8.71557427e-02,  1.04528463e-01,  1.21869343e-01, 1.39173101e-01,  1.56434465e-01,  1.73648178e-01,  1.90808995e-01, 2.07911691e-01,  2.24951054e-01,  2.41921896e-01,  2.58819045e-01, 2.75637356e-01,  2.92371705e-01,  3.09016994e-01,  3.25568154e-01, 3.42020143e-01,  3.58367950e-01,  3.74606593e-01,  3.90731128e-01, 4.06736643e-01,  4.22618262e-01,  4.38371147e-01,  4.53990500e-01, 4.69471563e-01,  4.84809620e-01,  5.00000000e-01,  5.15038075e-01, 5.29919264e-01,  5.44639035e-01,  5.59192903e-01,  5.73576436e-01, 5.87785252e-01,  6.01815023e-01,  6.15661475e-01,  6.29320391e-01, 6.42787610e-01,  6.56059029e-01,  6.69130606e-01,  6.81998360e-01, 6.94658370e-01,  7.07106781e-01,  7.19339800e-01,  7.31353702e-01, 7.43144825e-01,  7.54709580e-01,  7.66044443e-01,  7.77145961e-01, 7.88010754e-01,  7.98635510e-01,  8.09016994e-01,  8.19152044e-01, 8.29037573e-01,  8.38670568e-01,  8.48048096e-01,  8.57167301e-01, 8.66025404e-01,  8.74619707e-01,  8.82947593e-01,  8.91006524e-01, 8.98794046e-01,  9.06307787e-01,  9.13545458e-01,  9.20504853e-01, 9.27183855e-01,  9.33580426e-01,  9.39692621e-01,  9.45518576e-01, 9.51056516e-01,  9.56304756e-01,  9.61261696e-01,  9.65925826e-01, 9.70295726e-01,  9.74370065e-01,  9.78147601e-01,  9.81627183e-01, 9.84807753e-01,  9.87688341e-01,  9.90268069e-01,  9.92546152e-01, 9.94521895e-01,  9.96194698e-01,  9.97564050e-01,  9.98629535e-01, 9.99390827e-01,  9.99847695e-01,  1.00000000e+00,  9.99847695e-01, 9.99390827e-01,  9.98629535e-01,  9.97564050e-01,  9.96194698e-01, 9.94521895e-01,  9.92546152e-01,  9.90268069e-01,  9.87688341e-01, 9.84807753e-01,  9.81627183e-01,  9.78147601e-01,  9.74370065e-01, 9.70295726e-01,  9.65925826e-01,  9.61261696e-01,  9.56304756e-01, 9.51056516e-01,  9.45518576e-01,  9.39692621e-01,  9.33580426e-01, 9.27183855e-01,  9.20504853e-01,  9.13545458e-01,  9.06307787e-01, 8.98794046e-01,  8.91006524e-01,  8.82947593e-01,  8.74619707e-01, 8.66025404e-01,  8.57167301e-01,  8.48048096e-01,  8.38670568e-01, 8.29037573e-01,  8.19152044e-01,  8.09016994e-01,  7.98635510e-01, 7.88010754e-01,  7.77145961e-01,  7.66044443e-01,  7.54709580e-01, 7.43144825e-01,  7.31353702e-01,  7.19339800e-01,  7.07106781e-01, 6.94658370e-01,  6.81998360e-01,  6.69130606e-01,  6.56059029e-01, 6.42787610e-01,  6.29320391e-01,  6.15661475e-01,  6.01815023e-01, 5.87785252e-01,  5.73576436e-01,  5.59192903e-01,  5.44639035e-01, 5.29919264e-01,  5.15038075e-01,  5.00000000e-01,  4.84809620e-01, 4.69471563e-01,  4.53990500e-01,  4.38371147e-01,  4.22618262e-01, 4.06736643e-01,  3.90731128e-01,  3.74606593e-01,  3.58367950e-01, 3.42020143e-01,  3.25568154e-01,  3.09016994e-01,  2.92371705e-01, 2.75637356e-01,  2.58819045e-01,  2.41921896e-01,  2.24951054e-01, 2.07911691e-01,  1.90808995e-01,  1.73648178e-01,  1.56434465e-01, 1.39173101e-01,  1.21869343e-01,  1.04528463e-01,  8.71557427e-02, 6.97564737e-02,  5.23359562e-02,  3.48994967e-02,  1.74524064e-02, 1.22464680e-16, -1.74524064e-02, -3.48994967e-02, -5.23359562e-02,-6.97564737e-02, -8.71557427e-02, -1.04528463e-01, -1.21869343e-01,-1.39173101e-01, -1.56434465e-01, -1.73648178e-01, -1.90808995e-01,-2.07911691e-01, -2.24951054e-01, -2.41921896e-01, -2.58819045e-01,-2.75637356e-01, -2.92371705e-01, -3.09016994e-01, -3.25568154e-01,-3.42020143e-01, -3.58367950e-01, -3.74606593e-01, -3.90731128e-01,-4.06736643e-01, -4.22618262e-01, -4.38371147e-01, -4.53990500e-01,-4.69471563e-01, -4.84809620e-01, -5.00000000e-01, -5.15038075e-01,-5.29919264e-01, -5.44639035e-01, -5.59192903e-01, -5.73576436e-01,-5.87785252e-01, -6.01815023e-01, -6.15661475e-01, -6.29320391e-01,-6.42787610e-01, -6.56059029e-01, -6.69130606e-01, -6.81998360e-01,-6.94658370e-01, -7.07106781e-01, -7.19339800e-01, -7.31353702e-01,-7.43144825e-01, -7.54709580e-01, -7.66044443e-01, -7.77145961e-01,-7.88010754e-01, -7.98635510e-01, -8.09016994e-01, -8.19152044e-01,-8.29037573e-01, -8.38670568e-01, -8.48048096e-01, -8.57167301e-01,-8.66025404e-01, -8.74619707e-01, -8.82947593e-01, -8.91006524e-01,-8.98794046e-01, -9.06307787e-01, -9.13545458e-01, -9.20504853e-01,-9.27183855e-01, -9.33580426e-01, -9.39692621e-01, -9.45518576e-01,-9.51056516e-01, -9.56304756e-01, -9.61261696e-01, -9.65925826e-01,-9.70295726e-01, -9.74370065e-01, -9.78147601e-01, -9.81627183e-01,-9.84807753e-01, -9.87688341e-01, -9.90268069e-01, -9.92546152e-01,-9.94521895e-01, -9.96194698e-01, -9.97564050e-01, -9.98629535e-01,-9.99390827e-01, -9.99847695e-01, -1.00000000e+00, -9.99847695e-01,-9.99390827e-01, -9.98629535e-01, -9.97564050e-01, -9.96194698e-01,-9.94521895e-01, -9.92546152e-01, -9.90268069e-01, -9.87688341e-01,-9.84807753e-01, -9.81627183e-01, -9.78147601e-01, -9.74370065e-01,-9.70295726e-01, -9.65925826e-01, -9.61261696e-01, -9.56304756e-01,-9.51056516e-01, -9.45518576e-01, -9.39692621e-01, -9.33580426e-01,-9.27183855e-01, -9.20504853e-01, -9.13545458e-01, -9.06307787e-01,-8.98794046e-01, -8.91006524e-01, -8.82947593e-01, -8.74619707e-01,-8.66025404e-01, -8.57167301e-01, -8.48048096e-01, -8.38670568e-01,-8.29037573e-01, -8.19152044e-01, -8.09016994e-01, -7.98635510e-01,-7.88010754e-01, -7.77145961e-01, -7.66044443e-01, -7.54709580e-01,-7.43144825e-01, -7.31353702e-01, -7.19339800e-01, -7.07106781e-01,-6.94658370e-01, -6.81998360e-01, -6.69130606e-01, -6.56059029e-01,-6.42787610e-01, -6.29320391e-01, -6.15661475e-01, -6.01815023e-01,-5.87785252e-01, -5.73576436e-01, -5.59192903e-01, -5.44639035e-01,-5.29919264e-01, -5.15038075e-01, -5.00000000e-01, -4.84809620e-01,-4.69471563e-01, -4.53990500e-01, -4.38371147e-01, -4.22618262e-01,-4.06736643e-01, -3.90731128e-01, -3.74606593e-01, -3.58367950e-01,-3.42020143e-01, -3.25568154e-01, -3.09016994e-01, -2.92371705e-01,-2.75637356e-01, -2.58819045e-01, -2.41921896e-01, -2.24951054e-01,-2.07911691e-01, -1.90808995e-01, -1.73648178e-01, -1.56434465e-01,-1.39173101e-01, -1.21869343e-01, -1.04528463e-01, -8.71557427e-02,-6.97564737e-02, -5.23359562e-02, -3.48994967e-02, -1.74524064e-02],
        [1.00000000e+00, 9.99847695e-01, 9.99390827e-01, 9.98629535e-01,9.97564050e-01, 9.96194698e-01, 9.94521895e-01, 9.92546152e-01,9.90268069e-01, 9.87688341e-01, 9.84807753e-01, 9.81627183e-01,9.78147601e-01, 9.74370065e-01, 9.70295726e-01, 9.65925826e-01,9.61261696e-01, 9.56304756e-01, 9.51056516e-01, 9.45518576e-01,9.39692621e-01, 9.33580426e-01, 9.27183855e-01, 9.20504853e-01,9.13545458e-01, 9.06307787e-01, 8.98794046e-01, 8.91006524e-01,8.82947593e-01, 8.74619707e-01, 8.66025404e-01, 8.57167301e-01,8.48048096e-01, 8.38670568e-01, 8.29037573e-01, 8.19152044e-01,8.09016994e-01, 7.98635510e-01, 7.88010754e-01, 7.77145961e-01,7.66044443e-01, 7.54709580e-01, 7.43144825e-01, 7.31353702e-01,7.19339800e-01, 7.07106781e-01, 6.94658370e-01, 6.81998360e-01,6.69130606e-01, 6.56059029e-01, 6.42787610e-01, 6.29320391e-01,6.15661475e-01, 6.01815023e-01, 5.87785252e-01, 5.73576436e-01,5.59192903e-01, 5.44639035e-01, 5.29919264e-01, 5.15038075e-01,5.00000000e-01, 4.84809620e-01, 4.69471563e-01, 4.53990500e-01,4.38371147e-01, 4.22618262e-01, 4.06736643e-01, 3.90731128e-01,3.74606593e-01, 3.58367950e-01, 3.42020143e-01, 3.25568154e-01,3.09016994e-01, 2.92371705e-01, 2.75637356e-01, 2.58819045e-01,2.41921896e-01, 2.24951054e-01, 2.07911691e-01, 1.90808995e-01,1.73648178e-01, 1.56434465e-01, 1.39173101e-01, 1.21869343e-01,1.04528463e-01, 8.71557427e-02, 6.97564737e-02, 5.23359562e-02,3.48994967e-02, 1.74524064e-02, 6.12323400e-17, -1.74524064e-02,-3.48994967e-02, -5.23359562e-02, -6.97564737e-02, -8.71557427e-02,-1.04528463e-01, -1.21869343e-01, -1.39173101e-01, -1.56434465e-01,-1.73648178e-01, -1.90808995e-01, -2.07911691e-01, -2.24951054e-01,-2.41921896e-01, -2.58819045e-01, -2.75637356e-01, -2.92371705e-01,-3.09016994e-01, -3.25568154e-01, -3.42020143e-01, -3.58367950e-01,-3.74606593e-01, -3.90731128e-01, -4.06736643e-01, -4.22618262e-01,-4.38371147e-01, -4.53990500e-01, -4.69471563e-01, -4.84809620e-01,-5.00000000e-01, -5.15038075e-01, -5.29919264e-01, -5.44639035e-01,-5.59192903e-01, -5.73576436e-01, -5.87785252e-01, -6.01815023e-01,-6.15661475e-01, -6.29320391e-01, -6.42787610e-01, -6.56059029e-01,-6.69130606e-01, -6.81998360e-01, -6.94658370e-01, -7.07106781e-01,-7.19339800e-01, -7.31353702e-01, -7.43144825e-01, -7.54709580e-01,-7.66044443e-01, -7.77145961e-01, -7.88010754e-01, -7.98635510e-01,-8.09016994e-01, -8.19152044e-01, -8.29037573e-01, -8.38670568e-01,-8.48048096e-01, -8.57167301e-01, -8.66025404e-01, -8.74619707e-01,-8.82947593e-01, -8.91006524e-01, -8.98794046e-01, -9.06307787e-01,-9.13545458e-01, -9.20504853e-01, -9.27183855e-01, -9.33580426e-01,-9.39692621e-01, -9.45518576e-01, -9.51056516e-01, -9.56304756e-01,-9.61261696e-01, -9.65925826e-01, -9.70295726e-01, -9.74370065e-01,-9.78147601e-01, -9.81627183e-01, -9.84807753e-01, -9.87688341e-01,-9.90268069e-01, -9.92546152e-01, -9.94521895e-01, -9.96194698e-01,-9.97564050e-01, -9.98629535e-01, -9.99390827e-01, -9.99847695e-01,-1.00000000e+00, -9.99847695e-01, -9.99390827e-01, -9.98629535e-01,-9.97564050e-01, -9.96194698e-01, -9.94521895e-01, -9.92546152e-01,-9.90268069e-01, -9.87688341e-01, -9.84807753e-01, -9.81627183e-01,-9.78147601e-01, -9.74370065e-01, -9.70295726e-01, -9.65925826e-01,-9.61261696e-01, -9.56304756e-01, -9.51056516e-01, -9.45518576e-01,-9.39692621e-01, -9.33580426e-01, -9.27183855e-01, -9.20504853e-01,-9.13545458e-01, -9.06307787e-01, -8.98794046e-01, -8.91006524e-01,-8.82947593e-01, -8.74619707e-01, -8.66025404e-01, -8.57167301e-01,-8.48048096e-01, -8.38670568e-01, -8.29037573e-01, -8.19152044e-01,-8.09016994e-01, -7.98635510e-01, -7.88010754e-01, -7.77145961e-01,-7.66044443e-01, -7.54709580e-01, -7.43144825e-01, -7.31353702e-01,-7.19339800e-01, -7.07106781e-01, -6.94658370e-01, -6.81998360e-01,-6.69130606e-01, -6.56059029e-01, -6.42787610e-01, -6.29320391e-01,-6.15661475e-01, -6.01815023e-01, -5.87785252e-01, -5.73576436e-01,-5.59192903e-01, -5.44639035e-01, -5.29919264e-01, -5.15038075e-01,-5.00000000e-01, -4.84809620e-01, -4.69471563e-01, -4.53990500e-01,-4.38371147e-01, -4.22618262e-01, -4.06736643e-01, -3.90731128e-01,-3.74606593e-01, -3.58367950e-01, -3.42020143e-01, -3.25568154e-01,-3.09016994e-01, -2.92371705e-01, -2.75637356e-01, -2.58819045e-01,-2.41921896e-01, -2.24951054e-01, -2.07911691e-01, -1.90808995e-01,-1.73648178e-01, -1.56434465e-01, -1.39173101e-01, -1.21869343e-01,-1.04528463e-01, -8.71557427e-02, -6.97564737e-02, -5.23359562e-02,-3.48994967e-02, -1.74524064e-02, -1.83697020e-16, 1.74524064e-02,3.48994967e-02, 5.23359562e-02, 6.97564737e-02, 8.71557427e-02,1.04528463e-01, 1.21869343e-01, 1.39173101e-01, 1.56434465e-01,1.73648178e-01, 1.90808995e-01, 2.07911691e-01, 2.24951054e-01,2.41921896e-01, 2.58819045e-01, 2.75637356e-01, 2.92371705e-01,3.09016994e-01, 3.25568154e-01, 3.42020143e-01, 3.58367950e-01,3.74606593e-01, 3.90731128e-01, 4.06736643e-01, 4.22618262e-01,4.38371147e-01, 4.53990500e-01, 4.69471563e-01, 4.84809620e-01,5.00000000e-01, 5.15038075e-01, 5.29919264e-01, 5.44639035e-01,5.59192903e-01, 5.73576436e-01, 5.87785252e-01, 6.01815023e-01,6.15661475e-01, 6.29320391e-01, 6.42787610e-01, 6.56059029e-01,6.69130606e-01, 6.81998360e-01, 6.94658370e-01, 7.07106781e-01,7.19339800e-01, 7.31353702e-01, 7.43144825e-01, 7.54709580e-01,7.66044443e-01, 7.77145961e-01, 7.88010754e-01, 7.98635510e-01,8.09016994e-01, 8.19152044e-01, 8.29037573e-01, 8.38670568e-01,8.48048096e-01, 8.57167301e-01, 8.66025404e-01, 8.74619707e-01,8.82947593e-01, 8.91006524e-01, 8.98794046e-01, 9.06307787e-01,9.13545458e-01, 9.20504853e-01, 9.27183855e-01, 9.33580426e-01,9.39692621e-01, 9.45518576e-01, 9.51056516e-01, 9.56304756e-01,9.61261696e-01, 9.65925826e-01, 9.70295726e-01, 9.74370065e-01,9.78147601e-01, 9.81627183e-01, 9.84807753e-01, 9.87688341e-01,9.90268069e-01, 9.92546152e-01, 9.94521895e-01, 9.96194698e-01,9.97564050e-01, 9.98629535e-01, 9.99390827e-01, 9.99847695e-01],
        [-1.00000000e+00, -9.82395289e-01, -9.64491330e-01, -9.46293579e-01,-9.27807577e-01, -9.09038955e-01, -8.89993432e-01, -8.70676808e-01,-8.51094968e-01, -8.31253876e-01, -8.11159575e-01, -7.90818188e-01,-7.70235910e-01, -7.49419010e-01, -7.28373831e-01, -7.07106781e-01,-6.85624340e-01, -6.63933051e-01, -6.42039522e-01, -6.19950421e-01,-5.97672477e-01, -5.75212477e-01, -5.52577261e-01, -5.29773725e-01,-5.06808815e-01, -4.83689525e-01, -4.60422900e-01, -4.37016024e-01,-4.13476030e-01, -3.89810087e-01, -3.66025404e-01, -3.42129226e-01,-3.18128832e-01, -2.94031533e-01, -2.69844669e-01, -2.45575608e-01,-2.21231742e-01, -1.96820487e-01, -1.72349278e-01, -1.47825570e-01,-1.23256833e-01, -9.86505512e-02, -7.40142191e-02, -4.93553416e-02,-2.46814299e-02, -1.11022302e-16, 2.46814299e-02, 4.93553416e-02,7.40142191e-02, 9.86505512e-02, 1.23256833e-01, 1.47825570e-01,1.72349278e-01, 1.96820487e-01, 2.21231742e-01, 2.45575608e-01,2.69844669e-01, 2.94031533e-01, 3.18128832e-01, 3.42129226e-01,3.66025404e-01, 3.89810087e-01, 4.13476030e-01, 4.37016024e-01,4.60422900e-01, 4.83689525e-01, 5.06808815e-01, 5.29773725e-01,5.52577261e-01, 5.75212477e-01, 5.97672477e-01, 6.19950421e-01,6.42039522e-01, 6.63933051e-01, 6.85624340e-01, 7.07106781e-01,7.28373831e-01, 7.49419010e-01, 7.70235910e-01, 7.90818188e-01,8.11159575e-01, 8.31253876e-01, 8.51094968e-01, 8.70676808e-01,8.89993432e-01, 9.09038955e-01, 9.27807577e-01, 9.46293579e-01,9.64491330e-01, 9.82395289e-01, 1.00000000e+00, 1.01730010e+00,1.03429032e+00, 1.05096549e+00, 1.06732052e+00, 1.08335044e+00,1.09905036e+00, 1.11441550e+00, 1.12944117e+00, 1.14412281e+00,1.15845593e+00, 1.17243618e+00, 1.18605929e+00, 1.19932112e+00,1.21221762e+00, 1.22474487e+00, 1.23689905e+00, 1.24867646e+00,1.26007351e+00, 1.27108673e+00, 1.28171276e+00, 1.29194838e+00,1.30179045e+00, 1.31123598e+00, 1.32028210e+00, 1.32892605e+00,1.33716519e+00, 1.34499702e+00, 1.35241916e+00, 1.35942933e+00,1.36602540e+00, 1.37220538e+00, 1.37796736e+00, 1.38330960e+00,1.38823048e+00, 1.39272848e+00, 1.39680225e+00, 1.40045053e+00,1.40367223e+00, 1.40646635e+00, 1.40883205e+00, 1.41076861e+00,1.41227543e+00, 1.41335206e+00, 1.41399817e+00, 1.41421356e+00,1.41399817e+00, 1.41335206e+00, 1.41227543e+00, 1.41076861e+00,1.40883205e+00, 1.40646635e+00, 1.40367223e+00, 1.40045053e+00,1.39680225e+00, 1.39272848e+00, 1.38823048e+00, 1.38330960e+00,1.37796736e+00, 1.37220538e+00, 1.36602540e+00, 1.35942933e+00,1.35241916e+00, 1.34499702e+00, 1.33716519e+00, 1.32892605e+00,1.32028210e+00, 1.31123598e+00, 1.30179045e+00, 1.29194838e+00,1.28171276e+00, 1.27108673e+00, 1.26007351e+00, 1.24867646e+00,1.23689905e+00, 1.22474487e+00, 1.21221762e+00, 1.19932112e+00,1.18605929e+00, 1.17243618e+00, 1.15845593e+00, 1.14412281e+00,1.12944117e+00, 1.11441550e+00, 1.09905036e+00, 1.08335044e+00,1.06732052e+00, 1.05096549e+00, 1.03429032e+00, 1.01730010e+00,1.00000000e+00, 9.82395289e-01, 9.64491330e-01, 9.46293579e-01,9.27807577e-01, 9.09038955e-01, 8.89993432e-01, 8.70676808e-01,8.51094968e-01, 8.31253876e-01, 8.11159575e-01, 7.90818188e-01,7.70235910e-01, 7.49419010e-01, 7.28373831e-01, 7.07106781e-01,6.85624340e-01, 6.63933051e-01, 6.42039522e-01, 6.19950421e-01,5.97672477e-01, 5.75212477e-01, 5.52577261e-01, 5.29773725e-01,5.06808815e-01, 4.83689525e-01, 4.60422900e-01, 4.37016024e-01,4.13476030e-01, 3.89810087e-01, 3.66025404e-01, 3.42129226e-01,3.18128832e-01, 2.94031533e-01, 2.69844669e-01, 2.45575608e-01,2.21231742e-01, 1.96820487e-01, 1.72349278e-01, 1.47825570e-01,1.23256833e-01, 9.86505512e-02, 7.40142191e-02, 4.93553416e-02,2.46814299e-02, 2.22044605e-16, -2.46814299e-02, -4.93553416e-02,-7.40142191e-02, -9.86505512e-02, -1.23256833e-01, -1.47825570e-01,-1.72349278e-01, -1.96820487e-01, -2.21231742e-01, -2.45575608e-01,-2.69844669e-01, -2.94031533e-01, -3.18128832e-01, -3.42129226e-01,-3.66025404e-01, -3.89810087e-01, -4.13476030e-01, -4.37016024e-01,-4.60422900e-01, -4.83689525e-01, -5.06808815e-01, -5.29773725e-01,-5.52577261e-01, -5.75212477e-01, -5.97672477e-01, -6.19950421e-01,-6.42039522e-01, -6.63933051e-01, -6.85624340e-01, -7.07106781e-01,-7.28373831e-01, -7.49419010e-01, -7.70235910e-01, -7.90818188e-01,-8.11159575e-01, -8.31253876e-01, -8.51094968e-01, -8.70676808e-01,-8.89993432e-01, -9.09038955e-01, -9.27807577e-01, -9.46293579e-01,-9.64491330e-01, -9.82395289e-01, -1.00000000e+00, -1.01730010e+00,-1.03429032e+00, -1.05096549e+00, -1.06732052e+00, -1.08335044e+00,-1.09905036e+00, -1.11441550e+00, -1.12944117e+00, -1.14412281e+00,-1.15845593e+00, -1.17243618e+00, -1.18605929e+00, -1.19932112e+00,-1.21221762e+00, -1.22474487e+00, -1.23689905e+00, -1.24867646e+00,-1.26007351e+00, -1.27108673e+00, -1.28171276e+00, -1.29194838e+00,-1.30179045e+00, -1.31123598e+00, -1.32028210e+00, -1.32892605e+00,-1.33716519e+00, -1.34499702e+00, -1.35241916e+00, -1.35942933e+00,-1.36602540e+00, -1.37220538e+00, -1.37796736e+00, -1.38330960e+00,-1.38823048e+00, -1.39272848e+00, -1.39680225e+00, -1.40045053e+00,-1.40367223e+00, -1.40646635e+00, -1.40883205e+00, -1.41076861e+00,-1.41227543e+00, -1.41335206e+00, -1.41399817e+00, -1.41421356e+00,-1.41399817e+00, -1.41335206e+00, -1.41227543e+00, -1.41076861e+00,-1.40883205e+00, -1.40646635e+00, -1.40367223e+00, -1.40045053e+00,-1.39680225e+00, -1.39272848e+00, -1.38823048e+00, -1.38330960e+00,-1.37796736e+00, -1.37220538e+00, -1.36602540e+00, -1.35942933e+00,-1.35241916e+00, -1.34499702e+00, -1.33716519e+00, -1.32892605e+00,-1.32028210e+00, -1.31123598e+00, -1.30179045e+00, -1.29194838e+00,-1.28171276e+00, -1.27108673e+00, -1.26007351e+00, -1.24867646e+00,-1.23689905e+00, -1.22474487e+00, -1.21221762e+00, -1.19932112e+00,-1.18605929e+00, -1.17243618e+00, -1.15845593e+00, -1.14412281e+00,-1.12944117e+00, -1.11441550e+00, -1.09905036e+00, -1.08335044e+00,-1.06732052e+00, -1.05096549e+00, -1.03429032e+00, -1.01730010e+00],
        [1.00000000e+00, 1.01730010e+00, 1.03429032e+00, 1.05096549e+00,1.06732052e+00, 1.08335044e+00, 1.09905036e+00, 1.11441550e+00,1.12944117e+00, 1.14412281e+00, 1.15845593e+00, 1.17243618e+00,1.18605929e+00, 1.19932112e+00, 1.21221762e+00, 1.22474487e+00,1.23689905e+00, 1.24867646e+00, 1.26007351e+00, 1.27108673e+00,1.28171276e+00, 1.29194838e+00, 1.30179045e+00, 1.31123598e+00,1.32028210e+00, 1.32892605e+00, 1.33716519e+00, 1.34499702e+00,1.35241916e+00, 1.35942933e+00, 1.36602540e+00, 1.37220538e+00,1.37796736e+00, 1.38330960e+00, 1.38823048e+00, 1.39272848e+00,1.39680225e+00, 1.40045053e+00, 1.40367223e+00, 1.40646635e+00,1.40883205e+00, 1.41076861e+00, 1.41227543e+00, 1.41335206e+00,1.41399817e+00, 1.41421356e+00, 1.41399817e+00, 1.41335206e+00,1.41227543e+00, 1.41076861e+00, 1.40883205e+00, 1.40646635e+00,1.40367223e+00, 1.40045053e+00, 1.39680225e+00, 1.39272848e+00,1.38823048e+00, 1.38330960e+00, 1.37796736e+00, 1.37220538e+00,1.36602540e+00, 1.35942933e+00, 1.35241916e+00, 1.34499702e+00,1.33716519e+00, 1.32892605e+00, 1.32028210e+00, 1.31123598e+00,1.30179045e+00, 1.29194838e+00, 1.28171276e+00, 1.27108673e+00,1.26007351e+00, 1.24867646e+00, 1.23689905e+00, 1.22474487e+00,1.21221762e+00, 1.19932112e+00, 1.18605929e+00, 1.17243618e+00,1.15845593e+00, 1.14412281e+00, 1.12944117e+00, 1.11441550e+00,1.09905036e+00, 1.08335044e+00, 1.06732052e+00, 1.05096549e+00,1.03429032e+00, 1.01730010e+00, 1.00000000e+00, 9.82395289e-01,9.64491330e-01, 9.46293579e-01, 9.27807577e-01, 9.09038955e-01,8.89993432e-01, 8.70676808e-01, 8.51094968e-01, 8.31253876e-01,8.11159575e-01, 7.90818188e-01, 7.70235910e-01, 7.49419010e-01,7.28373831e-01, 7.07106781e-01, 6.85624340e-01, 6.63933051e-01,6.42039522e-01, 6.19950421e-01, 5.97672477e-01, 5.75212477e-01,5.52577261e-01, 5.29773725e-01, 5.06808815e-01, 4.83689525e-01,4.60422900e-01, 4.37016024e-01, 4.13476030e-01, 3.89810087e-01,3.66025404e-01, 3.42129226e-01, 3.18128832e-01, 2.94031533e-01,2.69844669e-01, 2.45575608e-01, 2.21231742e-01, 1.96820487e-01,1.72349278e-01, 1.47825570e-01, 1.23256833e-01, 9.86505512e-02,7.40142191e-02, 4.93553416e-02, 2.46814299e-02, 1.11022302e-16,-2.46814299e-02, -4.93553416e-02, -7.40142191e-02, -9.86505512e-02,-1.23256833e-01, -1.47825570e-01, -1.72349278e-01, -1.96820487e-01,-2.21231742e-01, -2.45575608e-01, -2.69844669e-01, -2.94031533e-01,-3.18128832e-01, -3.42129226e-01, -3.66025404e-01, -3.89810087e-01,-4.13476030e-01, -4.37016024e-01, -4.60422900e-01, -4.83689525e-01,-5.06808815e-01, -5.29773725e-01, -5.52577261e-01, -5.75212477e-01,-5.97672477e-01, -6.19950421e-01, -6.42039522e-01, -6.63933051e-01,-6.85624340e-01, -7.07106781e-01, -7.28373831e-01, -7.49419010e-01,-7.70235910e-01, -7.90818188e-01, -8.11159575e-01, -8.31253876e-01,-8.51094968e-01, -8.70676808e-01, -8.89993432e-01, -9.09038955e-01,-9.27807577e-01, -9.46293579e-01, -9.64491330e-01, -9.82395289e-01,-1.00000000e+00, -1.01730010e+00, -1.03429032e+00, -1.05096549e+00,-1.06732052e+00, -1.08335044e+00, -1.09905036e+00, -1.11441550e+00,-1.12944117e+00, -1.14412281e+00, -1.15845593e+00, -1.17243618e+00,-1.18605929e+00, -1.19932112e+00, -1.21221762e+00, -1.22474487e+00,-1.23689905e+00, -1.24867646e+00, -1.26007351e+00, -1.27108673e+00,-1.28171276e+00, -1.29194838e+00, -1.30179045e+00, -1.31123598e+00,-1.32028210e+00, -1.32892605e+00, -1.33716519e+00, -1.34499702e+00,-1.35241916e+00, -1.35942933e+00, -1.36602540e+00, -1.37220538e+00,-1.37796736e+00, -1.38330960e+00, -1.38823048e+00, -1.39272848e+00,-1.39680225e+00, -1.40045053e+00, -1.40367223e+00, -1.40646635e+00,-1.40883205e+00, -1.41076861e+00, -1.41227543e+00, -1.41335206e+00,-1.41399817e+00, -1.41421356e+00, -1.41399817e+00, -1.41335206e+00,-1.41227543e+00, -1.41076861e+00, -1.40883205e+00, -1.40646635e+00,-1.40367223e+00, -1.40045053e+00, -1.39680225e+00, -1.39272848e+00,-1.38823048e+00, -1.38330960e+00, -1.37796736e+00, -1.37220538e+00,-1.36602540e+00, -1.35942933e+00, -1.35241916e+00, -1.34499702e+00,-1.33716519e+00, -1.32892605e+00, -1.32028210e+00, -1.31123598e+00,-1.30179045e+00, -1.29194838e+00, -1.28171276e+00, -1.27108673e+00,-1.26007351e+00, -1.24867646e+00, -1.23689905e+00, -1.22474487e+00,-1.21221762e+00, -1.19932112e+00, -1.18605929e+00, -1.17243618e+00,-1.15845593e+00, -1.14412281e+00, -1.12944117e+00, -1.11441550e+00,-1.09905036e+00, -1.08335044e+00, -1.06732052e+00, -1.05096549e+00,-1.03429032e+00, -1.01730010e+00, -1.00000000e+00, -9.82395289e-01,-9.64491330e-01, -9.46293579e-01, -9.27807577e-01, -9.09038955e-01,-8.89993432e-01, -8.70676808e-01, -8.51094968e-01, -8.31253876e-01,-8.11159575e-01, -7.90818188e-01, -7.70235910e-01, -7.49419010e-01,-7.28373831e-01, -7.07106781e-01, -6.85624340e-01, -6.63933051e-01,-6.42039522e-01, -6.19950421e-01, -5.97672477e-01, -5.75212477e-01,-5.52577261e-01, -5.29773725e-01, -5.06808815e-01, -4.83689525e-01,-4.60422900e-01, -4.37016024e-01, -4.13476030e-01, -3.89810087e-01,-3.66025404e-01, -3.42129226e-01, -3.18128832e-01, -2.94031533e-01,-2.69844669e-01, -2.45575608e-01, -2.21231742e-01, -1.96820487e-01,-1.72349278e-01, -1.47825570e-01, -1.23256833e-01, -9.86505512e-02,-7.40142191e-02, -4.93553416e-02, -2.46814299e-02, -3.33066907e-16,2.46814299e-02, 4.93553416e-02, 7.40142191e-02, 9.86505512e-02,1.23256833e-01, 1.47825570e-01, 1.72349278e-01, 1.96820487e-01,2.21231742e-01, 2.45575608e-01, 2.69844669e-01, 2.94031533e-01,3.18128832e-01, 3.42129226e-01, 3.66025404e-01, 3.89810087e-01,4.13476030e-01, 4.37016024e-01, 4.60422900e-01, 4.83689525e-01,5.06808815e-01, 5.29773725e-01, 5.52577261e-01, 5.75212477e-01,5.97672477e-01, 6.19950421e-01, 6.42039522e-01, 6.63933051e-01,6.85624340e-01, 7.07106781e-01, 7.28373831e-01, 7.49419010e-01,7.70235910e-01, 7.90818188e-01, 8.11159575e-01, 8.31253876e-01,8.51094968e-01, 8.70676808e-01, 8.89993432e-01, 9.09038955e-01,9.27807577e-01, 9.46293579e-01, 9.64491330e-01, 9.82395289e-01]
    ])
    tt2 = tt2/ss
    tt2 = np.row_stack((tt2, -tt2))
    term2 = ss*(1-tt2+tt2**2)

    #term2 = []
    #for t2_item in tt2:
    #    term2.append(ss * (1 - t2_item + t2_item ** 2))

    #case_value = []
    #for c_i in range(8):
    #    case_value.append((1/ ss ** 2) * term1[c_i%4,:] * term2[c_i])

    case_value = (1 / ss ** 2) * term1 * term2

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
    edge_segs = np.row_stack([((e[0], e[1]), (e[-1], e[-2])) for e in segs])

    # subtract neighboring segment vertex from edge point to get vector pointing
    # in direction of edge
    # we also get the midpoints of the last segment,
    # because pointing towards the actual last point in the segment can be noisy
    edge_vects = normalize(edge_segs[:, 0, :] - edge_segs[:, 1, :])
    edge_mids = np.mean(edge_segs, axis=1)

    # cosine distance between direction of edge at endpoints and direction to other points
    # note this distance is asymmetrical, affinity from point in row to point in column
    affinity_metric = np.row_stack(
        [distance.cdist(edge_vects[i, :].reshape(1, 2), edge_mids - edge_points[i, :], "cosine")
         for i in range(len(edge_segs))]
    )

    # get upper/lower indices manually so they match up right
    triu_indices = np.triu_indices(affinity_metric.shape[0], k=1)
    tril_indices = triu_indices[::-1]

    # average top and bottom triangles - both edges should point at each other
    pair_affinity = np.mean((affinity_metric[tril_indices],
                             affinity_metric[triu_indices]), axis=0)
    affinity_metric[tril_indices] = pair_affinity
    affinity_metric[triu_indices] = pair_affinity

    # Clean up & combine merge metrics
    # want high values to be good, and for vals to be 0-1
    dist_metric = 1 - ((dist_metric - np.min(dist_metric)) / (np.max(dist_metric) - np.min(dist_metric)))
    # dist_metric = 1-dist_metric
    grad_metric = 1 - (grad_metric / 2.)  # cosine distance is 0-2
    affinity_metric = 1 - (affinity_metric / 2.)

    merge_score = dist_metric ** 2 * grad_metric ** 2 * affinity_metric ** 2
    merge_score[np.isnan(merge_score)] = 0.
    merge_score[triu_indices] = 0

    return merge_score