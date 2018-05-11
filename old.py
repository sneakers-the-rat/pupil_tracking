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

def kmeans_img(img, n_clusters, k_criteria):
    img_shape = img.shape
    img = img.reshape(-1, 1)
    img = np.float32(img)
    ret, label, center = cv2.kmeans(img, n_clusters, None, k_criteria, 16, cv2.KMEANS_RANDOM_CENTERS)
    res = center[label]
    res = res.reshape(img_shape)
    return res


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


def scharr_canny_new(image, sigma, low_threshold=0.2, high_threshold=0.5):
    # skimage's canny but we get scharr grads instead of sobel,
    # and use the eigenvalues of the structure tensor rather than the hypotenuse

    isobel = filters.gaussian(cv2.Scharr(image, ddepth=-1, dx=0, dy=1), sigma=sigma)
    jsobel = filters.gaussian(cv2.Scharr(image, ddepth=-1, dx=1, dy=0), sigma=sigma)


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
    mask = np.zeros(image.shape, dtype=np.bool)
    mask[1:-1, 1:-1] = True
    mask_x, mask_y = np.where(mask)
    #s = generate_binary_structure(2, 2)
    #eroded_mask = binary_erosion(mask, s, border_value=0)
    #eroded_mask = eroded_mask & (magnitude > 0)
    #
    #--------- Find local maxima --------------
    local_maxima = np.zeros(image.shape, bool)

    # make mask arrays to use as additive slicers on magnitude array
    # want to see if pixels +/- the direction of the gradient are less than us
    # two arrays will be the perpendicular pixels [-1,0], [0,1], etc.
    # and the other two will be the diags [-1,-1], etc.
    ij_sign = np.abs(isobel) >= np.abs(jsobel)
    ij_mag = np.stack((np.abs(isobel), np.abs(jsobel)))
    diagonality = (np.min(ij_mag, axis=0)/np.max(ij_mag, axis=0))[mask_x, mask_y]

    isob_sign = np.sign(isobel, casting="unsafe", dtype=np.int8)
    jsob_sign = np.sign(jsobel, casting="unsafe", dtype=np.int8)

    # first compute all the diagonal neighbors - plus and minus
    # multiply by diagonality (0-horiz/vert vector, 1-perfectly diagonal)

    corner_plus_mag = diagonality*magnitude[mask_x+isob_sign[mask_x, mask_y], mask_y+jsob_sign[mask_x, mask_y]]
    corner_minus_mag = diagonality*magnitude[mask_x-isob_sign[mask_x, mask_y], mask_y-jsob_sign[mask_x, mask_y]]

    # now zero non-dominant component to get rectangular sides
    isob_sign[np.logical_not(ij_sign)] = 0
    jsob_sign[ij_sign] = 0

    side_plus_mag = (1.-diagonality)*magnitude[mask_x+isob_sign[mask_x, mask_y], mask_y+jsob_sign[mask_x, mask_y]]
    side_minus_mag = (1.-diagonality)*magnitude[mask_x-isob_sign[mask_x, mask_y], mask_y-jsob_sign[mask_x, mask_y]]


    # set local maxima
    local_maxima[mask] = np.logical_and(magnitude[mask_x, mask_y] > corner_plus_mag+side_plus_mag, magnitude[mask_x, mask_y]>corner_minus_mag+side_minus_mag)


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


def order_points_old(edge_points):
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
        #connected_pts = {k: dists[pt_i,k] for k in np.where(dists[pt_i,:])[0] if k in inds}
        connected_pts = [k for k in np.where(dists[pt_i, :])[0] if k in inds]
        print(connected_pts)

        # if we get one, cool. just append
        if len(connected_pts) == 1:
            #pt_i = connected_pts.keys()[0]
            pt_i = connected_pts[0]
            if forwards:
                new_pts.append(edge_points[inds.pop(inds.index(pt_i))])
            else:
                new_pts.appendleft(edge_points[inds.pop(inds.index(pt_i))])

        # if we get more than one, find the longest one
        elif len(connected_pts) > 1:
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
        elif len(connected_pts) == 0:
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

###########################33

# a ripoff of asyncsrv http://zguide.zeromq.org/py:all
# and http://pyzmq.readthedocs.io/en/latest/serialization.html

import zmq
import multiprocessing as mp
import numpy as np

import runops

class Client(mp.Process):
    def __init__(self, id, fxn, port=5570):
        self.id = id
        self.fxn = fxn
        self.port = port
        super(Client, self).__init__()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'worker-%d' % self.id
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:{}'.format(self.port))
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)
        reqs = 0
        while True:
            reqs = reqs + 1
            socket.send_string(u'request #%d' % (reqs))
            for i in range(5):
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    msg = socket.recv()

        socket.close()
        context.term()

class Router(mp.Process):
    """ServerTask"""
    def __init__(self, id, n_workers=1, front_port = 5570, back_port = 5580):
        super(Router, self).__init__()
        self.id = id
        self.n_workers=n_workers
        self.front_port = front_port
        self.back_port = back_port

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://localhost:{}'.format(self.front_port))

        backend = context.socket(zmq.DEALER)
        backend.bind('tcp://localhost:{}'.format(self.back_port))

        workers = []
        for i in range(self.n_workers):
            worker = Worker(i, router_port = self.back_port)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class Worker(mp.Process):
    """ServerWorker"""
    def __init__(self, id, fxn, router_port = 6680):
        super(Worker, self).__init__()
        self.id = id
        self.fxn = fxn
        self.router_port = router_port

    def run(self):
        self.context = zmq.Context()
        worker = self.context.socket(zmq.DEALER)
        worker.connect('tcp://localhost:{}'.format(self.router_port))
        while True:
            # a message will have kwargs and perhaps a numpy array
            # we first send the kwargs and then the array with a sendmore flag
            msg = worker.recv_json()
            array =




            # do work replies
            reply = self.fxn()
            for i in range(replies):
                worker.send_multipart([ident, msg])

        worker.close()


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


