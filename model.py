import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque
from pandas import ewma, ewmstd
from itertools import count


class Pupil_Model(object):

    def __init__(self, x, y, rad, st_memory=60, lt_ratio = 10, hl = 30):
        # We initialize with circle params for now because that's all we can trace on the image
        # st_memory = how many sequential frames we keep
        # lt_ratio = how many frames to skip before stashing a long-term estimate
        # (lt params are just as long as st_memory)
        # we want to use the st params to calculate the stdev for filtering points
        # and the lt params for setting constraints.
        # hl = half-life for the exponentially weighted moving average

        self.lt_ratio = lt_ratio
        self.hl = hl

        # we store the initial params to make sanity checks
        x, y, rad = float(x), float(y), float(rad)
        self.initial_params = (x, y, rad)

        # lists of deque to store the parameter history
        self.st_params = [deque(maxlen=st_memory) for i in range(5)]
        self.lt_params = [deque(maxlen=st_memory) for i in range(5)]

        # store the standard deviation used to include/exclude points
        # we initialize to 1/8 of the radius of the circle for no particular reason for the first n frames
        self.stdev_a = rad/8.
        self.stdev_b = rad/8.

        # Make our ellipse model, set, and stash initial parameters
        self.model = measure.EllipseModel()
        self.model.params = (x, y, rad, rad, 0) # x, y, a, b, theta




        # counter to keep track of frames
        self.frame_counter = count()
        self.n_frames = 0

    def update(self, points):
        self.n_frames = self.frame_counter.next()
        f_points = self.filter_points(points)




    def stash_params(self):
        # if we have hit our lt ratio...
        if self.n_frames % self.lt_ratio == 0:
            # TODO START HERE
            for param, dq in zip(self.model.params, self.st_params):
                dq.append(param)


            for param, dq in zip(self.model.params)






    def filter_points(self, points):
        # find only those points that are within a standard deviation of our model
        # with respect to https://stackoverflow.com/questions/37031356/check-if-points-are-inside-ellipse-faster-than-contains-point-method

        # get model params handy
        x, y, a, b, t = self.model.params

        # make min/max ellipse axes from standard dev and normalize
        # then pick the more restrictive values
        min_r = np.max([(a-self.stdev_a)/a, (b-self.stdev_b)/b])
        max_r = np.min([(a+self.stdev_a)/a, (b+self.stdev_b)/b])

        # split points into x and y coords
        pts_x, pts_y = points[:,0], points[:,1]

        cos_e = np.cos(t)
        sin_e = np.sin(t)

        # get distance from center of ellipse
        dist_x = pts_x - x
        dist_y = pts_y - y

        # transform so that coordinates aligned w/ major/minor ax
        tf_x = dist_x*cos_e - dist_y*sin_e
        tf_y = dist_x*sin_e + dist_y*cos_e

        # get normalized distance from center
        norm_dist = (tf_x**2/a**2) + (tf_y**2/b**2)

        good_pts = np.logical_and(norm_dist > min_r, norm_dist < max_r)

        return points[good_pts, :]


# c_array = []
# for pt in norm_dist:
#     if min_r < pt < max_r:
#         c_array.append('green')
#     else:
#         c_array.append('black')
#
# ell_patch = patches.Ellipse((x, y), a*2, b*2, angle=np.rad2deg(t), fill=False, linewidth=2)
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.add_patch(ell_patch)
# ax.scatter(pts_x, pts_y, c=c_array)
#



class Constraint(object):

    def __init__(self):
        pass

