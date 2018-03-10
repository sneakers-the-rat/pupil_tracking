import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float
from collections import deque
from pandas import ewma, ewmstd
from itertools import count


##############################
# Initialization
# TODO: Ask for file, list of files

vid_file = '/home/lab/pupil_vids/test3.avi'
vid = cv2.VideoCapture(vid_file)
ret, frame = vid.read()
frame_orig = frame.copy()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get ROIs - cropping & pupil center
roi = cv2.selectROI(frame)
frame = crop(frame, roi)
cv2.destroyAllWindows()

# Select pupil
frame_pupil = frame.copy()
drawing = False # true if mouse is pressed
ix,iy,rad = -1,-1,5 # global vars where we'll store x/y pos & radius
cv2.namedWindow('pupil')
cv2.setMouseCallback('pupil',draw_circle)
while True:
    cv2.imshow('pupil',frame_pupil)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        frame_pupil = frame.copy()
    elif k == ord('\r'):
        break
cv2.destroyAllWindows()

print(ix,iy,rad)

# Adjust preprocessing parameters
## Initial values -- have to use ints, we'll convert back later
frame_params = preprocess_image(frame_orig, roi)
edges_params = feature.canny(frame_params, sigma=5)

sig_cutoff = 50
sig_gain = 5
n_colors = 12
k_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
gauss_sig = 100
canny_sig = 5

cv2.namedWindow('params', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Sigmoid Cutoff', 'params', sig_cutoff, 100, nothing)
cv2.createTrackbar('Sigmoid Gain', 'params', sig_gain, 20, nothing)
cv2.createTrackbar('n Colors', 'params', n_colors, 16, nothing)
cv2.createTrackbar('Gaussian Sigma', 'params', gauss_sig, 200, nothing)
cv2.createTrackbar('Canny Sigma', 'params', canny_sig, 10, nothing)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()

    sig_cutoff = cv2.getTrackbarPos('Sigmoid Cutoff', 'params')
    sig_gain = cv2.getTrackbarPos('Sigmoid Gain', 'params')
    n_colors = cv2.getTrackbarPos('n Colors', 'params')
    gauss_sig = cv2.getTrackbarPos('Gaussian Sigma', 'params')
    canny_sig = cv2.getTrackbarPos('Canny Sigma', 'params')

    frame_params = preprocess_image(frame_orig, roi, sig_cutoff=sig_cutoff/100.,
                                    sig_gain=sig_gain, n_colors=n_colors,
                                    gauss_sig=gauss_sig/100.,
                                    k_criteria=k_criteria)
    edges_params = feature.canny(frame_params, sigma=canny_sig)
    edge_ret, edge_pts, ellipse, edges = process_edges(edges_params, ix, iy, rad)

    try:
    #if edge_ret:
        ell_points = np.round(ellipse.predict_xy(np.arange(0, 2 * np.pi, .1))).astype(np.int)
        edges[np.clip(ell_points[:,1], a_min=0,a_max=edges.shape[0]),
              np.clip(ell_points[:,0], a_min=0,a_max=edges.shape[1])] = np.max(edges)
        edges_params = edges
    #else:
    except:
        edges_params = feature.canny(frame_params, sigma=canny_sig)

    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_orig = crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)


    cv2.imshow('params', np.vstack([frame_orig, frame_params, edges_params]))





