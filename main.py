import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float, draw
from collections import deque
from pandas import ewma, ewmstd
from itertools import count
import matplotlib.pyplot as plt

import imutils
import model

def draw_circle(event,x,y,flags,param):
    # Draw a circle on the frame to outline the pupil
    global ix,iy,drawing,rad,frame_pupil
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            rad = np.round(euclidean([ix,iy], [x,y])).astype(np.int)
            cv2.circle(frame_pupil,(ix,iy),rad,(255,255,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(frame_pupil,(ix,iy),rad,(255,255,255),-1)

##############################
# Initialization
# TODO: Ask for file, list of files

vid_file = '/home/lab/pupil_vids/ira1.mp4'
vid = cv2.VideoCapture(vid_file)
ret, frame = vid.read()
frame_orig = frame.copy()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get ROIs - cropping & pupil center
roi = cv2.selectROI(frame)
frame = imutils.crop(frame, roi)
cv2.destroyAllWindows()

# Select pupil

global drawing, ix, iy, rad, frame_pupil
frame_pupil = frame.copy()
drawing = False # true if mouse is pressed
ix,iy,rad = -1,-1,5 # global vars where we'll store x/y pos & radius
cv2.namedWindow('pupil')
cv2.setMouseCallback('pupil', draw_circle)
while True:
    cv2.imshow('pupil',frame_pupil)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        frame_pupil = frame.copy()
    elif k == ord('\r'):
        break
cv2.destroyAllWindows()


# Adjust preprocessing parameters
## Initial values -- have to use ints, we'll convert back later
frame_params = imutils.preprocess_image(frame_orig, roi)
edges_params = feature.canny(frame_params, sigma=5)

sig_cutoff = 50
sig_gain = 5
n_colors = 12
k_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
gauss_sig = 100
canny_sig = 5

cv2.namedWindow('params', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Sigmoid Cutoff', 'params', sig_cutoff, 100, imutils.nothing)
cv2.createTrackbar('Sigmoid Gain', 'params', sig_gain, 20, imutils.nothing)
cv2.createTrackbar('n Colors', 'params', n_colors, 16, imutils.nothing)
cv2.createTrackbar('Gaussian Sigma', 'params', gauss_sig, 200, imutils.nothing)
cv2.createTrackbar('Canny Sigma', 'params', canny_sig, 10, imutils.nothing)

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

    frame_params = imutils.preprocess_image(frame_orig, roi, sig_cutoff=sig_cutoff/100.,
                                    sig_gain=sig_gain, n_colors=n_colors,
                                    gauss_sig=gauss_sig/100.,
                                    k_criteria=k_criteria)
    edges_params = feature.canny(frame_params, sigma=canny_sig)
    edges_idx = np.where(edges_params)
    edges_xy = np.column_stack(edges_idx)
    # flip lr because coords are flipped for images
    edges_xy = np.fliplr(edges_xy)

    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_orig = imutils.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)


    cv2.imshow('params', np.vstack([frame_orig, frame_params, edges_params]))
cv2.destroyAllWindows()

# run once we get good params

# remake video object to restart from frame 0
vid = cv2.VideoCapture(vid_file)
pmod = model.Pupil_Model(ix, iy, rad)

cv2.namedWindow('run', flags=cv2.WINDOW_NORMAL)
fig, ax = plt.subplots()
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()

    frame_params = imutils.preprocess_image(frame_orig, roi, sig_cutoff=sig_cutoff / 100.,
                                            sig_gain=sig_gain, n_colors=n_colors,
                                            gauss_sig=gauss_sig / 100.,
                                            k_criteria=k_criteria)

    # canny edge detection & reshaping coords
    edges_params = feature.canny(frame_params, sigma=canny_sig)
    edges_xy = np.where(edges_params)
    edges_xy = np.column_stack(edges_xy)

    # flip lr because coords are flipped for images
    edges_xy = np.fliplr(edges_xy)

    # update our model and get some points to plot
    pmod.update(edges_xy, frame_params)
    #print(pmod.stdev, pmod.n_points[-1], pmod.qp)
    (model_x, model_y) = pmod.make_points(100, splitxy=True)
    model_x, model_y = model_x.astype(np.int), model_y.astype(np.int)

    # draw points on the images
    frame_orig = imutils.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)

    # make other images color
    frame_params_c = np.repeat(frame_params[:,:,np.newaxis], 3, axis=2)
    edges_params_c = np.repeat(edges_params[:,:,np.newaxis], 3, axis=2)

    # draw circle, have to flip x/y coords again...
    draw.set_color(frame_orig, (model_y, model_x), (1,0,0))
    draw.set_color(frame_params_c, (model_y, model_x), (1,0,0))
    draw.set_color(edges_params_c, (model_y, model_x), (1, 0, 0))

    ax.clear()
    ax.plot(range(len(pmod.pupil_diam)), pmod.pupil_diam)
    ax.plot(range(len(pmod.stdevs)), pmod.stdevs)
    plt.pause(0.01)

    cv2.imshow('run', np.vstack([frame_orig, frame_params_c, edges_params_c]))











