import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import filters, exposure, feature, morphology, measure, img_as_float, draw
from collections import deque
from pandas import ewma, ewmstd
from itertools import count
import matplotlib.pyplot as plt
from scipy import ndimage
import skvideo.io
from time import time

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
pmask        = imutils.circle_mask(frame_params, ix, iy, rad)
#logistic     = imutils.infer_sigmoid(frame_params, pmask)
#frame_params = imutils.preprocess_image(frame_orig, roi, logistic=
#logistic)
edges_params = feature.canny(frame_params, sigma=3)

sig_cutoff = 50
sig_gain = 5
n_colors = 12
k_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
gauss_sig = 200
canny_sig  = 200
canny_high = 50
canny_low  = 10

cv2.namedWindow('params', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Sigmoid Cutoff', 'params', sig_cutoff, 100, imutils.nothing)
cv2.createTrackbar('Sigmoid Gain', 'params', sig_gain, 20, imutils.nothing)
#cv2.createTrackbar('n Colors', 'params', n_colors, 16, imutils.nothing)
cv2.createTrackbar('Gaussian Blur', 'params', canny_sig, 700, imutils.nothing)
cv2.createTrackbar('Canny High Threshold', 'params', canny_high, 100, imutils.nothing)
cv2.createTrackbar('Canny Low Threshold', 'params', canny_low, 100, imutils.nothing)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()

    sig_cutoff = cv2.getTrackbarPos('Sigmoid Cutoff', 'params')
    sig_gain = cv2.getTrackbarPos('Sigmoid Gain', 'params')
    #n_colors = cv2.getTrackbarPos('n Colors', 'params')
    canny_sig = cv2.getTrackbarPos('Gaussian Blur', 'params')
    canny_high = cv2.getTrackbarPos('Canny High Threshold', 'params')
    canny_low = cv2.getTrackbarPos('Canny Low Threshold', 'params')

    sig_cutoff = sig_cutoff/100.
    canny_sig = canny_sig/100.
    canny_high = canny_high/100.
    canny_low  = canny_low/100.

    frame = imutils.preprocess_image(frame_orig, roi,
                                            sig_cutoff=sig_cutoff,
                                            sig_gain=sig_gain)
    edges_params = feature.canny(frame, sigma=canny_sig,
                                 high_threshold=canny_high, low_threshold=canny_low)

    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_orig = imutils.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)


    cv2.imshow('params', np.vstack([frame_orig, frame, edges_params]))
cv2.destroyAllWindows()

# run once we get good params
#fourcc = cv2.VideoWriter_fourcc(*'X264')
#writer = cv2.VideoWriter('/home/lab/pupil_vids/tracking_human.mkv', fourcc, 30., frame_params.shape, True)

#writer = skvideo.io.FFmpegWriter('/home/lab/pupil_vids/tracking_human2.mp4')


# remake video object to restart from frame 0
vid = cv2.VideoCapture(vid_file)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
#pmod = model.Pupil_Model(ix, iy, rad)

cv2.namedWindow('run', flags=cv2.WINDOW_NORMAL)
#fig, ax = plt.subplots(4,1)
thetas = np.linspace(0, np.pi*2, num=100, endpoint=False)
frame_counter = count()

frame_params = np.ndarray(shape=(0, 6))

starttime = time()
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()
    if ret == False:
        break

    n_frame = frame_counter.next()
    if n_frame % 1000 == 0:
        now = time()
        fps = n_frame/(now-starttime)
        print('frame {} of {}, {} fps'.format(n_frame, total_frames, fps))


    frame = imutils.preprocess_image(frame_orig, roi,
                                            sig_cutoff=sig_cutoff,
                                            sig_gain=sig_gain)

    # canny edge detection & reshaping coords
    edges_params = feature.canny(frame, sigma=canny_sig,
                                 high_threshold=canny_high, low_threshold=canny_low)

    labeled_edges = morphology.label(edges_params)
    uq_edges = np.unique(labeled_edges)
    uq_edges = uq_edges[uq_edges>0]
    ellipses = [imutils.fit_ellipse(labeled_edges, e) for e in uq_edges]
    ell_pts = np.ndarray(shape=(0,2))
    for e in ellipses:
        try:
            points = e.predict_xy(thetas)
        except:
            continue
        if any(points.flatten()<0) or any(points[:,0]>labeled_edges.shape[1])\
            or any(points[:,1]>labeled_edges.shape[0]):
            # outside the image, skip
            continue

        ell_pts = np.concatenate((ell_pts, points), axis=0)
        ell_params = e.params
        ell_params.append(n_frame)
        frame_params = np.vstack((frame_params, ell_params))



    ell_pts = ell_pts.astype(np.int)






    # update our model and get some points to plot
    #pmod.update(edges_params, frame_params)
    #print(pmod.stdev, pmod.n_points[-1], pmod.qp)
    #(model_x, model_y) = pmod.make_points(200, splitxy=True)
    #model_x, model_y = model_x.astype(np.int), model_y.astype(np.int)
    #model_x, model_y = np.concatenate((model_x, model_x+1, model_x-1)), np.concatenate((model_y, model_y+1, model_y-1))

    #(st_mod_x, st_mod_y) = pmod.make_points(200, splitxy=True, st_mod=True)
    #st_mod_x, st_mod_y = st_mod_x.astype(np.int), st_mod_y.astype(np.int)
    #st_mod_x, st_mod_y = np.concatenate((st_mod_x, st_mod_x+1, st_mod_x-1)), np.concatenate((st_mod_y, st_mod_y+1, st_mod_y-1))


    # draw points on the images
    frame_orig = imutils.crop(frame_orig, roi)
    frame_orig = img_as_float(frame_orig)

    # make other images color
    #frame_params_c = np.repeat(frame_params[:,:,np.newaxis], 3, axis=2)
    edges_params_c = np.repeat(edges_params[:,:,np.newaxis], 3, axis=2)

    # draw circle, have to flip x/y coords again...
    draw.set_color(frame_orig, (ell_pts[:,1], ell_pts[:,0]), (0,0,255))
    #draw.set_color(frame_params_c, (model_y, model_x), (1,0,0))
    draw.set_color(edges_params_c, (ell_pts[:,1], ell_pts[:,0]), (0, 0, 1))
    #try:
    #    draw.set_color(edges_params_c, (pmod.f_points[:,1], pmod.f_points[:,0]), (1,0,0))
    #except:
    #    pass
    #draw.set_color(frame_orig, (st_mod_y, st_mod_x), (0,255,0))

    #frame_orig = cv2.cvtColor(frame_orig,cv2.COLOR_BGR2RGB)



    # ax[0].clear()
    # ax[1].clear()
    # ax[2].clear()
    # ax[3].clear()
    # if len(pmod.pupil_diam) < 200:
    #     ax[0].plot(range(len(pmod.pupil_diam)), pmod.pupil_diam)
    # else:
    #     ax[0].plot(range(200), pmod.pupil_diam[-200:])
    # ax[1].plot(range(len(pmod.n_points)), pmod.n_points)
    # ax[2].plot(range(len(pmod.resids)), pmod.resids)
    # ax[3].plot(range(len(pmod.obl)), pmod.obl)
    # plt.pause(0.001)

    #writer.writeFrame(frame_orig)

    cv2.imshow('run', np.vstack([frame_orig, edges_params_c]))
    #cv2.imshow('run', frame_orig)

#writer.close()











