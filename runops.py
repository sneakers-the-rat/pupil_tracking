import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import feature, morphology, img_as_float, draw
from itertools import count
from time import time
import multiprocessing as mp
from tqdm import tqdm, trange

import imops
import fitutils

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



def get_crop_roi(vid):
    ret, frame = vid.read()
    if ret == False:
        Exception("No Frame from first video!")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.selectROI(frame)
    frame = imops.crop(frame, roi)
    cv2.destroyAllWindows()
    return roi, frame


def draw_pupil(frame):
    global drawing, ix, iy, rad, frame_pupil
    frame_pupil = frame.copy()
    drawing = False  # true if mouse is pressed
    ix, iy, rad = -1, -1, 5  # global vars where we'll store x/y pos & radius
    cv2.namedWindow('pupil')
    cv2.setMouseCallback('pupil', draw_circle)
    while True:
        cv2.imshow('pupil', frame_pupil)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            frame_pupil = frame.copy()
        elif k == ord('\r'):
            break
    cv2.destroyAllWindows()

    return ix, iy, rad

def set_params(vid, roi):
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imops.crop(frame, roi)
    frame_params = imops.preprocess_image(frame, roi)
    edges_params = imops.scharr_canny(frame_params, sigma=3)

    # initial values, empirically set.
    # have to use ints with cv2's windows, we'll convert later
    sig_cutoff = 50
    sig_gain = 5
    canny_sig = 200
    canny_high = 50
    canny_low = 10
    closing_rad = 3

    cv2.namedWindow('params', flags=cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Sigmoid Cutoff', 'params', sig_cutoff, 100, imops.nothing)
    cv2.createTrackbar('Sigmoid Gain', 'params', sig_gain, 20, imops.nothing)
    cv2.createTrackbar('Gaussian Blur', 'params', canny_sig, 700, imops.nothing)
    cv2.createTrackbar('Canny High Threshold', 'params', canny_high, 300, imops.nothing)
    cv2.createTrackbar('Canny Low Threshold', 'params', canny_low, 300, imops.nothing)
    cv2.createTrackbar('Closing Radius', 'params', closing_rad, 10, imops.nothing)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break

        ret, frame_orig = vid.read()
        frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame = imops.crop(frame, roi)

        sig_cutoff = cv2.getTrackbarPos('Sigmoid Cutoff', 'params')
        sig_gain = cv2.getTrackbarPos('Sigmoid Gain', 'params')
        canny_sig = cv2.getTrackbarPos('Gaussian Blur', 'params')
        canny_high = cv2.getTrackbarPos('Canny High Threshold', 'params')
        canny_low = cv2.getTrackbarPos('Canny Low Threshold', 'params')
        closing_rad = cv2.getTrackbarPos('Closing Radius', 'params')

        sig_cutoff = sig_cutoff / 100.
        canny_sig = canny_sig / 100.
        canny_high = canny_high / 100.
        canny_low = canny_low / 100.

        frame = imops.preprocess_image(frame, roi,
                                       sig_cutoff=sig_cutoff,
                                       sig_gain=sig_gain,
                                       closing=closing_rad)
        edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                          high_threshold=canny_high, low_threshold=canny_low)

        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame_orig = imops.crop(frame_orig, roi)
        frame_orig = img_as_float(frame_orig)

        cv2.imshow('params', np.vstack([frame_orig, frame, edges_params]))
    cv2.destroyAllWindows()

    params = {
        "sig_cutoff": sig_cutoff,
        "sig_gain": sig_gain,
        "canny_sig": canny_sig,
        "canny_high": canny_high,
        "canny_low": canny_low
    }

    return params

def serve_frames(vid, output):
    pass

def process_frames(frames, params, order):
    # do the thing
    frames = frames.copy()
    ell_params = None

    proc = mp.current_process()
    proc_num = int(proc.name[-1])

    frame_counter = count()

    # get params
    roi = params['roi']


    # frame_params = np.ndarray(shape=(0, 7))
    x_list = []
    y_list = []
    a_list = []
    b_list = []
    t_list = []
    n_list = []
    v_list = []
    print(proc_num)
    pbar = tqdm(total=frames.shape[2], position=10)

    for i in xrange(frames.shape[2]):
        frame = frames[:,:,i].squeeze()
        n_frame = frame_counter.next()
        pbar.update()

        frame = imops.preprocess_image(frame, params['roi'],
                                       sig_cutoff=params['sig_cutoff'],
                                       sig_gain=params['sig_gain'])

        # canny edge detection & reshaping coords
        edges = imops.scharr_canny(frame, sigma=params['canny_sig'],
                                          high_threshold=params['canny_high'], low_threshold=params['canny_low'])

        edges = imops.repair_edges(edges, frame)


        labeled_edges = morphology.label(edges)

        uq_edges = np.unique(labeled_edges)
        uq_edges = uq_edges[uq_edges>0]
        ellipses = [imops.fit_ellipse(labeled_edges, e) for e in uq_edges]
        ell_pts = np.ndarray(shape=(0,2))
        for e in ellipses:
            if not e:
                continue

            x_list.append(e.params[0])
            y_list.append(e.params[1])
            a_list.append(e.params[2])
            b_list.append(e.params[3])
            t_list.append(e.params[4])
            n_list.append(n_frame)
            # get mean darkness
            ell_mask_y, ell_mask_x = draw.ellipse(ell_params[0], ell_params[1], ell_params[2], ell_params[3],
                                    shape=(labeled_edges.shape[1], labeled_edges.shape[0]), rotation=ell_params[4])

            v_list.append(np.mean(frame[ell_mask_x, ell_mask_y]))

    ell_params = fitutils.clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list)

    return order, ell_params.tolist()

