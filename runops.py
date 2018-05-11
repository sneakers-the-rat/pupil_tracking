import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import feature, morphology, img_as_float, draw, measure
from skvideo import io
from itertools import count, cycle
from time import time, sleep
import multiprocessing as mp
from tqdm import tqdm, trange
import pandas as pd
import json

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

def set_params(files):
    # cycle through files..
    filecyc = cycle(files)
    # start with the first video
    vid = cv2.VideoCapture(filecyc.next())

    # crop roi (x, y, width, height)
    roi, frame = get_crop_roi(vid)

    # draw pupil (sry its circular)
    ix, iy, rad = draw_pupil(frame)

    # initial values, empirically set.
    # have to use ints with cv2's windows, we'll convert later
    sig_cutoff = 50
    sig_gain = 5
    canny_sig = 200
    canny_high = 50
    canny_low = 10
    closing_rad = 3

    cv2.namedWindow('params', flags=cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Sigmoid Cutoff',       'params', sig_cutoff,  100, imops.nothing)
    cv2.createTrackbar('Sigmoid Gain',         'params', sig_gain,    20,  imops.nothing)
    cv2.createTrackbar('Gaussian Blur',        'params', canny_sig,   700, imops.nothing)
    cv2.createTrackbar('Canny High Threshold', 'params', canny_high,  300, imops.nothing)
    cv2.createTrackbar('Canny Low Threshold',  'params', canny_low,   300, imops.nothing)
    cv2.createTrackbar('Closing Radius',       'params', closing_rad, 10,  imops.nothing)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break

        ret, frame_orig = vid.read()
        if ret == False:
            # cycle to the next video (or restart) and skip this iter of param setting
            vid = cv2.VideoCapture(filecyc.next())
            continue
        frame = frame_orig.copy()

        sig_cutoff  = cv2.getTrackbarPos('Sigmoid Cutoff', 'params')
        sig_gain    = cv2.getTrackbarPos('Sigmoid Gain', 'params')
        canny_sig   = cv2.getTrackbarPos('Gaussian Blur', 'params')
        canny_high  = cv2.getTrackbarPos('Canny High Threshold', 'params')
        canny_low   = cv2.getTrackbarPos('Canny Low Threshold', 'params')
        closing_rad = cv2.getTrackbarPos('Closing Radius', 'params')

        # see i toldya it would be fine to have ints...
        sig_cutoff = sig_cutoff / 100.
        canny_sig  = canny_sig / 100.
        canny_high = canny_high / 100.
        canny_low  = canny_low / 100.

        frame = imops.preprocess_image(frame, roi,
                                       sig_cutoff=sig_cutoff,
                                       sig_gain=sig_gain,
                                       closing=closing_rad)
        edges_params = imops.scharr_canny(frame, sigma=canny_sig,
                                          high_threshold=canny_high, low_threshold=canny_low)

        # TODO: Also respect grayscale param here
        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame_orig = imops.crop(frame_orig, roi)
        frame_orig = img_as_float(frame_orig)

        cv2.imshow('params', np.vstack([frame_orig, frame, edges_params]))
    cv2.destroyAllWindows()

    # collect parameters
    params = {
        "sig_cutoff": sig_cutoff,
        "sig_gain": sig_gain,
        "canny_sig": canny_sig,
        "canny_high": canny_high,
        "canny_low": canny_low,
        "mask" : {'x': ix, 'y': iy, 'r': rad},
        "files": files,
        "roi": roi,
        "shape": (roi[3], roi[2]),
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

def process_frame(frame, params):
    frame = imops.preprocess_image(frame, params['roi'],
                                   sig_cutoff=params['sig_cutoff'],
                                   sig_gain=params['sig_gain'])

    # get gradients
    grad_x, grad_y, edge_mag = imops.edge_vectors(frame, sigma=params['canny_sig'], return_angles=False)

    edges = imops.scharr_canny(frame, sigma=params['canny_sig'],
                               high_threshold=params['canny_high'],
                               low_threshold=params['canny_low'],
                               grads={'grad_x': grad_x,
                                      'grad_y': grad_y,
                                      'edge_mag': edge_mag})

    edges_rep = imops.repair_edges(edges, frame, grads={'grad_x': grad_x,
                                                        'grad_y': grad_y,
                                                        'edge_mag': edge_mag})

    # return [(ellipse, n_pts)]
    try:
        return [(imops.fit_ellipse(e), len(e)) for e in edges_rep], frame, edge_mag
    except TypeError:
        return None


def play_fit(vid, roi, params, fps=30):
    thetas = np.linspace(0, np.pi * 2, num=200, endpoint=False)

    # start vid at first frame in params
    if "n" in params.keys():

        first_frame = params.n.min()
    else:
        first_frame = params.index.min()

    ret = vid.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    frame_counter = count()

    emod = measure.EllipseModel()

    cv2.namedWindow('play', flags=cv2.WINDOW_NORMAL)
    for i in xrange(len(params)):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break

        ret, frame_orig = vid.read()
        if ret == False:
            break
        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

        n_frame = frame_counter.next()

        if 'n' in params.keys():
            ell_rows = params[params.n==n_frame]
            frame_orig = imops.crop(frame_orig, roi)
            frame_orig = img_as_float(frame_orig)
            for i, e in ell_rows.iterrows():
                e_points = emod.predict_xy(thetas, params=(e.x, e.y, e.a, e.b, e.t))
                e_points = e_points.astype(np.int)

                draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] + 1, e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1] - 1), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] + 1, e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] + 1, e_points[:, 1] - 1), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1] - 1), (1, 0, 0))


        else:
            p = params.iloc[n_frame]
            e_points = emod.predict_xy(thetas, params=(p.x, p.y, p.a, p.b, p.t))
            e_points = e_points.astype(np.int)

            frame_orig = imops.crop(frame_orig, roi)
            frame_orig = img_as_float(frame_orig)

            draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1]), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0]+1, e_points[:, 1]), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1]), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1]+1), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0], e_points[:, 1]-1), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0] + 1, e_points[:, 1]+1), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0] + 1, e_points[:, 1]-1), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1]+1), (1, 0, 0))
            draw.set_color(frame_orig, (e_points[:, 0] - 1, e_points[:, 1]-1), (1, 0, 0))
        cv2.imshow('play', frame_orig)
        sleep(1./fps)

        # frame_orig = frame_orig*255
        # frame_orig = frame_orig.astype(np.uint8)

        # writer.writeFrame(frame_orig)

    cv2.destroyAllWindows()

def video_from_params(param_fn, ell_fn, which_vid = 0):
    thetas = np.linspace(0, np.pi * 2, num=300, endpoint=False)

    # load params from .json file, vid filenames will be in there
    with open(param_fn, 'r') as param_f:
        params = json.load(param_f)

    # for now just do one video
    vid_fn = str(params['files'][which_vid])

    vid = cv2.VideoCapture(vid_fn)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    ell_df = pd.read_csv(ell_fn)

    vid_path, vid_name = os.path.split(vid_fn)
    vid_name = "Ellone_" + vid_name.rsplit('.',1)[0] + ".mp4"
    vid_out_fn = vid_path+"/"+vid_name


    writer = io.FFmpegWriter(vid_out_fn, outputdict={'-vcodec': 'libx264'})

    emod = measure.EllipseModel()

    ell_frame = ell_df.groupby('n')

    for i in trange(total_frames):

        ret, frame = vid.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imops.crop(frame, params['roi'])
        frame = img_as_float(frame)

        try:
            ell_rows = ell_frame.get_group(i)

            for i, e in ell_rows.iterrows():
                e_points = emod.predict_xy(thetas, params=(e.x, e.y, e.a, e.b, e.t))
                e_points = e_points.astype(np.int)

                draw.set_color(frame, (e_points[:, 0], e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] + 1, e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] - 1, e_points[:, 1]), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0], e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0], e_points[:, 1] - 1), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] + 1, e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] + 1, e_points[:, 1] - 1), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] - 1, e_points[:, 1] + 1), (1, 0, 0))
                draw.set_color(frame, (e_points[:, 0] - 1, e_points[:, 1] - 1), (1, 0, 0))


        except KeyError:
            # no ellipses this frame, just write frame
            pass

        writer.writeFrame(frame*255)

    writer.close()




