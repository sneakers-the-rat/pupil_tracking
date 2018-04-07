import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import feature, morphology, img_as_float, draw, measure
from itertools import count
from time import time
import argparse
import Tkinter as tk, tkFileDialog
import os
from datetime import datetime
import json
import multiprocessing as mp
from tqdm import tqdm


import imops
import runops
import fitutils
import model



##############################
# Initialization
# Argument parser

pars = argparse.ArgumentParser()
pars.add_argument("--dir", help="Base directory for videos")
pars.add_argument("--n_procs", help="Number of processes to spawn")

###############
args = pars.parse_args()

if args.dir:
    base_dir = args.dir
else:
    base_dir = os.sep

if args.n_procs:
    try:
        n_procs = int(args.n_procs)

    except TypeError:
        SyntaxWarning("n_procs must be an integer, resorting to 1")
        n_procs = 1
else:
    n_procs = 1

###############
# Get Files

# hide tk root window
# https://stackoverflow.com/a/14119223

#root = tk.Tk()
#root.withdraw()

#files = tkFileDialog.askopenfilenames(parent=root, title="Select Input Videos", initialdir=base_dir)

# create file to save params
param_dir = os.path.join(os.path.expanduser("~"), "pupil_stuff")

p_name = raw_input("What to call this batch of videos?")
p_name = p_name.strip().strip('/').lower().replace(" ", "_")
p_name = "_".join([p_name, datetime.now().strftime("%y-%m-%d-%H%M")])
param_fn = os.sep.join([param_dir, p_name + ".json"])

if not os.path.exists(param_dir):
    try:
        os.makedirs(param_dir, mode=0774)

    except OSError:
        RuntimeWarning("Couldn't make param save directory, proceeding w/o saving")
        param_fn = None



###############################
# Get initial ROI and image preprocessing params

files=['/home/lab/pupil_vids/nick4.avi']
vid_fn = files[0]
vid = cv2.VideoCapture(vid_fn)
#vid.set(cv2.CAP_PROP_POS_MSEC, 350000)

# crop roi
roi, frame = runops.get_crop_roi(vid)
ret, frame = vid.read()
frame = frame_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = imops.crop(frame, roi)

# draw pupil (sry its circular)
ix, iy, rad = runops.draw_pupil(frame)

# adjust preprocessing params
params = runops.set_params(vid, roi)

# collect & save run parameters
params['mask'] = {'x':ix, 'y':iy, 'r':rad}
params['files'] = files
params['n_procs'] = n_procs
params['roi'] = roi

with open(param_fn, 'w') as param_file:
    json.dump(params, param_file)

##################################
# run

batch_size = 200 # frames

# run once we get good params
# remake video object to restart from frame 0
vid = cv2.VideoCapture(vid_fn)
#vid.set(cv2.CAP_PROP_POS_MSEC, 350000)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
n_frames = 2000
#thetas = np.linspace(0, np.pi*2, num=100, endpoint=False)
frame_counter = count()
batch_counter = count()
thetas = np.linspace(0, np.pi*2, num=200, endpoint=False)

#pool = mp.Pool(processes=n_procs)
pbar = tqdm(total=n_frames, position=0)
#batch_frames = np.ndarray((frame.shape[0], frame.shape[1], batch_size), dtype=np.uint8)
params_frames = []
results = []

emod = measure.EllipseModel()

x_list = []
y_list = []
a_list = []
b_list = []
t_list = []
n_list = []
v_list = []

cv2.namedWindow('run', flags=cv2.WINDOW_NORMAL)

for i in xrange(n_frames):
    k = cv2.waitKey(1) & 0xFF
    if k == ord('\r'):
        break

    ret, frame_orig = vid.read()
    if ret == False:
        break


    n_frame = frame_counter.next()
    pbar.update()
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_orig = imops.crop(frame_orig, roi)

    frame = imops.preprocess_image(frame_orig, params['roi'],
                                   sig_cutoff=params['sig_cutoff'],
                                   sig_gain=params['sig_gain'])

    edges = imops.scharr_canny(frame, sigma=params['canny_sig'],
                               high_threshold=params['canny_high'], low_threshold=params['canny_low'])

    edges_rep = imops.repair_edges(edges, frame)
    if not edges_rep:
        continue

    edges_3 = np.repeat(edges[:,:,np.newaxis], 3, axis=2).astype(np.uint8)
    edges_3 = edges_3 * 255
    edges_rep_im = np.zeros(edges_3.shape, dtype=np.uint8)
    frame_orig = np.repeat(frame_orig[:,:,np.newaxis], 3, axis=2).astype(np.uint8)

    for e in edges_rep:
        edges_rep_im[e[:,0],e[:,1],:] = 255

    ellipses = [imops.fit_ellipse(e) for e in edges_rep]
    #ell_pts = np.ndarray(shape=(0, 2))
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
        ell_mask_y, ell_mask_x = draw.ellipse(e.params[0], e.params[1], e.params[2], e.params[3],
                                              shape=(frame.shape[1], frame.shape[0]),
                                              rotation=e.params[4])

        v_list.append(np.mean(frame[ell_mask_x, ell_mask_y]))

        points = emod.predict_xy(thetas, params=e.params)
        points = points.astype(np.int)

        draw.set_color(edges_3, (points[:,0], points[:,1]), (0,0,255))
        draw.set_color(edges_rep_im, (points[:,0], points[:,1]), (0,0,255))
        draw.set_color(frame_orig, (points[:, 0], points[:, 1]), (0, 0, 255))

    cv2.imshow('run', np.vstack((frame_orig, edges_3, edges_rep_im)))

ell_df = fitutils.clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list)
ell_df = fitutils.basic_filter(ell_df, ix, iy, rad)
ell_df_out = fitutils.filter_outliers(ell_df, neighbors=1000)
ell_df_smooth = fitutils.smooth_estimates(ell_df_out, hl=2)



        #batch_frames[:, :, n_frame%batch_size] = frame_orig

    # if n_frame % batch_size == 0:
    #     batch_number = batch_counter.next()
    #     results.append(pool.apply_async(runops.process_frames, args=(batch_frames.copy(), params, batch_number)))
    #     batch_frames = np.ndarray((frame.shape[0], frame.shape[1], batch_size), dtype=np.uint8)
    #     print(len(params_frames))





#params_frames = [job.get() for job in results]
#pool.close()
#pool.join()

print(params_frames)












