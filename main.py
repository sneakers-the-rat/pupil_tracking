import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import feature, morphology, img_as_float, draw
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

files=['/Users/jonny/pupil_vids/nick3.avi']
vid_fn = files[0]
vid = cv2.VideoCapture(vid_fn)

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
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
#thetas = np.linspace(0, np.pi*2, num=100, endpoint=False)
frame_counter = count()
batch_counter = count()

pool = mp.Pool(processes=n_procs)
pbar = tqdm(total=total_frames, position=0)
batch_frames = np.ndarray((frame.shape[0], frame.shape[1], batch_size), dtype=np.uint8)
params_frames = []
results = []

for i in xrange(501):
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
    batch_frames[:, :, n_frame%batch_size] = frame_orig

    if n_frame % batch_size == 0:
        batch_number = batch_counter.next()
        results.append(pool.apply_async(runops.process_frames, args=(batch_frames.copy(), params, batch_number)))
        batch_frames = np.ndarray((frame.shape[0], frame.shape[1], batch_size), dtype=np.uint8)
        print(len(params_frames))



params_frames = [job.get() for job in results]
pool.close()
pool.join()

print(params_frames)












