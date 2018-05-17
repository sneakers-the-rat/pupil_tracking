import numpy as np
np.seterr(divide='ignore')
np.seterr(invalid='ignore')
import cv2
from skimage import draw
from itertools import count
import argparse
import Tkinter as tk, tkFileDialog
import os
from datetime import datetime
import json
from tqdm import trange, tqdm
import multiprocessing as mp
import pandas as pd

import imops
import runops
import fitutils
import workers



##################################

def run_mp(file, params, data_dir, n_proc=7):
    vid = cv2.VideoCapture(file)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # start processes
    procs = []
    for i in range(n_proc):
        procs.append(mp.Process(target=workers.frame_worker2, args=(task_queue, done_queue, params, i+1)))
        procs[-1].start()

    # and the grabber
    #grabber = mp.Process(target=workers.result_grabber, args=(done_queue,))
    #grabber.start()

    for i in trange(total_frames, position=0):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = imops.crop(frame, params['roi'])

        n_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

        #task_queue.put((frame, n_frame), block=True, timeout=60)
        task_queue.put((frame, n_frame), block=False)
        # try:
        #     results.append(done_queue.get())
        # except:
        #     pass
        # if i%100==0:
        #     print(len(results))

    results = []
    for i in range(n_proc):
        results.append(done_queue.get())





def run(files, params, data_dir):
    thetas = np.linspace(0, np.pi * 2, num=200, endpoint=False)
    # loop through videos...
    pool = mp.Pool(8)

    for fn in tqdm(files, total=len(files), position=0):

        # open video, get params, make basic objects
        vid = cv2.VideoCapture(fn)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        #frame_counter = count()

        # appending to lists is actually pretty fast in python when dealing w/ uncertain quantities
        # store ellipse parameters here, rejoin into a pandas dataframe at the end
        # x_list = [] # x position of ellipse center
        # y_list = [] # y position of ellipse center
        # a_list = [] # major axis (enforced when lists are combined - fitutils.clean_lists)
        # b_list = [] # minor axis ("")
        # t_list = [] # theta, angle of a from x axis, radians, increasing counterclockwise
        # n_list = [] # frame number
        # v_list = [] # mean value of points contained within ellipse
        # c_list = [] # coverage - n_points/perimeter
        # g_list = [] # gradient magnitude of edge points

        results = []
        for i in trange(total_frames, position=1):
            ret, frame = vid.read()
            if ret == False:
                # ret aka "if return == true"
                # aka didn't return a frame
                # so
                # yno
                # we got ta take a
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = imops.crop(frame, params['roi'])

            n_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

            results.append(pool.apply_async(runops.process_frame_all, args=(frame, params, n_frame)))

            #
            # # Chew up a frame, return a list of ellipses
            # try:
            #     ellipses, frame_preproc, edge_mag = runops.process_frame(frame, params)
            # except TypeError:
            #     # if doesn't fit any ellipses, will return a single None, which will throw a
            #     # typeerror because it tries to unpack None into the three values above...
            #     continue
            #
            # for e, n_pts in ellipses:
            #     # ellipses actually gets returned as a tuple (ellipse object, n_pts)
            #     if not e:
            #         continue
            #
            #     x_list.append(e.params[0])
            #     y_list.append(e.params[1])
            #     a_list.append(e.params[2])
            #     b_list.append(e.params[3])
            #     t_list.append(e.params[4])
            #     n_list.append(n_frame)
            #
            #     # get mean darkness within each ellipse
            #     # TODO: Validate - make sure we're getting the right shit here.
            #     ell_mask_y, ell_mask_x = draw.ellipse(e.params[0], e.params[1], e.params[2], e.params[3],
            #                                           shape=(frame_preproc.shape[1], frame_preproc.shape[0]),
            #                                           rotation=e.params[4])
            #     v_list.append(np.mean(frame_preproc[ell_mask_x, ell_mask_y]))
            #
            #
            #
            #     # coverage - number of points vs. circumference
            #     # perim: https://stackoverflow.com/a/42311034
            #     perimeter = np.pi * (3 * (e.params[2] + e.params[3]) -
            #                          np.sqrt((3 * e.params[2] + e.params[3]) *
            #                         (e.params[2] + 3 * e.params[3])))
            #
            #     c_list.append(float(n_pts)/perimeter)
            #
            #     # get the mean edge mag for predicted points on the ellipse,
            #     # off-target ellipses often go through the pupil aka through areas with low gradients...
            #     e_points = np.round(e.predict_xy(thetas)).astype(np.int)
            #     e_points[:,0] = np.clip(e_points[:,0], 0, frame_preproc.shape[0]-1)
            #     e_points[:, 1] = np.clip(e_points[:,1], 0, frame_preproc.shape[1]-1)
            #     g_list.append(np.mean(edge_mag[e_points[:,0], e_points[:,1]]))

        got_results = []
        for r in tqdm(results, total=total_frames, position=2):
            got_results.append(r.get())

        flat_results = got_results[0]
        for i in got_results[1:]:
            for k, v in i.items():
                flat_results[k].extend(v)


        df = pd.DataFrame.from_dict(flat_results)
        vid_name = os.path.basename(fn).rsplit('.', 1)[0]
        save_fn = os.path.join(data_dir, "Ellall_" + vid_name + ".csv")
        df.to_csv(save_fn)
        # clean and combine parameter lists
        # ell_df = fitutils.clean_lists(x_list, y_list, a_list, b_list, t_list, v_list, n_list, c_list, g_list)
        # # remove extremely bad ellipses
        # ell_df = fitutils.basic_filter(ell_df, params['mask']['x'], params['mask']['y'], params['mask']['r'])
        # # remove more bad ellipses with knn outlier detection
        # ell_df_out = fitutils.filter_outliers(ell_df, neighbors=50, outlier_thresh=0.3)
        # # TODO: Use these to seed the segmentation algorithm:
        # max_idx = ell_df.groupby("n")['v'].transform(max) == ell_df['v']
        # ell_df_max = ell_df[max_idx]
        # Finally, smooth estimates
        #ell_df_smooth = fitutils.smooth_estimates(ell_df_out, hl=2)

        # save ellipses to file

        #ell_df.to_csv(save_fn)



if __name__ == "__main__":
    #######################################
    #######################################
    # Initialization
    # Argument parser

    pars = argparse.ArgumentParser()
    pars.add_argument("--dir", help="Base directory for output & params")
    pars.add_argument("--vdir", help="Base directory for videos")
    pars.add_argument("--n_procs", help="Number of processes to spawn")
    pars.add_argument("--params", help="Prespecify a .json parameter set")
    pars.add_argument("--gray", help="Videos are grayscale (y/n)")

    # then parse em
    args = pars.parse_args()

    # base directory
    if args.dir:
        base_dir = args.dir
    else:
        base_dir = os.path.join(os.path.expanduser("~"), 'pupellipse')

    # video directory
    if args.vdir:
        vdir = args.vdir
    else:
        vdir = base_dir

    # number of processes
    if args.n_procs:
        try:
            n_procs = int(args.n_procs)

        except TypeError:
            SyntaxWarning("n_procs must be an integer, resorting to 1")
            n_procs = 1
    else:
        n_procs = 1

    # params file
    if args.params:
        try:
            with open(args.params, 'r') as param_f:
                params = json.load(param_f)
            passed_params = True
        except:
            passed_params = False
            UserWarning("Couldn't load param file {}, will prompt".format(args.params))
    else:
        passed_params = False

    # grayscale
    # TODO: make sure this is enforced everywhere/properly by decorating the preprocessing fxn
    if args.gray:
        if args.gray.lower() == "n":
            gray = False
        elif args.gray.lower() == "n":
            gray = True
        else:
            Warning("Couldn't parse gray = {}, (should be \"y\" or \"n\", defaulting to y".format(args.gray.lower()))
            gray = True
    else:
        gray = True

    #######################################
    #######################################
    # Ensure filestructure

    param_dir = os.path.join(base_dir, "params")
    data_dir = os.path.join(base_dir, "data")
    make_dirs = [param_dir, data_dir]

    for d in make_dirs:
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise

    #######################################
    # Get Files from user

    # hide tk root window
    # https://stackoverflow.com/a/14119223
    root = tk.Tk()
    root.withdraw()

    files = tkFileDialog.askopenfilenames(parent=root, title="Select Input Videos", initialdir=vdir)

    #######################################
    #######################################
    # Params
    # ask whether we want to define new parameters...

    if passed_params == False:
        load_params = raw_input("Load existing params? (y/n)")
        if load_params == "y":
            param_fn = tkFileDialog.askopenfilename(parent=root, title="Select Param .json file",
                                                    initialdir=param_dir)
            try:
                with open(param_fn, 'r') as params_f:
                    params = json.load(params_f)
                # p name is the filename without the extension
                p_name = os.path.basename(param_fn).rsplit('.', 1)[0]
                passed_params = True

            except:
                Warning("Couldn't load params file {}, prompting".format(param_fn))

                p_name = raw_input("What to call this batch of videos?")
                p_name = p_name.strip().strip('/').lower().replace(" ", "_")
                p_name = "_".join([p_name, datetime.now().strftime("%y-%m-%d-%H%M")])
                param_fn = os.sep.join([param_dir, p_name + ".json"])

        else:
            # If they say no or anything yno else happens
            p_name = raw_input("What to call this batch of videos?")
            p_name = p_name.strip().strip('/').lower().replace(" ", "_")
            p_name = "_".join([p_name, datetime.now().strftime("%y-%m-%d-%H%M")])
            param_fn = os.sep.join([param_dir, p_name + ".json"])

    #######################################
    # Get initial ROI and image preprocessing params if not passed

    if passed_params == False:
        # ya you gotta get em somewhere
        # returns params as a dict, refer to the function to see what it contains
        params = runops.set_params(files)
    else:
        # Append current files to list
        params['files'].extend(files)

    with open(param_fn, 'w') as param_file:
        json.dump(params, param_file)

    #######################################
    #######################################
    # do the rest
    # https://stackoverflow.com/a/11241708
    run(files, params, data_dir)

















