import runops
import multiprocessing as mp
import cv2
import json
from skimage import draw
import numpy as np
from tqdm import tqdm

from time import time, sleep

# first_time = time()
# first_size = task_queue.qsize()
# sleep(5)
# second_time = time()
# second_size = task_queue.qsize()
#
# pps = float((first_size-second_size))/(second_time-first_time)
# print(pps)
#############################
# from the python multiprocessing examples
def frame_worker(input, output, params):
    for frame, n in iter(input.get, 'END'):
        try:
            result = runops.process_frame_all(frame, params, n)
        except:
            print('an exception...')
        output.put(result)

def frame_worker2(input, output, params, pbar):
    x_list = []  # x position of ellipse center
    y_list = []  # y position of ellipse center
    a_list = []  # major axis (enforced when lists are combined - fitutils.clean_lists)
    b_list = []  # minor axis ("")
    t_list = []  # theta, angle of a from x axis, radians, increasing counterclockwise
    n_list = []  # frame number
    v_list = []  # mean value of points contained within ellipse
    c_list = []  # coverage - n_points/perimeter
    g_list = []  # gradient magnitude of edge points

    thetas = np.linspace(0, np.pi*2, 200)
    for frame, n in iter(input.get, 'END'):
        try:
            ellipses, frame_preproc, edge_mag = runops.process_frame(frame, params, crop=False, preproc=False)
        except TypeError:
            # if doesn't fit any ellipses, will return a single None, which will throw a
            # typeerror because it tries to unpack None into the three values above...
            pbar.put(1)
            continue

        for e, n_pts in ellipses:
            # ellipses actually gets returned as a tuple (ellipse object, n_pts)
            if not e:
                continue

            x_list.append(e.params[0])
            y_list.append(e.params[1])
            a_list.append(e.params[2])
            b_list.append(e.params[3])
            t_list.append(e.params[4])
            n_list.append(n)

            # get mean darkness within each ellipse
            # TODO: Validate - make sure we're getting the right shit here.
            ell_mask_y, ell_mask_x = draw.ellipse(e.params[0], e.params[1], e.params[2], e.params[3],
                                                  shape=(frame_preproc.shape[1], frame_preproc.shape[0]),
                                                  rotation=e.params[4])
            v_list.append(np.mean(frame_preproc[ell_mask_x, ell_mask_y]))

            # coverage - number of points vs. circumference
            # perim: https://stackoverflow.com/a/42311034
            perimeter = np.pi * (3 * (e.params[2] + e.params[3]) -
                                 np.sqrt((3 * e.params[2] + e.params[3]) *
                                         (e.params[2] + 3 * e.params[3])))

            c_list.append(float(n_pts) / perimeter)

            # get the mean edge mag for predicted points on the ellipse,
            # off-target ellipses often go through the pupil aka through areas with low gradients...
            e_points = np.round(e.predict_xy(thetas)).astype(np.int)
            e_points[:, 0] = np.clip(e_points[:, 0], 0, frame_preproc.shape[0] - 1)
            e_points[:, 1] = np.clip(e_points[:, 1], 0, frame_preproc.shape[1] - 1)
            g_list.append(np.mean(edge_mag[e_points[:, 0], e_points[:, 1]]))

        pbar.put(1)

    # combine and return lists
    ret_dict = {
        'x': x_list,
        'y': y_list,
        'a': a_list,
        'b': b_list,
        't': t_list,
        'n': n_list,
        'v': v_list,
        'c': c_list,
        'g': g_list
    }
    output.put(ret_dict)


def result_grabber(input):
    results = []
    for result in iter(input.get, 'END'):
        results.append(result)

    testfile = '/home/lab/testdata.json'
    with open(testfile, 'w') as tf:
        json.dump(results, tf)


def test(params, n_proc=7):
    vid = cv2.VideoCapture(params['files'][0])
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    for i in range(n_proc):
        mp.Process(target=frame_worker, args=(task_queue, done_queue, params)).start()

    for i in trange(100):
        ret, frame = vid.read()
        n_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

        task_queue.put((frame, n_frame))

    results = []
    for i in range(100):
        results.append(done_queue.get())

