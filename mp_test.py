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

import imops
import runops
import fitutils
import workers

from time import time, sleep

def listener(q):
    pbar = tqdm(total=500, position=1)
    for item in iter(q.get, None):
        pbar.update()

def main():
    params = {"files": ["/home/lab/pupellipse/Transient/Eye_mouse-8288_2018-04-28T14_25_33.mkv"], "roi": [426, 525, 528, 370], "shape": [370, 528], "sig_gain": 13, "canny_high": 1.0, "canny_sig": 5.0, "canny_low": 0.4, "sig_cutoff": 0.82, "mask": {"y": 212, "x": 248, "r": 134}}
    file = params['files'][0]
    n_proc = 7

    vid = cv2.VideoCapture(file)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create queues
    # task_queue = mp.Queue()
    # done_queue = mp.Queue()
    # iter_queue = mp.Queue()

    pool = mp.Pool(7)


    # start processes
    # procs = []
    # pbar = tqdm(total=500, position=1)
    # for i in range(n_proc):
    #     procs.append(mp.Process(target=workers.frame_worker2, args=(task_queue, done_queue, params, iter_queue)))
    #     procs[-1].start()

    #mp.Process(target=listener, args=(iter_queue,)).start()

    # and the grabber
    # grabber = mp.Process(target=workers.result_grabber, args=(done_queue,))
    # grabber.start(

    results = []
    starttime = time()
    for i in trange(500, position=0):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = imops.crop(frame, params['roi'])
        # frame = imops.preprocess_image(frame, params['roi'],
        #                                    sig_cutoff=params['sig_cutoff'],
        #                                    sig_gain=params['sig_gain'])

        n_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

        results.append(pool.apply_async(runops.process_frame_all, args=(frame, params, n_frame)))

        # task_queue.put((frame, n_frame), block=True, timeout=60)
        #task_queue.put((frame, n_frame), block=False)
        # try:
        #     results.append(done_queue.get())
        # except:
        #     pass
        # if i%100==0:
        #     print(len(results))

    got_results = []
    for r in tqdm(results, total=500):
        got_results.append(r.get())
    # while not task_queue.empty():
    #     sleep(1)

    endtime = time()

    print("total time: {}, time per frame: {}".format(endtime-starttime, 500./(endtime-starttime)))

    # for i in range(n_proc):
    #     task_queue.put("END")

    # results = []
    # for i in range(n_proc):
    #     results.append(done_queue.get())

    with open('/home/lab/test3.json', 'w') as jf:
        json.dump(got_results, jf)

if __name__ == "__main__":
    main()