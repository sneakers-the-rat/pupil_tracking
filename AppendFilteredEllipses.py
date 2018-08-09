



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
import runops
import imops
import fitutils
# Make FilteredEllipse csv
param_fn = '/home/lab/pupellipse/params/m8288_2018-08-04_18-08-06-1453.json'
with open(param_fn, 'r') as param_f:
    params = json.load(param_f)
fn = params['files']
for i in range(len(fn)):
    vid_name = os.path.basename(fn[i]).rsplit('.', 1)[0]
    data_dir = '/home/lab/pupellipse/data'
    ell_fn =  os.path.join(data_dir, "Ellall_" + vid_name + ".csv")
    ell_df = pd.read_csv(ell_fn)
    # add eccentricity column
    #ell_df ['e'] = ell_df['b'] / ell_df['a']
    #ell_df = ell_df[ell_df.e>0.7]
    ell_df['gn'] = (ell_df.g-ell_df.g.min())/(ell_df.g.max()-ell_df.g.min())
    ell_df['cn'] = (ell_df.c-ell_df.c.min())/(ell_df.c.max()-ell_df.c.min())
    ell_df['en'] = (ell_df.e-ell_df.e.min())/(ell_df.e.max()-ell_df.e.min())
    #ell_df['an'] = 1/np.power((ell_df.a-ell_df.a.mean()),2)
    #ell_df['bn'] = 1/np.power((ell_df.b-ell_df.b.mean()),2)
    #ell_df['xn'] = 1/np.power((ell_df.x-ell_df.x.mean()),2)
    #ell_df['yn'] = 1/np.power((ell_df.y-ell_df.y.mean()),2)




    ell_frame = ell_df.groupby("n")
    idxes = []
    for name, group in ell_frame:
        # TODO: once V is fixed, also add variance, and then use V adjusted by area - a high V over a large area is better than a high V over a small area
        good_vals = group.gn * group.cn * group.en #* group.xn * group.yn * group.an *group.bn
        idxes.append(group.loc[good_vals.idxmax()][0])
    only_good = ell_df.loc[np.isin(ell_df['Unnamed: 0'], idxes)]
    save_fn = os.path.join(data_dir, "Ellone_" + vid_name + ".csv")
    only_good.to_csv(save_fn)
    # Make FilteredEllipse mp4
    ell_fn = save_fn
    runops.video_from_params(param_fn,ell_fn,i)
#Then copy over FilteredEllipse....csv & FilteredEllipse....mp4
