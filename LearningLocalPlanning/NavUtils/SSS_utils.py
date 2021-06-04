

import numpy as np
from numba import njit, jit 
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt as edt


def plot_safety_scan(scan, starts, ends, dw_ds, valid_window, pp_action, new_action):
    plt.figure(1)
    plt.clf()
    plt.title(f"Lidar Scan with window: n")

    plt.ylim([-2, 8])
    # plt.xlim([-1.5, 1.5])
    plt.xlim([-4, 4])

    xs, ys = segment_lidar_scan(scan)
    plt.plot(xs, ys, '-+')

    sines, cosines = get_trigs(len(scan))
    for s, e in zip(starts, ends):
        xss = [0, scan[s]*sines[s], scan[e]*sines[e], 0]
        yss = [0, scan[s]*cosines[s], scan[e]*cosines[e], 0]
        plt.plot(xss, yss, '-+')

    scale = 2
    for j, d in enumerate(dw_ds):
        if valid_window[j]:
            plt.plot(d*scale, -1, 'x', color='green', markersize=14)
        else:
            plt.plot(d*scale, -1, 'x', color='red', markersize=14)

    plt.plot(pp_action[0]*scale, -0.5, '+', color='red', markersize=22)
    plt.plot(new_action[0]*scale, -0.5, '*', color='green', markersize=16)

    plt.pause(0.0001)

# @njit(cache=True)
def segment_lidar_scan(scan):
    """ 
    Takes a lidar scan and reduces it to a set of points that make straight lines 
    TODO: possibly change implmentation to work completely in r, ths 
    """
    xs, ys = convert_scan_xy(scan)
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
    i_pts = [0]
    d_thresh = 0.3
    for i in range(len(diffs)):
        if diffs[i] > d_thresh:
            i_pts.append(i)
            i_pts.append(i+1)
    i_pts.append(len(scan)-1)

    if len(i_pts) < 10:
        i_pts.append(np.argmax(scan))
        i_pts.append(200)
        i_pts.append(300)
        i_pts.append(400)
        i_pts.append(450)
        i_pts.append(500)
        i_pts.append(550)
        i_pts.append(600)
        i_pts.append(700)
        i_pts.append(800)
        

    i_pts = np.array(i_pts)
    i_pts = np.sort(i_pts)
    x_pts = xs[i_pts]
    y_pts = ys[i_pts]

    return x_pts, y_pts


@njit(cache=True)
def convert_scan_xy(scan):
    sines, cosines = get_trigs(len(scan))
    xs = scan * sines
    ys = scan * cosines    
    return xs, ys

@njit(cache=True)
def get_trigs(n_beams, fov=np.pi):
    angles = np.arange(n_beams) * fov / 999 -  np.ones(n_beams) * fov /2 
    return np.sin(angles), np.cos(angles)
