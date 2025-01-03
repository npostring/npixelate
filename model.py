import math

import cv2
import numpy as np


def sampling_clustering(img, pixel_size=8):
    '''
    Purely image reduction and enlargement method.
    '''
    # sampling(shrink)
    org_shape = img.shape
    ratio = 1 / pixel_size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    # sampling(expand)
    img = cv2.resize(img, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


def k_means_clustering(img, k=16, pixel_size=8):
    # sampling(shrink)
    org_shape = img.shape
    ratio = 1 / pixel_size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    # k-means
    out_img = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(out_img, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    out_img = centers[labels.flatten()].reshape(img.shape)

    # sampling(expand)
    img = cv2.resize(out_img, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img

    
def mda_clustering(img, color_num=16, color_thresh=10, pixel_size=8):
    '''
    Maximum Distance Algorithm
    https://www.jstage.jst.go.jp/article/itej/66/11/66_J399/_pdf
    '''

    # sampling(shrink)
    org_shape = img.shape
    ratio = 1 / pixel_size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
 
    # select representative color
    target = np.array((127, 127, 127), dtype=np.int32)
    max_dist = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pixel = img[y, x].astype(np.int32)
            dist = math.sqrt(((pixel[0]-target[0])*(100/255))**2 + ((pixel[1]-target[1])*(120/255))**2 + ((pixel[2]-target[2])*(120/255))**2)
            if dist > max_dist:
                max_dist = dist
                max_y = y
                max_x = x
    target = img[max_y, max_x].astype(np.int32)

    # setting initial cluster
    clusters = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    min_dists = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for y in range(min_dists.shape[0]):
        for x in range(min_dists.shape[1]):
            min_dists[y, x] = 1000000

    # update cluster
    cur_k = 0
    while True:
        # caluculate distance
        max_dist = 0
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                pixel = img[y, x].astype(np.int32)
                dist = math.sqrt(((pixel[0]-target[0])*(100/255))**2 + ((pixel[1]-target[1])*(120/255))**2 + ((pixel[2]-target[2])*(120/255))**2)
                if dist < min_dists[y, x]:
                    min_dists[y, x] = dist
                    clusters[y, x] = cur_k
                    if dist > max_dist:
                        max_dist = dist
                        max_y = y
                        max_x = x

        # if the distance is less than the threshold, the process is terminated
        if max_dist > color_thresh and cur_k < color_num:
            target = img[max_y, max_x].astype(np.int32)
            cur_k += 1
        else:
            break

    # calculate the centroid of each cluster and replace the color of the pixel with the centroid color
    color_palette = []
    for k_num in range(cur_k):
        y, x = np.where(clusters == k_num)
        color = np.zeros((3), dtype=np.int64)
        for ii in range(len(y)):
            color[0] += img[y[ii], x[ii]][0]
            color[1] += img[y[ii], x[ii]][1]
            color[2] += img[y[ii], x[ii]][2]
        color = (color / len(y)).astype(np.uint8)
        img[y, x] = color
        color_palette.append(color)

    # sampling(expand)
    img = cv2.resize(img, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img

def ndcd_clustering(img, h):
    '''
    Generating Pixel Art by Noniterative Dominant Color Decision Algorithm
    https://www.jstage.jst.go.jp/article/itej/74/3/74_597/_pdf
    '''
    org_shape = img.shape

    M = img.shape[0] // h
    N = img.shape[1] // h
    out_img = np.zeros((M, N, 3), dtype=np.uint8)

    # Divide the image into M x N blocks
    for i in range(M):
        for j in range(N):
            dist1 = cv2.norm(img[i*h][j*h], img[(i+1)*h-1][(j+1)*h-1], cv2.NORM_L2)
            dist2 = cv2.norm(img[i*h][(j+1)*h-1], img[(i+1)*h-1][j*h], cv2.NORM_L2)
            if dist1 > dist2:
                u = img[i*h][j*h]
                v = img[(i+1)*h-1][(j+1)*h-1]
            else:
                u = img[i*h][(j+1)*h-1]
                v = img[(i+1)*h-1][j*h]

            # clustering u and v 
            u_pixels = []
            v_pixels = []
            for iu in range(h):
                for jv in range(h):
                    dist_iu = cv2.norm(u, img[i*h+iu][j*h+jv], cv2.NORM_L2)
                    dist_jv = cv2.norm(v, img[i*h+iu][j*h+jv], cv2.NORM_L2)
                    if dist_iu > dist_jv:
                        u_pixels.append(img[i*h+iu][j*h+jv])
                    else:
                        v_pixels.append(img[i*h+iu][j*h+jv])

                    # calculate the average color of the many pixels in the block(u or v) 
                    if len(u_pixels) > len(v_pixels):
                        out_img[i, j] = np.mean(u_pixels, axis=0)
                    else:
                        out_img[i, j] = np.mean(v_pixels, axis=0)

    # sampling(expand)
    img = cv2.resize(out_img, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)

    return img


def convert(img, pixel_size=8, color_num=4, color_thresh=10, mode='kmeans', is_noise_reduction=True):
    # save alpha channel
    before_img = img.copy()
    if before_img.shape[2] == 4:
        img = img[:, :, :3]

    # resize to 1000px if the image is larger than 1000px
    ratio = None
    max_size = 1000
    if img.shape[0] > max_size or img.shape[1] > max_size:
        ratio = max_size / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    # noise reduction
    if is_noise_reduction:
        img = cv2.bilateralFilter(img, 5, 50, 20)

    # clustering
    if mode == 'sampling':
        img = sampling_clustering(img, pixel_size)
    if mode == 'k-means':
        img = k_means_clustering(img, color_num, pixel_size)
    elif mode == 'mda':
        img = mda_clustering(img, color_num=color_num, color_thresh=color_thresh, pixel_size=pixel_size)
    elif mode == 'ndcd':
        img = ndcd_clustering(img, pixel_size)

    # restore size
    if ratio is not None:
        img = cv2.resize(img, (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_NEAREST)
   
    # restore alpha channel
    if before_img.shape[2] == 4:
        before_img[:,:,:3] = img
        img = before_img

    return img
