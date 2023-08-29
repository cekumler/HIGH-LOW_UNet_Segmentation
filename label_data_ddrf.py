#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   labelData_ddrf.py

   original code to covert lat/lon points to grids points and method to generate binary mask for labels

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

   update mar 2022: read the FIQAS json file and labels
"""

import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Setup logging
logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("LabelData")

# Set to true to show image of data and labels,
# useful for debugging, pain for processing
show_image = False
gray = plt.cm.gray


def pointsToDataPoints(points, lats, lons, offsetX=False, width=0):
    """ given points with lat lons, create x,y mapping to points in grid

        edit: we have the indicies, so we can sort of skip this mapping step    
    """
    points = np.array(points)
    logger.info("Processing % cyclones", points.shape[0])
    xypoints = []

    for p in points:

        # FIQAS data is already in the correct index, so this is where we pull that index value
        # if there is a shift, then we can make that correction here.
        xypoints.append([p[0], p[1], p[2], p[3]])
    xypoints = np.array(xypoints)

    return xypoints


def createLabeledData(data, xypoints, test_data,
                      center_mark=10, range_value=0):
    """ create mask for underlying data based on points
    
    creates a binary mask same size of data using xy points as the center

    Arguments:
        data {2d array} -- only need for shape of mask or test images
        xypoints {list(int, int)} -- list of grid x,y points for labels
        test_data {list} -- array to append list too
    
    Keyword Arguments:
        center_mark {number} -- number of points off center_mark to create label (default: {10})
        range_value {number} -- if you are testing, specify range_value to help color code image (default: {0})
    """
    logger.info("Creating labeled data")

    img = None
    if show_image:
       if range_value == 0:
            max_value = np.max(data)
            min_value = np.min(data)
            range_value = max_value - min_value
            logger.info("range: %s", range_value)

       img = Image.fromarray(gray((data.astype(np.float32)/range_value), bytes=True))
       img.show()

    # Mark Points on original data
    if show_image:
        logger.debug(xypoints)
        for p in xypoints:
            for i in range(-center_mark, center_mark):
                for j in range(-center_mark, center_mark):
                    if p[1]+j < img.width and p[1]+j >= 0 and p[0]+j < img.height and p[0]+j >= 0:
                    #   r, g, b, a = img.getpixel((p[1]+j, p[0]+i))
                       img.putpixel((p[1] + j, p[0] + i), (255, 0, 0))

    test = np.zeros(data.shape, dtype=np.uint8)
    # I'm adding the next line to get info on size
    print('data shape: ', data.shape)
    for p in xypoints:
        ymin = int(round(max(0, p[0] - center_mark)))
        xmin = int(round(max(0, p[1] - center_mark)))
        ymax = int(round(min(data.shape[0]-1,p[0] + center_mark)))
        xmax = int(round(min(data.shape[1]-1,p[1] + center_mark)))
 
       # first way that worked here is where we set our 1's or 2's to the labels based on the flag
        # lets assume it's a low unless it's a high ;P
        category = 0
#catlow        if p[2] ==  1:
#catlow           category = 1 
#cathigh        if p[2] ==  2:
#cathigh           category = 1
#        category = 0
        if p[2] >  1:
           category = 2
        elif p[2] > 0:
           category = 1 
        test[ymin:ymax, xmin:xmax] = category


    if np.isnan(data).any():
       mask = np.where(np.isnan(data))
       test[mask[0], mask[1]] = 0

    if show_image:
        img2 = Image.fromarray(gray(test / 1.0, bytes=True))
        img2.show()
        img.show()

    test_data.append(test)
