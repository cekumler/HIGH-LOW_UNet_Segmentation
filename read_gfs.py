#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   read_gfs.py

   script to read data from grib files 

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

   Edited by Christina Kumler
   Edit mar 10: The script shouldn't have to change because FIQAS is set up in grib2
   Just make sure that the correct data field, MSLP, is specified in the proc script

"""

import pygrib

def getDataWithLatLons(filename, varname):

    grbs = pygrib.open(filename)
    try:
        tmpdata = grbs.select(name=varname)[0]
        lats, lons = tmpdata.latlons()
        data = tmpdata.values
        return data, lats, lons
    except:
        print ("can't find variable", varname)
        return None, None, None

def getData(filename, varname):

    grbs = pygrib.open(filename)
    try:
        tmpdata = grbs.select(name=varname)[0]
        data = tmpdata.values
        return data
    except:
        print ("can't find variable", varname)
        return None, None, None

    
