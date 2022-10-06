#!/usr/bin/env python3

#   Copyright (C) IMDEA Networks Institute 2022
#   This program is free software: you can redistribute it and/or modify
#
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see http://www.gnu.org/licenses/.
#
#   Authors: Yago Lizarribar <yago.lizarribar [at] imdea [dot] org>
#


################################################################################
# Imports
import itertools

import numpy as np
import pandas as pd
from scipy.signal import resample
import scipy.optimize as optimize

from hacktdoa.geodesy import geodesy
from hacktdoa.optim import exact, lls, nlls
from hacktdoa.util import generate_heatmap, generate_hyperbola

from hacktdoa.geodesy.geodesy import SPEED_OF_LIGHT, latlon2xy

def linoptim(sensors, tdoas):
    """
    Obtain the position by linear methods

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)

    Returns:
    np.array([lat,lon]) with the resulting latitude and longitude
    """

    sensors_llh = sensors[["latitude", "longitude", "height"]].to_numpy()
    reference_c = np.mean(sensors_llh, axis=0)

    sensors_xyz = geodesy.latlon2xy(
        sensors_llh[:, 0], sensors_llh[:, 1], reference_c[0], reference_c[1]
    )

    if sensors.shape[0] > 3:
        res = lls.lls(sensors_xyz, tdoas)
        if res.shape[0] > 1:
            print("[WARNING]: Multiple solutions")
    else:
        res = exact.fang(sensors_xyz, tdoas)

    return geodesy.xy2latlon(res[:,0], res[:,1], reference_c[0], reference_c[1]).squeeze()


def brutefoptim(
    sensors,
    tdoas,
    combinations,
    ltrange=2,
    lnrange=2,
    step=0.05,
    epsilon=1e-4,
    maxiter=10,
    workers=1,
):
    """
    Obtain the position by brute force

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)
    combinations: list with sensor combinations when computing tdoa pairs

    Returns:
    np.array([lat,lon, height]) with the resulting latitude and longitude
    """

    sensors_llh = sensors[["latitude", "longitude", "height"]].to_numpy()
    X0 = np.mean(sensors_llh, axis=0)
    altitude = X0[2]

    sensors_ecef = geodesy.llh2ecef(
        sensors[["latitude", "longitude", "height"]].to_numpy()
    )
    sensors_mean = np.mean(sensors_ecef, axis=0)

    optimfun = lambda X: nlls.nlls_llh(
        X, altitude, sensors_ecef - sensors_mean, sensors_mean, tdoas, combinations
    )

    Xr = np.array([X0[0], X0[1]])
    F_prev = None
    lt = ltrange
    ln = lnrange
    st = step
    Ns = int(2 * ltrange / step)
    for i in range(maxiter):
        llrange = (slice(Xr[0] - lt, Xr[0] + lt, st), slice(Xr[1] - ln, Xr[1] + ln, st))

        summary = optimize.brute(
            optimfun, llrange, full_output=True, finish=None, workers=workers
        )

        # We update all the values for the next iteration
        if F_prev is None:
            Xr = summary[0]
            F_prev = summary[1]

            lt = lt * 0.1
            ln = ln * 0.1
            st = 2 * lt / Ns
        else:
            Xr = summary[0]
            F = summary[1]

            if np.abs((F - F_prev) / F) < epsilon:
                return Xr

            F_prev = F

            lt = lt * 0.1
            ln = ln * 0.1
            st = 2 * lt / Ns

    print("[WARNING]: Reached maximum number of iterations")
    return Xr


def nonlinoptim(sensors, tdoas, combinations, llh=None):
    """
    Obtain the position by non linear methods

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)
    combinations: array with combinations per sensor
    LLH0: Initial guess for latitude, longitude and altitude

    Returns:
    np.array([lat,lon, height]) with the resulting latitude and longitude
    """

    sensors_ecef = geodesy.llh2ecef(
        sensors[["latitude", "longitude", "height"]].to_numpy()
    )
    sensors_mean = np.mean(sensors_ecef, axis=0)

    sensors_ecef = sensors_ecef - sensors_mean
    optimfun = lambda X: nlls.nlls(X, sensors_ecef, tdoas, combinations)

    # Minimization routine
    # If no initial point is given we start at the center
    if llh is None:
        X0 = np.zeros(shape=(3, 1))
    else:
        X0 = (geodesy.llh2ecef(llh.reshape(-1,3)) - sensors_mean).reshape(3,1)

    summary = optimize.minimize(optimfun, X0, method="BFGS")

    res = np.array(summary.x, copy=False).reshape(-1, 3)
    return geodesy.ecef2llh(res + sensors_mean).squeeze()

def nonlinoptim_enu(sensors, tdoas, combinations, dimensions=2, enu=None):
    """
    Obtain the position by non linear methods

    Parameters:
    sensors: numpy matrix of Nx3 with (east, north, up) parameters
    tdoas: array of tdoa values of shape(n-1,1)
    combinations: array with combinations per sensor
    LLH0: Initial guess for latitude, longitude and altitude

    Returns:
    np.array([lat,lon, height]) with the resulting latitude and longitude
    """

    sensors_mean = np.mean(sensors, axis=0)

    sensors = sensors - sensors_mean
    optimfun = lambda X: nlls.nlls(X, sensors, tdoas, combinations)

    # Minimization routine
    # If no initial point is given we start at the center
    if enu is None:
        X0 = np.zeros(shape=(3,1))
    else:
        X0 = (enu.reshape(-1,3) - sensors_mean).reshape(3,1)

    summary = optimize.minimize(optimfun, X0, method="BFGS")

    if dimensions == 2:
        res = np.array(summary.x, copy=False).reshape(-1, 2)
    elif dimensions == 3:
        res = np.array(summary.x, copy=False).reshape(-1, 3)
    else:
        raise RuntimeError("[ERROR]: Only valid dimensions are 2 or 3.")

    return (res + sensors_mean).squeeze()
