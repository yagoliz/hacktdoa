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

import numpy as np

from hacktdoa.geodesy import geodesy


def nlls(X, positions, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contain the ECEF coordinates of the estimated
    transmitter position
    ---
    """

    # Compute all distances to the sensor
    d = geodesy.ecef_distance(positions, X)

    si = combinations[:, 0]
    sj = combinations[:, 1]

    t = d[si] - d[sj]

    err = np.square(tdoas - t)
    F = np.sum(err)

    return F

def nlls_enu(X, positions, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contain the ECEF coordinates of the estimated
    transmitter position
    ---
    """

    # Compute all distances to the sensor
    dim = positions.shape[1]
    d = np.sqrt(np.sum(np.square(positions - X.reshape(-1, dim)), axis=1))

    si = combinations[:, 0]
    sj = combinations[:, 1]

    t = d[si] - d[sj]

    err = np.square(tdoas - t)
    F = np.sum(err)

    return F


def nlls_llh(X, height, positions, positions_mean, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contain the ecef coordinates of the estimated
    transmitter position
    ---
    """

    # In this case, X contains the latitude, longitude and height distribution
    X_ecef = geodesy.llh2ecef(np.append(X, height).reshape(1, 3)) - positions_mean

    # Compute all distances to the sensor
    d = geodesy.ecef_distance(positions, X_ecef)

    si = combinations[:, 0]
    sj = combinations[:, 1]

    t = d[si] - d[sj]

    err = np.square(tdoas - t)
    F = np.sum(err)

    return F
