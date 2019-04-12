import numpy as np
'''
    Functions for calculating angles, dihedrals planes
'''


def vectorNorm(vec, axis=None):
    ''' Returns vector norm. Singleton in dimension along which the norm was made, so that numpy math can proceed'''
    vecShape = len(vec.shape)
    # do along last axis if none is given
    if not axis:
        axis = vecShape - 1

    if axis >= vecShape:
        raise IndexError("Attempted to sum along axis {}, but the vector has shape {}".format(axis, vecShape))
    return np.expand_dims(np.sqrt((vec ** 2).sum(axis=axis)), axis)


def angleFromVectors(v1, v2):
    axis = len(v1.shape) - 1
    nv1 = v1 / vectorNorm(v1)
    nv2 = v2 / vectorNorm(v2)
    return np.arccos(np.clip((nv1 * nv2).sum(axis=axis), -1.0, 1.0))


def dihedralFromVectors(v1, v2, v3):
    ''' Calculates the dihedral angle between 3 vectors, with v1, v2, v3 in sequence.
        DOES NOT ACCOUNT FOR PBC

        The math here is taken from Rahul on stackexchange
        https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    '''
    axis = len(v1.shape) - 1

    # normalize in case not done so yet
    nv1 = v1 / vectorNorm(v1)
    nv2 = v2 / vectorNorm(v2)
    nv3 = v3 / vectorNorm(v3)

    # calculate normal vectors to the planes defined by (v1, v2) and (v2, v3)
    plane1NormalVec = np.cross(nv1, nv2)
    plane2NormalVec = np.cross(nv2, nv3)

    # calculate coordinates in an orthonormal frame
    x = (plane1NormalVec * plane2NormalVec).sum(axis=axis)
    y = (np.cross(plane1NormalVec, nv2) * plane2NormalVec).sum(axis=axis)
    return np.arctan2(y, x)


def dihedralFromPoints(p1, p2, p3, p4):
    ''' Calculates the dihedral angle between 4 points with p1,p2,p3,p4 in sequence.
        DOES NOT ACCOUNT FOR PBC
        '''
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    return dihedralFromVectors(v1, v2, v3)


print(__name__)
