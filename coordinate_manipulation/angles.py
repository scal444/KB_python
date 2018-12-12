import numpy as np

'''
    Functions for calculating angles, dihedrals planes
'''


def normalizeVector(vec, axis=None):

    vecShape = len(vec.shape)
    # do along last axis if none is given
    if not axis:
        axis = vecShape - 1

    if axis >= vecShape:
        raise IndexError("Attempted to sum along axis {}, but the vector has shape {}".format(axis, vecShape))
    return np.sqrt((vec ** 2).sum(axis=axis))


def normalVectorToPlane(vec1, vec2):
    pass
