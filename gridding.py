import numpy as np
from scipy.fftpack import fftfreq, fftshift
from scipy.ndimage.interpolation import geometric_transform
from scipy import ndimage
import scipy.spatial.qhull as qhull
import itertools

"""
Gridding and projection classes for plotting
and spectral analysis etc.
"""


def interp_weights(xy, uv, d=2):
    """
    Hack to speed up scipy.interpolate.griddata by performing 
    the output grid triangulation (slow) first, then performing the
    interpolation (fast) separately. This and the interpolate() function
    below are borrowed from user Jaime at
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-
    for-multiple-interpolations-between-two-irregular-grids
    
    Background from the original link:
    
    'There are several things going on every time you make a call to
    scipy.interpolate.griddata:

    1. First, a call to sp.spatial.qhull.Delaunay is made to triangulate
    the irregular grid coordinates.
    2. Then, for each point in the new grid,
    the triangulation is searched to find in which triangle (actually, 
    in which simplex, which in your 3D case will be in which tetrahedron)
    does it lay. 
    3. The barycentric coordinates of each new grid point with
    respect to the vertices of the enclosing simplex are computed.
    4. An interpolated values is computed for that grid point, using the
    barycentric coordinates, and the values of the function at the 
    vertices of the enclosing simplex. 
    
    The first three steps are identical 
    for all your interpolations, so if you could store, for each new grid
    point, the indices of the vertices of the enclosing simplex and the
    weights for the interpolation, you would minimize the amount of
    computations by a lot.
    
    Example:
    
    import scipy.interpolate as spint
    import scipy.spatial.qhull as qhull
    import itertools
    
    m, n, d = 3.5e4, 3e3, 2
    # make sure no new grid point is extrapolated
    bounding_cube = np.array(list(itertools.product([0, 1], repeat=d)))
    xyz = np.vstack((bounding_cube,
                 np.random.rand(m - len(bounding_cube), d)))
    f = np.random.rand(m)
    g = np.random.rand(m)
    uvw = np.random.rand(n, d)

    In [2]: vtx, wts = interp_weights(xyz, uvw, d)

    In [3]: np.allclose(interpolate(f, vtx, wts), spint.griddata(xyz, f, uvw))
    Out[3]: True'
    """
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
    """
    Interpolate points to grid using vertices and weights from
    interp_weights(). See that function for example. Generates output
    that is identical to scipy.interpolate.griddata.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret




