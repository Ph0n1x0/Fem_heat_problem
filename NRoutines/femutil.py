 # -*- coding: utf-8 -*-
"""
FEM routines
------------

Functions to compute kinematics variables for the Finite
Element Analysis.

The elements included are:
    1. 4 node bilinear quadrilateral.
    2. 6 node quadratic triangle.
    3. 3 node linear triangle.

The notation used is similar to the one used by Bathe [1]_.


References
----------
.. [1] Bathe, Klaus-Jürgen. Finite element procedures. Prentice Hall,
   Pearson Education, 2006.

"""
import numpy as np
from NRoutines.gaussutil import gauss_nd, gauss_1d


def eletype(eletype):
    """Assigns number to degrees of freedom

    According to iet assigns number of degrees of freedom, number of
    nodes and minimum required number of integration points.

    Parameters
    ----------
    eletype :  int
      Type of element. These are:
        1. 4 node bilinear quadrilateral.
        2. 6 node quadratic triangle.
        3. 3 node linear triangle.
        5. 2 node spring.
        6. 2 node truss element.
        7. 2 node beam (3 DOF per node).
        8. 2 node beam with axial force (3 DOF per node).

    Returns
    -------
    ndof : int
      Number of degrees of freedom for the selected element.
    nnodes : int
      Number of nodes for the selected element.
    ngpts : int
      Number of Gauss points for the selected element.

    """
    elem_id = {
        1: (8, 4, 4),
        2: (12, 6, 7),
        3: (6, 3, 3),
        4: (18, 9, 9),
        5: (4, 2, 3),
        6: (4, 2, 3),
        7: (6, 2, 3),
        8: (6, 2, 3)}
    try:
        return elem_id[eletype]
    except:
        raise ValueError("You entered an invalid type of element.")


#%% Shape functions and derivatives

# Quadrilaterals
def shape_quad4(r, s):
    """
    Shape functions and derivatives for a bilinear element

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.

    Returns
    -------
    N : ndarray (float)
        Array with the shape functions evaluated at the point (r, s).
    dNdr : ndarray (float)
        Array with the derivative of the shape functions evaluated at
        the point (r, s).

    Examples
    --------
    We can check evaluating at two different points, namely (0, 0) and
    (1, 1). Thus

    >>> N, _ = shape_quad4(0, 0)
    >>> N_ex = np.array([
    ...    [1/4, 0, 1/4, 0, 1/4, 0, 1/4, 0],
    ...    [0, 1/4, 0, 1/4, 0, 1/4, 0, 1/4]])
    >>> np.allclose(N, N_ex)
    True

    and

    >>> N, _ = shape_quad4(1, 1)
    >>> N_ex = np.array([
    ...    [0, 0, 0, 0, 1, 0, 0, 0],
    ...    [0, 0, 0, 0, 0, 1, 0, 0]])
    >>> np.allclose(N, N_ex)
    True

    """
    N = 0.25*np.array(
        [(1 - r)*(1 - s),
         (1 + r)*(1 - s),
         (1 + r)*(1 + s),
         (1 - r)*(1 + s)])
    dNdr = 0.25*np.array([
        [s - 1, -s + 1, s + 1, -s - 1],
        [r - 1, -r - 1, r + 1, -r + 1]])
    return N, dNdr


#%% Derivative matrices
def elast_diff_2d(r, s, coord, element):
    """
    Interpolation matrices for elements for plane elasticity

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s).
    det : float
        Determinant of the Jacobian.
    """
    N, dNdr = element(r, s)
    det, jaco_inv = jacoper(dNdr, coord)
    dNdx = jaco_inv @ dNdr
    
    B = dNdx
    return B, det



#%%
def jacoper(dNdr, coord):
    """
    Compute the Jacobian of the transformation evaluated at
    the Gauss point

    Parameters
    ----------
    dNdr : ndarray
      Derivatives of the interpolation function with respect to the
      natural coordinates.
    coord : ndarray
      Coordinates of the nodes of the element (nnodes, ndim).

    Returns
    -------
    jaco_inv : ndarray (ndim, ndim)
      Jacobian of the transformation evaluated at a point.

    """
    jaco = dNdr @ coord
    det = np.linalg.det(jaco)
    if np.isclose(np.abs(det), 0.0):
        msg = "Jacobian close to zero. Check the shape of your elements!"
        raise ValueError(msg)
    jaco_inv = np.linalg.inv(jaco)
    if det < 0.0:
        msg = "Jacobian is negative. Check your elements orientation!"
        raise ValueError(msg)
    return det, jaco_inv


#%% Elemental strains

def str_el4(coord, ul):
    """Compute the strains at each element integration point

    This one is used for 4-noded quadrilateral elements.

    Parameters
    ----------
    coord : ndarray
      Coordinates of the nodes of the element (4, 2).
    ul : ndarray
      Array with displacements for the element.

    Returns
    -------
    epsGT : ndarray
      Strain components for the Gauss points.
    xl : ndarray
      Configuration of the Gauss points after deformation.

    """
    epsG = np.zeros([3, 4])
    xl = np.zeros([4, 2])
    gpts, _ = gauss_nd(2)
    for i in range(gpts.shape[0]):
        ri, si = gpts[i, :]
        H, B, _= elast_diff_2d(ri, si, coord, shape_quad4)
        epsG[:, i] = B @ ul
        xl[i, 0] = np.dot(H[0, ::2], coord[:, 0])
        xl[i, 1] = np.dot(H[0, ::2], coord[:, 0])
    return epsG.T, xl


if __name__ == "__main__":
    import doctest
    doctest.testmod()
