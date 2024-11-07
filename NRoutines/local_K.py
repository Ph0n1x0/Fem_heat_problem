import numpy as np 
from NRoutines.kinematics import elast_diff_2d, shape_quad4
from NRoutines.gauss_functions import gauss_1d, gauss_nd


def uel4nquad(coord, params):
    """Quadrilateral element with 4 nodes for 2D heat transfer

    Parameters
    ----------
    coord : ndarray
        Coordinates for the nodes of the element (4, 2).
    params : float
        Conductivity parameter `k` (scalar value) for heat transfer.

    Returns
    -------
    stiff_mat : ndarray
        Local stiffness matrix for the element (4, 4).
    """
    # Inicializar matriz de rigidez local
    stiff_mat = np.zeros([4, 4])
    
    # Conductividad térmica
    k = params  # `params` ahora se considera como la conductividad térmica, un escalar

    # Obtener puntos de Gauss y sus pesos para integración numérica
    gpts, gwts = gauss_nd(2)
    
    # Loop sobre cada punto de Gauss
    for cont in range(gpts.shape[0]):
        r, s = gpts[cont, :]
        
        # Obtener derivadas de funciones de forma y determinante usando heat_diff_2d
        dNdx, det = elast_diff_2d(r, s, coord, shape_quad4)
        
        # Ensamblaje de la matriz de rigidez usando la conductividad térmica `k`
        factor = det * gwts[cont] * k
        stiff_mat += factor * (dNdx.T @ dNdx)
        
    return stiff_mat


def uel3ntria(coord, params):
    """Triangular element with 3 nodes for 2D heat transfer

    Parameters
    ----------
    coord : ndarray
        Coordinates for the nodes of the element (3, 2).
    params : float
        Conductivity parameter `k` (scalar value) for heat transfer.

    Returns
    -------
    stiff_mat : ndarray
        Local stiffness matrix for the element (3, 3).
    """
    # Inicializar matriz de rigidez local
    stiff_mat = np.zeros([3, 3])
    
    # Conductividad térmica
    k = params  # `params` se considera como la conductividad térmica, un escalar
    
    # Extraer coordenadas de los nodos
    x1, y1 = coord[0]
    x2, y2 = coord[1]
    x3, y3 = coord[2]
    
    # Cálculo del área del triángulo
    A = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Coeficientes de las funciones de forma
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    
    # Ensamblaje de la matriz de rigidez local
    factor = k / (4 * A)
    stiff_mat[0, 0] = b1**2 + c1**2
    stiff_mat[0, 1] = b1 * b2 + c1 * c2
    stiff_mat[0, 2] = b1 * b3 + c1 * c3
    stiff_mat[1, 0] = b2 * b1 + c2 * c1
    stiff_mat[1, 1] = b2**2 + c2**2
    stiff_mat[1, 2] = b2 * b3 + c2 * c3
    stiff_mat[2, 0] = b3 * b1 + c3 * c1
    stiff_mat[2, 1] = b3 * b2 + c3 * c2
    stiff_mat[2, 2] = b3**2 + c3**2
    
    # Multiplicar por el factor de conductividad y área
    stiff_mat *= factor
    
    return stiff_mat



