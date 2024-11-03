from NRoutines.gaussutil import gauss_1d, gauss_nd
import numpy as np 
from NRoutines.femutil import elast_diff_2d, shape_quad4

def elast_quad4(coord, params):
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