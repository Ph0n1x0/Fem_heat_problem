# -*- coding: utf-8 -*-
"""
Assembly routines
-----------------

Functions to assemble the system of equations for a finite element
analysis.

"""
import numpy as np
from scipy.sparse import coo_matrix
import solidspy.uelutil as ue
import solidspy.femutil as fem
from NRoutines.femutil import eletype
from NRoutines.uelutil import elast_quad4


def eqcounter(cons, ndof_node=1):
    """Count active equations for heat problem

    Creates boundary conditions array bc_array for a heat transfer problem.

    Parameters
    ----------
    cons : ndarray
      Array with constraints for each node.

    Returns
    -------
    neq : int
      Number of equations in the system after applying constraints.
    bc_array : ndarray (int)
      Array that maps the nodes with number of equations.

    """
    nnodes = cons.shape[0]
    bc_array = np.full(nnodes, -1, dtype=int)  # Inicializa con -1
    neq = 0
    for i in range(nnodes):
        if cons[i, -1] == 0:  # Si no hay restricción
            bc_array[i] = neq
            neq += 1

    return neq, bc_array


def DME_heat(cons, elements, ndof_node=1, ndof_el_max=4, ndof_el=None):
    """Create assembly array operator for heat transfer problem

    Count active equations, create boundary conditions array `bc_array`
    and the assembly operator `assem_op` for a heat transfer problem.

    Parameters
    ----------
    cons : ndarray
      Array with constraints for each degree of freedom in each node.
    elements : ndarray
      Array with the nodes in each element.
    ndof_node : int, optional
      Number of degrees of freedom per node, set to 1 for temperature.
    ndof_el_max : int, optional
      Maximum degrees of freedom per element, set to 4 for 4-node square.
    ndof_el : callable, optional
      Function that returns degrees of freedom for elements, needed for user elements.

    Returns
    -------
    assem_op : ndarray (int)
      Assembly operator.
    bc_array : ndarray (int)
      Boundary conditions array.
    neq : int
      Number of active equations in the system.

    """
    # Number of elements
    nels = elements.shape[0]
    
    # Initialize assembly operator
    assem_op = np.zeros([nels, ndof_el_max], dtype=int)
    
    # Count active equations and create boundary conditions array
    neq, bc_array = eqcounter(cons, ndof_node=ndof_node)
    
    # Loop over elements to populate assembly operator
    for ele in range(nels):
        iet = elements[ele, 1]  # Element type
        print(iet)
        if ndof_el is None:
            ndof, nnodes, _ = eletype(iet)  # Get DOFs and nodes for the element type
        else:
            ndof = ndof_el(iet)

        assem_op[ele, :nnodes] = bc_array[elements[ele, 3:3 + nnodes]].flatten()
    
    return assem_op, bc_array, neq


def uel_heat(element_nodes, params):
    """
    Calculate the local conductivity matrix K for a 4-node square element in heat transfer.

    Parameters
    ----------
    element_nodes : ndarray
        Array containing the coordinates of the element nodes.
    params : ndarray
        Array with material properties (e.g., thermal conductivity).

    Returns
    -------
    kloc : ndarray
        Local conductivity matrix for the heat element.
    """
    # Conductividad térmica
    k = params[0]  # Asumiendo que params contiene la conductividad térmica

    # Coordenadas de los nodos
    x0, y0 = element_nodes[0][1:-1]
    x1, y1 = element_nodes[1][1:-1]
    x2, y2 = element_nodes[2][1:-1]
    x3, y3 = element_nodes[3][1:-1]

    # Determinante de la matriz jacobiana
    det_J = ((x1 - x0) * (y3 - y0) - (x3 - x0) * (y1 - y0)) / 4.0

    # Matriz de forma B
    B = np.array([
        [-1/(2*det_J),  1/(2*det_J),  1/(2*det_J), -1/(2*det_J)],
        [-1/(2*det_J), -1/(2*det_J),  1/(2*det_J),  1/(2*det_J)]
    ])

    # Calcular la matriz de conductividad térmica local
    kloc = k * det_J * (B.T @ B)

    return kloc


def retriever_heat(elements, mats, nodes, ele, uel=None):
    """Computes the elemental stiffness (conductivity) matrix of element `ele` for heat transfer problems.

    Parameters
    ----------
    elements : ndarray
      Array with the number for the nodes in each element.
    mats : ndarray
      Array with the material profiles.
    nodes : ndarray
      Array with the nodal numbers and coordinates.
    ele : int
      Identifier of the element to be assembled.
    uel : callable, optional
      Function that returns the local stiffness matrix. Defaults to `uel_heat`.

    Returns
    -------
    kloc : ndarray (float)
      Array with the local stiffness (conductivity) matrix.
    """
    # Tipo de elemento
    elem_type = elements[ele, 1]
    # Parámetros del material, aquí obtenemos la conductividad térmica
    params = mats[elements[ele, 2], :]
    # Coordenadas del elemento
    elcoor = nodes[elements[ele, 3:], 1:]
    
    # Usa `uel_heat` si no se especifica otra función
    if uel is None:
        kloc = uel_heat(nodes, mats)
    
    # Solo devolvemos kloc ya que mloc no es necesario
    #kloc = uel(elcoor, params)
    return kloc


def assembler_heat(elements, mats, nodes, neq, DME, uel=None):
    # Assemble the global stiffness matrix KG for heat transfer
    KG = np.zeros((neq, neq))
    nels = elements.shape[0]
    nnodes = 4  # Each element has 4 nodes

    for el in range(nels):
        elcoor = np.zeros([nnodes, 2])
        mat_index = int(elements[el, 2])
        thermal_conductivity = mats[mat_index, 0]  # Extract thermal conductivity

        # Get coordinates of the element nodes
        for j in range(nnodes):
            node_index = elements[el, j+3]
            elcoor[j, 0] = nodes[node_index, 1]  # x-coordinate
            elcoor[j, 1] = nodes[node_index, 2]  # y-coordinate

        # Compute the local stiffness matrix for heat transfer
        kloc = elast_quad4(elcoor, thermal_conductivity)
        dme = DME[el, :]  # Get the DOF mapping for the element

        # Assemble local stiffness into the global stiffness matrix
        for row in range(nnodes):
            glob_row = dme[row]
            if glob_row != -1:  # Check if the row DOF is free
                for col in range(nnodes):
                    glob_col = dme[col]
                    if glob_col != -1:  # Check if the col DOF is free
                        KG[glob_row, glob_col] += kloc[row, col]

    return KG


def dense_assem(elements, mats, nodes, neq, assem_op, uel=None):
    """
    Assembles the global stiffness matrix
    using a dense storing scheme

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats : ndarray (float)
      Array with the material profiles.
    nodes : ndarray (float)
      Array with the nodal numbers and coordinates.
    assem_op : ndarray (int)
      Assembly operator.
    neq : int
      Number of active equations in the system.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.

    Returns
    -------
    kglob : ndarray (float)
      Array with the global stiffness matrix in a dense numpy
      array.

    """
    kglob = np.zeros((neq, neq))
    mglob = np.zeros((neq, neq))
    nels = elements.shape[0]
    for ele in range(nels):
        kloc, mloc = retriever_heat(elements, mats, nodes, ele, uel=uel)
        ndof = kloc.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        kglob[glob_row, glob_col] += kloc[row, col]
                        mglob[glob_row, glob_col] += mloc[row, col]

    return kglob, mglob


def sparse_assem(elements, mats, nodes, neq, assem_op, uel=None):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    The scheme used to assemble is COOrdinate list (COO), and
    it converted to Compressed Sparse Row (CSR) afterward
    for the solution phase [1]_.

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    nodes    : ndarray (float)
      Array with the nodal numbers and coordinates.
    assem_op : ndarray (int)
      Assembly operator.
    neq : int
      Number of active equations in the system.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.

    Returns
    -------
    kglob : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.

    References
    ----------
    .. [1] Sparse matrix. (2017, March 8). In Wikipedia,
        The Free Encyclopedia.
        https://en.wikipedia.org/wiki/Sparse_matrix

    """
    rows = []
    cols = []
    stiff_vals = []
    mass_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        kloc = retriever_heat(elements, mats, nodes, ele, uel=uel)
        ndof = kloc.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        rows.append(glob_row)
                        cols.append(glob_col)
                        stiff_vals.append(kloc[row, col])
    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()
    return stiff


def loadasem(loads, bc_array, neq, ndof_node=1):
    """Assembles the global Right Hand Side Vector

    Parameters
    ----------
    loads : ndarray
      Array with the loads imposed in the system.
    bc_array : ndarray (int)
      Array that maps the nodes with number of equations.
    neq : int
      Number of equations in the system after removing the nodes
      with imposed displacements.

    Returns
    -------
    rhs_vec : ndarray
      Array with the right hand side vector.

    """
    nloads = loads.shape[0]
    rhs_vec = np.zeros([neq])  # Vector RHS inicializado a cero
    
    for cont in range(nloads):
        node = int(loads[cont, 0])  # Obtiene el nodo de la carga
        dof_id = bc_array[node]  # Ahora solo accedemos con un índice

        if dof_id != -1:  # Verifica si el DOF es válido
            rhs_vec[dof_id] += loads[cont, 1]  # Suma la carga al vector RHS

    return rhs_vec






if __name__ == "__main__":
    import doctest
    doctest.testmod()
