"""
Preprocessor subroutines
-------------------------

This module contains functions to prepare the main files for simulations with a .msh file 

"""
import numpy as np 
import meshio
import matplotlib.pyplot as plt


def assem_files_w_msh(filename, heat_load, ele_type):

    mesh       = meshio.read(filename)   # Leer malla de gmsh
    points     = mesh.points
    cells      = mesh.cells
    point_data = mesh.point_data
    cell_data  = mesh.cell_data

    if ele_type =="quad":
        #compute nodes
        nodes_array = node_writer(points, point_data)
        nodes_array = boundary_conditions(cells, cell_data, 400 , nodes_array, -1)

        #compute elems
        nf, els1_array = ele_writer(cells, cell_data, "quad", 100, 1 , 0 , 0)
        nini = nf
        nf, els2_array = ele_writer(cells, cell_data, "quad", 200, 1 , 1,  nini)
        els_array = np.vstack((els1_array, els2_array))

        #compute loads
        cargas = loading(cells, cell_data, 500 , heat_load)
    elif ele_type=="triang":
        #compute nodes
        nodes_array = node_writer(points, point_data)
        nodes_array = boundary_conditions(cells, cell_data, 10 , nodes_array, -1)
        #compute elems
        nf, els_array = ele_writer(cells, cell_data, "triangle", 9, 3 , 0 , 0)  #elemnt type=3 for a 3-node triangle
        #compute loads
        cargas = loading(cells, cell_data, 11 , heat_load)


    #save files
    np.savetxt("files/eles.txt" , els_array, fmt="%d")
    np.savetxt("files/loads.txt", cargas, fmt=("%d", "%.6f"))
    np.savetxt("files/nodes.txt", nodes_array , fmt=("%d", "%.4f", "%.4f", "%d"))
    
    #load files
    nodes     = np.loadtxt('files/nodes.txt', ndmin=2)
    mats      = np.loadtxt('files/mater.txt', ndmin=2)
    elements  = np.loadtxt('files/eles.txt', ndmin=2, dtype=int)
    loads     = np.loadtxt('files/loads.txt', ndmin=2)

       
    return nodes, mats, elements, loads

def ele_writer(cells, cell_data, ele_tag, phy_sur,  ele_type, mat_tag, nini):
    """
    Extracts a subset of elements from a complete mesh according to the
    physical surface  phy_sur and writes down the proper fields into an
    elements array.

    Parameters
    ----------
        cell : dictionary
            Dictionary created by meshio with cells information.
        cell_data: dictionary
            Dictionary created by meshio with cells data information.
        ele_tag : string
            Element type according to meshio convention,
            e.g., quad9 or line3.
        phy_sur : int
            Physical surface for the subset.
        ele_type: int
            Element type.
        mat_tag : int
            Material profile for the subset.
        ndof : int
            Number of degrees of freedom for the elements.
        nnode : int
            Number of nodes for the element.
        nini : int
            Element id for the first element in the set.

    Returns
    -------
        nf : int
            Element id for the last element in the set
        els_array : int
            Elemental data.

    """
    eles = cells[ele_tag]
    dict_nnode = {'triangle': 3,
                  'triangle6': 6,
                  'quad': 4}
    nnode = dict_nnode[ele_tag]
    phy_surface = cell_data[ele_tag]['gmsh:physical']
    ele_id = [cont for cont, _ in enumerate(phy_surface[:])
              if phy_surface[cont] == phy_sur]
    els_array = np.zeros([len(ele_id) , 3 + nnode], dtype=int)
    els_array[: , 0] = range(nini , len(ele_id) + nini )
    els_array[: , 1] = ele_type
    els_array[: , 2] = mat_tag
    els_array[: , 3::] = eles[ele_id, :]
    nf = nini + len(ele_id)
    return nf , els_array

def node_writer(points, point_data):
    """Write nodal data for heat transfer analysis as required by SolidsPy

    Parameters
    ----------
    points : ndarray (float)
        Nodal points.
    point_data : dictionary
        Physical data associated with the nodes.

    Returns
    -------
    nodes_array : ndarray (float)
        Array with the nodal data for heat transfer analysis.
    """
    nodes_array = np.zeros([points.shape[0], 4])  # Only 4 columns: node number, x, y, BC
    nodes_array[:, 0] = range(points.shape[0])  # Node number
    nodes_array[:, 1:3] = points[:, :2]  # x and y coordinates
    nodes_array[:, 3] = 0  # Default to 0 (free temperature DOF)
    return nodes_array

def boundary_conditions(cells, cell_data, phy_lin, nodes_array, bc_temp):
    """Impose nodal point boundary conditions for heat transfer analysis

    Parameters
    ----------
    cells : dictionary
        Dictionary created by meshio with cells information.
    cell_data : dictionary
        Dictionary created by meshio with cells data information.
    phy_lin : int
        Physical line where BCs are to be imposed.
    nodes_array : ndarray (float)
        Array with the nodal data to be modified by boundary conditions.
    bc_temp : int
        Boundary condition flag for temperature:
            * -1: restrained (fixed temperature)
            *  0: free (no boundary condition)

    Returns
    -------
    nodes_array : ndarray (float)
        Array with the nodal data after imposing boundary conditions.
    """
    lines = cells["line"]
    # Find lines that correspond to the specified physical line
    phy_line = cell_data["line"]["gmsh:physical"]
    id_boundary = [cont for cont in range(len(phy_line)) if phy_line[cont] == phy_lin]
    nodes_boundary = lines[id_boundary]
    nodes_boundary = nodes_boundary.flatten()
    nodes_boundary = list(set(nodes_boundary))
    nodes_array[nodes_boundary, 3] = bc_temp  # Apply boundary condition
    return nodes_array

def loading(cells, cell_data, phy_lin, P):
    """Impose nodal heat loads for heat transfer analysis

    Parameters
    ----------
    cells : dictionary
        Dictionary created by meshio with cells information.
    cell_data : dictionary
        Dictionary created by meshio with cells data information.
    phy_lin : int
        Physical line where heat loads are to be imposed.
    P : float
        Total heat load to be applied along the line.

    Returns
    -------
    loads : ndarray (float)
        Array with the nodal heat loads. Each row contains:
        [node index, heat load value]
    """
    lines = cells["line"]
    # Find lines that correspond to the specified physical line
    phy_line = cell_data["line"]["gmsh:physical"]
    id_load = [cont for cont in range(len(phy_line)) if phy_line[cont] == phy_lin]
    nodes_load = lines[id_load]
    nodes_load = nodes_load.flatten()
    nodes_load = list(set(nodes_load))
    
    # Distribute the heat load evenly among all nodes
    nloads = len(nodes_load)
    loads = np.zeros((nloads, 2))
    loads[:, 0] = nodes_load  # Node index
    loads[:, 1] = P / nloads  # Heat load value per node

    return loads