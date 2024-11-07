# -*- coding: utf-8 -*-
"""
Postprocessor subroutines
-------------------------

This module contains functions to postprocess results.

"""
import numpy as np
import solidspy.femutil as fe
import solidspy.uelutil as uel
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from NRoutines.kinematics import eletype, elast_diff_2d, shape_quad4, jacoper
from NRoutines.gauss_functions import gauss_1d, gauss_nd



# Set plotting defaults
gray = '#757575'
plt.rcParams['image.cmap'] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.color"] = gray
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.color"] = gray
plt.rcParams["ytick.color"] = gray
plt.rcParams["axes.labelcolor"] = gray
plt.rcParams["axes.edgecolor"] = gray
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


#%% Plotting routines
def fields_plot(elements, nodes, temperature, ele_type ,gradients=None, fluxes=None, ):
    """Plot contours for temperature, temperature gradients, and heat flux

    Parameters
    ----------
    nodes : ndarray (float)
        Array with number and nodes coordinates:
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node numbers that correspond to each element.
    temperature : ndarray (float)
        Array with the temperature values at the nodes.
    gradients : ndarray (float)
        Array with temperature gradient field in the nodes (optional).
    fluxes : ndarray (float)
        Array with heat flux field in the nodes (optional).
    """
    # Plot the temperature field
    plot_node_field(ele_type,temperature, nodes, elements, title=["Temperature"],
                    figtitle=["Temperature Distribution"])

    if gradients is not None:
        plot_node_field(ele_type,gradients, nodes, elements,
                        title=["Gradient x", "Gradient y"],
                        figtitle=["Temperature Gradient x", "Temperature Gradient y"])

    if fluxes is not None:
        plot_node_field(ele_type,fluxes, nodes, elements,
                        title=["Flux x", "Flux y"],
                        figtitle=["Heat Flux x", "Heat Flux y"])

def plot_node_field(ele_type,field, nodes, elements, plt_type="contourf", levels=12,
                    savefigs=False, title=None, figtitle=None, filename=None):
    """Plot the nodal field using a triangulation

    Parameters
    ----------
    field : ndarray (float)
        Array with the field to be plotted. The number of columns
        determine the number of plots.
    nodes : ndarray (float)
        Array with node coordinates: `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node indices for each element.
    plt_type : string (optional)
        Plot the field as one of the options: `pcolor` or `contourf`.
    levels : int (optional)
        Number of levels to be used in `contourf`.
    savefigs : bool (optional)
        Whether to save the figure.
    title : list of strings (optional)
        Titles of the plots.
    figtitle : list of strings (optional)
        Titles of the plotting windows.
    filename : list of strings (optional)
        Filenames to save the figures. Used when `savefigs=True`.
    """
    tri = mesh2tri(nodes, elements, ele_type)
    if len(field.shape) == 1:
        nfields = 1
    else:
        _, nfields = field.shape
    if title is None:
        title = ["" for _ in range(nfields)]
    if figtitle is None:
        figs = plt.get_fignums()
        nfigs = len(figs)
        figtitle = [cont + 1 for cont in range(nfigs, nfigs + nfields)]
    if filename is None:
        filename = [f"output{cont}.pdf" for cont in range(nfields)]

    for cont in range(nfields):
        current_field = field if nfields == 1 else field[:, cont]
        plt.figure(figtitle[cont])
        tri_plot(tri, current_field, title=title[cont], levels=levels,
                 plt_type=plt_type, savefigs=savefigs, filename=filename[cont])
        if savefigs:
            plt.savefig(filename[cont])

def tri_plot(tri, field, title="", levels=12, savefigs=False,
             plt_type="contourf", filename="solution_plot.pdf"):
    """Plot contours over a triangulation

    Parameters
    ----------
    tri : Triangulation
        Triangulation object for plotting.
    field : ndarray (float)
        Array with data to be plotted for each node.
    title : string (optional)
        Title of the plot.
    levels : int (optional)
        Number of levels for `contourf`.
    savefigs : bool (optional)
        Whether to save the figure.
    plt_type : string (optional)
        Plot type: `pcolor` or `contourf`.
    filename : string (optional)
        Filename to save the figure.
    """
    if plt_type == "pcolor":
        disp_plot = plt.tripcolor
    elif plt_type == "contourf":
        disp_plot = plt.tricontourf
    disp_plot(tri, field, levels)
    plt.title(title)
    plt.colorbar(orientation='vertical')
    plt.axis("image")
    if savefigs:
        plt.savefig(filename)

#%% Auxiliar functions for plotting
def mesh2tri(nodes, elements,ele_type):
    """Generate a matplotlib.tri.Triangulation object from the mesh

    Parameters
    ----------
    nodes : ndarray (float)
        Array with node coordinates: `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node indices for each element.

    Returns
    -------
    tri : Triangulation
        An unstructured triangular grid consisting of points and triangles.
    """
    coord_x = nodes[:, 1]
    coord_y = nodes[:, 2]
    triangs = []

    if ele_type == "quad":
        for elem in elements:
            # Decompose the 4-noded quadrilateral element into two triangles
            triangs.append([elem[3], elem[4], elem[5]])  # First triangle
            triangs.append([elem[5], elem[6], elem[3]])  # Second triangle
    elif ele_type == "triang":
        for elem in elements:
            # Cada elemento ya es un triángulo con 3 nodos
            triangs.append([elem[3], elem[4], elem[5]])


    tri = Triangulation(coord_x, coord_y, np.array(triangs))
    return tri



#%% Auxiliar variables computation
def complete_disp(bc_array, nodes, sol):
    """
    Fill the temperature vector with imposed and computed values.

    Parameters
    ----------
    bc_array : ndarray (int)
        Indicates if the node has any type of boundary conditions
        applied to it (-1 for imposed temperature, >= 0 for free nodes).
    sol : ndarray (float)
        Array with the computed temperatures for free nodes.
    nodes : ndarray (float)
        Array with node numbers and coordinates.

    Returns
    -------
    sol_complete : ndarray (float)
        Array with temperatures for all nodes, including imposed values.
    """
    nnodes = nodes.shape[0]
    sol_complete = np.zeros(nnodes, dtype=float)
    for i in range(nnodes):
        if bc_array[i] == -1:  # Imposed temperature (boundary condition)
            sol_complete[i] = 0.0
        else:
            sol_complete[i] = sol[bc_array[i]]  # Use the solution value
    return sol_complete

def stdm3Ntria_heat(coord):
    """Gradient of shape functions for a 3-noded triangular element.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the nodes of the element (3, 2).

    Returns
    -------
    ddet : float
        Determinant of the Jacobian evaluated at the centroid.
    gradN : ndarray
        Gradient of shape functions evaluated at the centroid.
    """
    # Extraer coordenadas de los nodos
    x1, y1 = coord[0]
    x2, y2 = coord[1]
    x3, y3 = coord[2]
    
    # Cálculo del área del triángulo
    A = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Gradientes de las funciones de forma en el triángulo (constantes)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    gradN = np.array([[b1, c1], [b2, c2], [b3, c3]]) / (2 * A)
    
    return 2 * A, gradN  # Retornamos el doble del área como el determinante

def stdm4NQ_heat(r, s, coord):
    """Gradient of shape functions for a 4-noded quad element

    Parameters
    ----------
    r : float
      r component in the natural space.
    s : float
      s component in the natural space.
    coord : ndarray
      Coordinates of the nodes of the element (4, 2).

    Returns
    -------
    ddet : float
      Determinant of the Jacobian evaluated at (r, s).
    gradN : ndarray
      Gradient of shape functions evaluated at (r, s).
    """
    nn = 4
    dhdxi = 0.25 * np.array([
        [s - 1, -s + 1, s + 1, -s - 1],
        [r - 1, -r - 1, r + 1, -r + 1]
    ])
    ddet, jaco_inv = jacoper(dhdxi, coord)
    gradN = np.dot(jaco_inv, dhdxi)
    return ddet, gradN

def temperature_gradients_nodes(nodes, elements, mats, UC):
    """
    Compute averaged temperature gradients and heat flux at nodes using all Gauss points.

    Parameters
    ----------
    nodes : ndarray (float)
        Array with node coordinates.
    elements : ndarray (int)
        Array with the node indices for each element.
    mats : ndarray (float)
        Array with material properties, including thermal conductivity.
    UC : ndarray (float)
        Array with complete temperature values at all nodes.

    Returns
    -------
    gradients_nodes : ndarray
        Temperature gradients evaluated at the nodes.
    flux_nodes : ndarray
        Heat flux evaluated at the nodes.
    """
    nelems = elements.shape[0]
    nnodes = nodes.shape[0]
    nnodes_elem=4

    elcoor = np.zeros([nnodes_elem, 2])
    gradients_nodes = np.zeros([nnodes, 2])
    flux_nodes = np.zeros([nnodes, 2])
    el_nodes = np.zeros([nnodes], dtype=int)

    for i in range(nelems):
        # Get thermal conductivity from material properties
        k = mats[int(elements[i, 1]), 0]

        # Get the coordinates of the element nodes
        for j in range(nnodes_elem):
            node_index = elements[i, j + 3]
            elcoor[j, 0] = nodes[node_index, 1]  # x-coordinate
            elcoor[j, 1] = nodes[node_index, 2]  # y-coordinate

        # Extract temperatures at the element nodes
        temp = np.array([UC[elements[i, j + 3]] for j in range(nnodes_elem)])

        # Compute temperature gradients at all Gauss points
        gradients = np.zeros([4, 2])  # To store gradients at 4 Gauss points
        XP, XW = gauss_nd(2)
        for gp in range(4):
            ri, si = XP[gp, 0], XP[gp, 1]
            _, gradN = stdm4NQ_heat(ri, si, elcoor)
            gradients[gp, :] = np.dot(gradN, temp)

        # Average temperature gradients over the Gauss points
        avg_gradient = np.mean(gradients, axis=0)
        flux = -k * avg_gradient  # Compute heat flux using Fourier's law

        # Accumulate gradients and fluxes at each node
        for j in range(nnodes_elem):
            node_index = elements[i, j + 3]
            gradients_nodes[node_index, :] += avg_gradient
            flux_nodes[node_index, :] += flux
            el_nodes[node_index] += 1

    # Average the gradients and fluxes
    for i in range(nnodes):
        if el_nodes[i] > 0:
            gradients_nodes[i, :] /= el_nodes[i]
            flux_nodes[i, :] /= el_nodes[i]

    return gradients_nodes, flux_nodes

def temperature_gradients_nodes_tri(nodes, elements, mats, UC):
    """
    Compute averaged temperature gradients and heat flux at nodes for 3-noded triangular elements.

    Parameters
    ----------
    nodes : ndarray (float)
        Array with node coordinates.
    elements : ndarray (int)
        Array with the node indices for each element.
    mats : ndarray (float)
        Array with material properties, including thermal conductivity.
    UC : ndarray (float)
        Array with complete temperature values at all nodes.

    Returns
    -------
    gradients_nodes : ndarray
        Temperature gradients evaluated at the nodes.
    flux_nodes : ndarray
        Heat flux evaluated at the nodes.
    """
    nnodes_elem = 3  # Triangular elements with 3 nodes
    nelems = elements.shape[0]
    nnodes = nodes.shape[0]

    elcoor = np.zeros([nnodes_elem, 2])
    gradients_nodes = np.zeros([nnodes, 2])
    flux_nodes = np.zeros([nnodes, 2])
    el_nodes = np.zeros([nnodes], dtype=int)

    for i in range(nelems):
        # Conductividad térmica
        k = mats[0, 0]  


        # Obtener las coordenadas de los nodos del elemento
        for j in range(nnodes_elem):
            node_index = elements[i, j + 3]
            elcoor[j, 0] = nodes[node_index, 1]  # coordenada x
            elcoor[j, 1] = nodes[node_index, 2]  # coordenada y

        # Extraer temperaturas en los nodos del elemento
        temp = np.array([UC[elements[i, j + 3]] for j in range(nnodes_elem)])

        # Calcular el gradiente de temperatura en el centroide del triángulo
        _, gradN = stdm3Ntria_heat(elcoor)
        gradient = np.dot(gradN.T, temp)
        flux = -k * gradient  # Flujo de calor usando la ley de Fourier

        # Acumular gradientes y flujos en cada nodo
        for j in range(nnodes_elem):
            node_index = elements[i, j + 3]
            gradients_nodes[node_index, :] += gradient
            flux_nodes[node_index, :] += flux
            el_nodes[node_index] += 1

    # Promediar los gradientes y flujos
    for i in range(nnodes):
        if el_nodes[i] > 0:
            gradients_nodes[i, :] /= el_nodes[i]
            flux_nodes[i, :] /= el_nodes[i]

    return gradients_nodes, flux_nodes



#%% Doc-testing
if __name__ == "__main__":
    import doctest
    doctest.testmod()

