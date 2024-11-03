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
from NRoutines.femutil import eletype, elast_diff_2d, shape_quad4
from NRoutines.gaussutil import gauss_1d, gauss_nd



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
def fields_plot(elements, nodes, disp, E_nodes=None, S_nodes=None):
    """Plot contours for displacements, strains and stresses

    Parameters
    ----------
    nodes : ndarray (float)
        Array with number and nodes coordinates:
         `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    disp : ndarray (float)
        Array with the displacements.
    E_nodes : ndarray (float)
        Array with strain field in the nodes.
    S_nodes : ndarray (float)
        Array with stress field in the nodes.

    """
    # Check for structural elements in the mesh
    struct_pos = 5 in elements[:, 1] or \
             6 in elements[:, 1] or \
             7 in elements[:, 1]
    if struct_pos:
        # Still not implemented visualization for structural elements
        print(disp)
    else:
        plot_node_field(disp, nodes, elements, title=[r"$u_x$", r"$u_y$"],
                        figtitle=["Horizontal displacement",
                                  "Vertical displacement"])
        if E_nodes is not None:
            plot_node_field(E_nodes, nodes, elements,
                            title=[r"$\epsilon_{xx}$",
                                   r"$\epsilon_{yy}$",
                                   r"$\gamma_{xy}$",],
                            figtitle=["Strain epsilon-xx",
                                      "Strain epsilon-yy",
                                      "Strain gamma-xy"])
        if S_nodes is not None:
            plot_node_field(S_nodes, nodes, elements,
                            title=[r"$\sigma_{xx}$",
                                   r"$\sigma_{yy}$",
                                   r"$\tau_{xy}$",],
                            figtitle=["Stress sigma-xx",
                                      "Stress sigma-yy",
                                      "Stress tau-xy"])


def tri_plot(tri, field, title="", levels=12, savefigs=False,
             plt_type="contourf", filename="solution_plot.pdf"):
    """Plot contours over triangulation

    Parameters
    ----------
    tri : ndarray (float)
        Array with number and nodes coordinates:
        `number coordX coordY BCX BCY`
    field : ndarray (float)
        Array with data to be plotted for each node.
    title : string (optional)
        Title of the plot.
    levels : int (optional)
        Number of levels to be used in ``contourf``.
    savefigs : bool (optional)
        Allow to save the figure.
    plt_type : string (optional)
        Plot the field as one of the options: ``pcolor`` or
        ``contourf``
    filename : string (optional)
        Filename to save the figures.
    """
    if plt_type == "pcolor":
        disp_plot = plt.tripcolor
    elif plt_type == "contourf":
        disp_plot = plt.tricontourf
    disp_plot(tri, field, levels, shading="gouraud")
    plt.title(title)
    plt.colorbar(orientation='vertical')
    plt.axis("image")
    if savefigs:
        plt.savefig(filename)


def plot_node_field(field, nodes, elements, plt_type="contourf", levels=12,
                    savefigs=False, title=None, figtitle=None,
                    filename=None):
    """Plot the nodal displacement using a triangulation

    Parameters
    ----------
    field : ndarray (float)
          Array with the field to be plotted. The number of columns
          determine the number of plots.
    nodes : ndarray (float)
        Array with number and nodes coordinates
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each  element.
    plt_type : string (optional)
        Plot the field as one of the options: ``pcolor`` or
        ``contourf``.
    levels : int (optional)
        Number of levels to be used in ``contourf``.
    savefigs : bool (optional)
        Allow to save the figure.
    title : Tuple of strings (optional)
        Titles of the plots. If not provided the plots will not have
        a title.
    figtitle : Tuple of strings (optional)
        Titles of the plotting windows. If not provided the
        windows will not have a title.
    filename : Tuple of strings (optional)
        Filenames to save the figures. Only used when `savefigs=True`.
        If not provided the name of the figures would be "outputk.pdf",
        where `k` is the number of the column.
    """
    tri = mesh2tri(nodes, elements)
    if len(field.shape) == 1:
        nfields = 1
    else:
        _, nfields = field.shape
    if title is None:
        title = ["" for cont in range(nfields)]
    if figtitle is None:
        figs = plt.get_fignums()
        nfigs = len(figs)
        figtitle = [cont + 1 for cont in range(nfigs, nfigs + nfields)]
    if filename is None:
        filename = ["output{}.pdf".format(cont) for cont in range(nfields)]
    for cont in range(nfields):
        if nfields == 1:
            current_field = field
        else:
            current_field = field[:, cont]
        plt.figure(figtitle[cont])
        tri_plot(tri, current_field, title=title[cont], levels=levels,
                 plt_type=plt_type, savefigs=savefigs,
                 filename=filename[cont])
        if savefigs:
            plt.savefig(filename[cont])


def plot_truss(nodes, elements, mats, stresses=None, max_val=4,
               min_val=0.5, savefigs=False, title=None, figtitle=None,
               filename=None):
    """Plot a truss and encodes the stresses in a colormap

    Parameters
    ----------
    UC : (nnodes, 2) ndarray (float)
      Array with the displacements.
    nodes : ndarray (float)
        Array with number and nodes coordinates
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each  element.
    mats : ndarray (float)
        Array with material profiles.
    loads : ndarray (float)
        Array with loads.
    tol : float (optional)
        Minimum difference between cross-section areas of the members
        to be considered different.
    savefigs : bool (optional)
        Allow to save the figure.
    title : Tuple of strings (optional)
        Titles of the plots. If not provided the plots will not have
        a title.
    figtitle : Tuple of strings (optional)
        Titles of the plotting windows. If not provided the
        windows will not have a title.
    filename : Tuple of strings (optional)
        Filenames to save the figures. Only used when `savefigs=True`.
        If not provided the name of the figures would be "outputk.pdf",
        where `k` is the number of the column.

    """
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    if stresses is None:
        scaled_stress = np.ones_like(elements[:, 0])
    else:
        max_stress = max(-stresses.min(), stresses.max())
        scaled_stress = 0.5*(stresses + max_stress)/max_stress
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    plt.figure(figtitle)
    for elem in elements:
        ini, end = elem[3:]
        color = plt.cm.seismic(scaled_stress[elem[0]])
        plt.plot([nodes[ini, 1], nodes[end, 1]],
                 [nodes[ini, 2], nodes[end, 2]],
                 color=color, lw=widths[elem[2]])

    if title is None:
        title = ''
    if figtitle is None:
        figtitle = ""
    if filename is None:
        filename = "output.pdf"
    plt.title(title)
    plt.axis("image")
    if savefigs:
        plt.savefig(filename)


#%% Auxiliar functions for plotting
def mesh2tri(nodes, elements):
    """Generate a  matplotlib.tri.Triangulation object from the mesh

    Parameters
    ----------
    nodes : ndarray (float)
      Array with number and nodes coordinates:
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
      Array with the node number for the nodes that correspond to each
      element.

    Returns
    -------
    tri : Triangulation
        An unstructured triangular grid consisting of npoints points
        and ntri triangles.

    """
    coord_x = nodes[:, 1]
    coord_y = nodes[:, 2]
    triangs = []
    for elem in elements:
        if elem[1] == 1:
            triangs.append(elem[[3, 4, 5]])
            triangs.append(elem[[5, 6, 3]])
        if elem[1] == 2:
            triangs.append(elem[[3, 6, 8]])
            triangs.append(elem[[6, 7, 8]])
            triangs.append(elem[[6, 4, 7]])
            triangs.append(elem[[7, 5, 8]])
        if elem[1] == 3:
            triangs.append(elem[3:])
        if elem[1] == 4:
            triangs.append(elem[[3, 7, 11]])
            triangs.append(elem[[7, 4, 8]])
            triangs.append(elem[[3, 11, 10]])
            triangs.append(elem[[7, 8, 11]])
            triangs.append(elem[[10, 11, 9]])
            triangs.append(elem[[11, 8, 5]])
            triangs.append(elem[[10, 9, 6]])
            triangs.append(elem[[11, 5, 9]])

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

def grad_el4(elcoor, temp):
    """Compute the temperature gradients for a quadrilateral element
    at its nodes.
    
    Parameters
    ----------
    elcoor : ndarray (float)
        Coordinates of the element nodes (4x2 array)
    temp : ndarray (float)
        Array with nodal temperatures (4 values)
        
    Returns
    -------
    grad : ndarray
        Temperature gradients at the nodes (4x2 array)
        Each row contains [dT/dx, dT/dy] for each node
    """
    # Gauss points and weights for 2x2 quadrature
    gauss = np.array([
        [-0.577350269189626, -0.577350269189626],
        [0.577350269189626, -0.577350269189626],
        [0.577350269189626, 0.577350269189626],
        [-0.577350269189626, 0.577350269189626]])
    
    # Initialize gradient array
    grad = np.zeros((4, 2))
    
    # Shape functions derivatives with respect to xi and eta at nodes
    dNdxi = np.array([
        [-0.25*(1.0 - eta), 0.25*(1.0 - eta), 0.25*(1.0 + eta), -0.25*(1.0 + eta)]
        for eta in [-1.0, -1.0, 1.0, 1.0]])
    dNdeta = np.array([
        [-0.25*(1.0 - xi), -0.25*(1.0 + xi), 0.25*(1.0 + xi), 0.25*(1.0 - xi)]
        for xi in [-1.0, 1.0, 1.0, -1.0]])
    
    # Calculate gradients at nodes using extrapolation from Gauss points
    for i in range(4):  # For each node
        xi, eta = gauss[i]
        
        # Derivatives of shape functions at current Gauss point
        dN_dxi = np.array([
            -0.25*(1.0 - eta), 
            0.25*(1.0 - eta), 
            0.25*(1.0 + eta), 
            -0.25*(1.0 + eta)])
        dN_deta = np.array([
            -0.25*(1.0 - xi), 
            -0.25*(1.0 + xi), 
            0.25*(1.0 + xi), 
            0.25*(1.0 - xi)])
        
        # Compute Jacobian matrix
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, elcoor[:, 0])   # dx/dxi
        J[0, 1] = np.dot(dN_dxi, elcoor[:, 1])   # dy/dxi
        J[1, 0] = np.dot(dN_deta, elcoor[:, 0])  # dx/deta
        J[1, 1] = np.dot(dN_deta, elcoor[:, 1])  # dy/deta
        
        # Compute inverse of Jacobian
        detJ = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
        invJ = np.array([
            [J[1, 1]/detJ, -J[0, 1]/detJ],
            [-J[1, 0]/detJ, J[0, 0]/detJ]])
        
        # Compute derivatives with respect to x and y
        dN = np.zeros((2, 4))
        dN[0, :] = invJ[0, 0]*dN_dxi + invJ[0, 1]*dN_deta  # dN/dx
        dN[1, :] = invJ[1, 0]*dN_dxi + invJ[1, 1]*dN_deta  # dN/dy
        
        # Calculate temperature gradients
        grad[i, 0] = np.dot(dN[0, :], temp)  # dT/dx
        grad[i, 1] = np.dot(dN[1, :], temp)  # dT/dy
        
    return grad, None  # None added to match the original function structure

def heat_nodes(nodes, elements, mats, temp_complete):
    """Compute averaged temperature gradients and heat fluxes at nodes.
    
    Parameters
    ----------
    nodes : ndarray (float)
        Array with nodes coordinates.
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    mats : ndarray (float)
        Array with material thermal conductivity.
    temp_complete : ndarray (float)
        Array with the temperatures. This contains both the
        computed and imposed values.
    
    Returns
    -------
    grad_nodes : ndarray
        Temperature gradients evaluated at the nodes (dT/dx, dT/dy).
    flux_nodes : ndarray
        Heat fluxes evaluated at the nodes (qx, qy).
    """
    nelems = elements.shape[0]
    nnodes = nodes.shape[0]
    iet = elements[0, 1]
    ndof, nnodes_elem, _ = eletype(iet)
    elcoor = np.zeros([nnodes_elem, 2])
    grad_nodes = np.zeros([nnodes, 2])  # Solo necesitamos dT/dx y dT/dy
    flux_nodes = np.zeros([nnodes, 2])  # Solo necesitamos qx y qy
    el_nodes = np.zeros([nnodes], dtype=int)
    temp = np.zeros([nnodes_elem])  # Solo temperatura (un grado de libertad por nodo)
    IELCON = elements[:, 3:]
    
    for el in range(nelems):
        k =1.0 # Conductividad térmica del material
        elcoor[:, 0] = nodes[IELCON[el, :], 1]
        elcoor[:, 1] = nodes[IELCON[el, :], 2]
        temp[:] = temp_complete[IELCON[el, :]]  # Temperaturas nodales del elemento
        
        # Calcular gradientes en los nodos del elemento
        gradG = grad_el4(elcoor, temp)[0]

                    # Acumular gradientes y flujos en los nodos
        for cont, node in enumerate(IELCON[el, :]):
            grad_nodes[node, 0] += gradG[cont, 0]  # dT/dx
            grad_nodes[node, 1] += gradG[cont, 1]  # dT/dy
            # El flujo de calor es q = -k∇T
            flux_nodes[node, 0] += -k * gradG[cont, 0]  # qx
            flux_nodes[node, 1] += -k * gradG[cont, 1]  # qy
            el_nodes[node] = el_nodes[node] + 1
    
    # Promediar los valores
    for i in range(2):
        grad_nodes[:, i] /= el_nodes
        flux_nodes[:, i] /= el_nodes
        
    return grad_nodes, flux_nodes


#%% Doc-testing
if __name__ == "__main__":
    import doctest
    doctest.testmod()

