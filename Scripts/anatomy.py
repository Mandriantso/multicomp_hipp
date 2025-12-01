import numpy as np
import math
from scipy.interpolate import splprep, splev

def convert_to_µm(x):
    # return (x/3.77953)*1e2#*0.5
    return x * (96 / 2.54) #* 1e4


def bezier_func(t, control_points):
    '''
        Gets the cartesian coordinates of a point from its bézier parameter t
    '''
    n = len(control_points) - 1
    control_points = np.float64(control_points)
    point = np.array([[0.0, 0.0]])
    for i in range(n+1):
        point += math.comb(n, i) * (t**i) * ((1-t)**(n-i)) * control_points[i]
    return point


def stack_bezier(t, control_points_1, control_points_2):
    '''
        stacks 2 bézier curves together, with the same evolving t € [0,1]
    '''
    if t <= 0.5:
        point = bezier_func(2*t, control_points_1)
    else:
        point = bezier_func(2*t-1, control_points_2)
    return point


def mapping_bezier(x, curve_points, control_points):
    '''
        Redistributes 100 points of a bézier curve equidistantly
    '''
    n_points = len(curve_points) 
    total_length, arcs_lengths = get_shape_length(curve_points) 
    u = x * total_length
    index = 0
    low = 0
    high = n_points - 1

    while low < high:
        index = low + math.ceil(((high - low) / 2))
        if arcs_lengths[index] < u:
            low = index + 1
        else:
            if high != index:
                high = index
            else:
                break

    if arcs_lengths[index] > u:
        index -= 1
        
    if arcs_lengths[index] == u:
        t_prime = index/n_points
    else:
        t_prime = (index + (u - arcs_lengths[index]) / (arcs_lengths[index + 1] - arcs_lengths[index])) / n_points
    
    return stack_bezier(t_prime, control_points[:4], control_points[3:])


def coefficient_dir(seg):
    '''
        Computes the slope of a segment 
    '''
    assert isinstance(seg, np.ndarray), "input must be a numpy array of shape (2, 2)"
    assert np.shape(seg) == (2, 2), "input is of shape {}, must be a numpy array of shape (2, 2)".format(np.shape(seg))
    if (seg[1, 0] - seg[0, 0]) != 0:
        return (seg[1, 1] - seg[0, 1])/(seg[1, 0] - seg[0, 0])
    else:
        return (seg[1, 1] - seg[0, 1])/(seg[1, 0] + 0.5 - seg[0, 0])


def external_shape(curve_points, thickness):
    '''
        Reconstructs shape from skeleton curve by computing perpendicular lines of length thickness/2 for each point
        from : https://stackoverflow.com/a/57065334
    '''
    Cs = np.zeros(np.shape(curve_points)) # upper boundary
    Ds = np.zeros(np.shape(curve_points)) # lower boundary
    slopes = [coefficient_dir(np.array([curve_points[i+1], curve_points[i]]).reshape(-1,2)) for i in range(len(curve_points) - 1)]
    idx = [i for i in range(len(slopes)-1) if np.abs(slopes[i] - slopes[i+1]) > 25]

    for i in range(len(curve_points)-1):
        dy = math.sqrt((thickness**2)/((slopes[i]**2)+1))
        dx = -slopes[i]*dy

        if i > idx[0] and i <= idx[-1]:
            Cs[i, 0] = curve_points[i, 0] - dx
            Cs[i, 1] = curve_points[i, 1] - dy
            Ds[i, 0] = curve_points[i, 0] + dx
            Ds[i, 1] = curve_points[i, 1] + dy
        else:
            Cs[i, 0] = curve_points[i, 0] + dx
            Cs[i, 1] = curve_points[i, 1] + dy
            Ds[i, 0] = curve_points[i, 0] - dx
            Ds[i, 1] = curve_points[i, 1] - dy

    dy = math.sqrt((thickness**2)/((slopes[-1]**2)+1))
    dx = -slopes[-1]*dy    
    Cs[-1, 0] = curve_points[-1, 0] + dx
    Cs[-1, 1] = curve_points[-1, 1] + dy
    Ds[-1, 0] = curve_points[-1, 0] - dx
    Ds[-1, 1] = curve_points[-1, 1] - dy

    return np.array(Cs).reshape(-1, 2), np.array(Ds).reshape(-1, 2), idx


def get_shape_length(curve_points):
    n_points = len(curve_points)    
    total_length = 0
    arcs_lengths = list(range(n_points))
    arcs_lengths[0] = total_length
    
    for i in range(n_points - 1):
        total_length += math.sqrt((curve_points[i, 0] - curve_points[i+1, 0])**2 + (curve_points[i, 1] - curve_points[i+1, 1])**2)
        arcs_lengths[i+1] = total_length

    return total_length, arcs_lengths


def get_coords(x, dy_soma, thick, original_curve, paced_curve, idx, control_points):
    # déterminer les coordonées du point sur le squelette
    if x > 1:
        x = x*0.01
    p = mapping_bezier(x, original_curve, control_points) # coordonnées cartésiennes de p sur le squelette

    # décaler p de dy_soma
    new_p = np.array([[0., 0.]])
    x_high = math.ceil(x*100)
    p_low = np.array([p]).reshape(2,)
    p_high = paced_curve[x_high]
    slope = coefficient_dir(np.array([p_low, p_high]).reshape(-1, 2))
    dy = math.sqrt(((thick/2 - thick*dy_soma)**2)/((slope**2)+1))
    dx = -slope*dy
    if dy_soma == 0.5:
        new_p = p

    elif x*100 > idx[0] and x*100 <= idx[-1]:
        if dy_soma < 0.5:
            new_p[0, 0] = p[0, 0] + dx
            new_p[0, 1] = p[0, 1] + dy
        else:
            new_p[0, 0] = p[0, 0] - dx
            new_p[0, 1] = p[0, 1] - dy
    elif x*100 >=100:
        slope = coefficient_dir(np.array([paced_curve[99], p_low]).reshape(-1, 2))
        dy = math.sqrt(((thick/2 - thick*dy_soma)**2)/((slope**2)+1))
        dx = -slope*dy
        if dy_soma < 0.5:
            new_p[0, 0] = p[0, 0] - dx
            new_p[0, 1] = p[0, 1] - dy
        else:
            new_p[0, 0] = p[0, 0] + dx
            new_p[0, 1] = p[0, 1] + dy
    else:
        if dy_soma < 0.5:
            new_p[0, 0] = p[0, 0] - dx
            new_p[0, 1] = p[0, 1] - dy
        else:
            new_p[0, 0] = p[0, 0] + dx
            new_p[0, 1] = p[0, 1] + dy

    p_perp = np.array([p[0, 0], 0])
    theta = get_orientation(p[0], p_perp, new_p[0])

    if (dy_soma < 0.5 and p_perp[0] > new_p[0,0]) or (dy_soma > 0.5 and p_perp[0] < new_p[0,0]):
        theta = -theta

    if x*100 > idx[0] and x*100 <= idx[-1]:
        theta = math.pi - theta
        
    return new_p, theta


def get_orientation(p1, p2, p3):
    return math.acos(np.abs(np.dot(p1 - p2, p1 - p3))/(np.linalg.norm(p1 - p2) * np.linalg.norm(p1 - p3)))


def get_flattened_coordinates(x, y, x_total, y_total):
    x_flat = x * x_total
    y_flat = y * y_total

    return x_flat, y_flat


def pyr_placement(n_pyrs, region_min, region_max, thickness, original_curve, paced_curve, idx, control_points, constrain_y=False):
    x_pyrs = np.random.uniform(region_min*0.01, region_max*0.01, size=(n_pyrs,1))
    # sort ascending x 
    x_pyrs = np.sort(x_pyrs, axis=0)
    
    if constrain_y:
        # y_pyrs = np.random.uniform(0.25, 0.6, size=(n_pyrs, 1))
        y_pyrs = np.random.uniform(0.2, 0.3, size=(n_pyrs, 1))
    else:
        y_pyrs = np.random.uniform(0.01, 1, size=(n_pyrs, 1))
    # translate to cartesian coordinates
    pyr_coords = []
    for i in range(n_pyrs):
        coords, deg = get_coords(x_pyrs[i], y_pyrs[i], thickness, original_curve, paced_curve, idx, control_points)
        pyr_coords.extend(np.array([coords[0,0], coords[0,1], deg]))


    return np.array(pyr_coords).reshape(-1, 3), np.hstack((x_pyrs, y_pyrs)) # cartesian coordinates, intrinsic coordinates


def inh_placement(n_inh, region_min, region_max, cell_type, thickness, original_curve, paced_curve, idx, control_points):
    inh_dist_x = (region_max - region_min)/(n_inh+1)
    if cell_type == 'olm' or cell_type == 'hipp':
        # y_inhs = np.random.uniform(0, 0.5, size=(n_inh, 1))
        y_inhs = np.random.uniform(0.01, 0.2, size=(n_inh, 1))
    if cell_type == 'basket':
        # y_inhs = np.random.uniform(0.3, 0.7, size=(n_inh, 1))
        y_inhs = np.random.uniform(0.2, 0.3, size=(n_inh, 1))

    x_inhs = []
    for i in range(n_inh):
        x_inhs.append(region_min + (i+1)*inh_dist_x)

    # translate to cartesian coordinates
    inhs_coords = []
    for i in range(n_inh):
        coords, deg = get_coords(x_inhs[i], y_inhs[i], thickness, original_curve, paced_curve, idx, control_points)
        inhs_coords.extend(np.array([coords[0,0], coords[0,1], deg]))

    return np.array(inhs_coords).reshape(-1, 3), np.hstack((np.array(x_inhs).reshape(-1,1)*0.01, y_inhs)) # cartesian coordinates, intrinsic coordinates


def build_axonal_extension(paced_curve, thickness, threshold):
    Ds = np.zeros(np.shape(paced_curve)) # lower boundary
    slopes = [coefficient_dir(np.array([paced_curve[i+1], paced_curve[i]]).reshape(-1,2)) for i in range(len(paced_curve) - 1)]
    idx = [i for i in range(len(slopes)-1) if np.abs(slopes[i] - slopes[i+1]) > threshold]

    for i in range(len(paced_curve)-1):
        dy = math.sqrt((thickness**2)/((slopes[i]**2)+1))
        dx = -slopes[i]*dy

        if i > idx[0] and i <= idx[-1]:
            Ds[i, 0] = paced_curve[i, 0] + dx
            Ds[i, 1] = paced_curve[i, 1] + dy
        else:
            Ds[i, 0] = paced_curve[i, 0] - dx
            Ds[i, 1] = paced_curve[i, 1] - dy

    dy = math.sqrt((thickness**2)/((slopes[-1]**2)+1))
    dx = -slopes[-1]*dy    
    Ds[-1, 0] = paced_curve[-1, 0] - dx
    Ds[-1, 1] = paced_curve[-1, 1] - dy

    return np.array(Ds).reshape(-1, 2)

def build_axonal_extension_5(paced_curve, thickness, n_points=None):
    """
    Crée une courbe parallèle lisse à paced_curve, décalée de 'thickness',
    avec n_points points (par défaut = len(paced_curve)).
    paced_curve[i] correspond à offset_curve[i].
    """
    if n_points is None:
        n_points = len(paced_curve)

    # Étape 1 : Créer une spline paramétrique
    tck, u = splprep(paced_curve.T, s=0)  # interpolation exacte

    # Étape 2 : Créer les points t uniformes
    t_uniform = np.linspace(0, 1, n_points)

    # Étape 3 : Évaluer la courbe originale aux t_uniform
    curve_spline = splev(t_uniform, tck)  # (x,y)

    # Étape 4 : Calculer les dérivées (tangentes)
    dx, dy = splev(t_uniform, tck, der=1)

    # Étape 5 : Normaliser les tangentes
    speed = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / speed
    dy_norm = dy / speed

    # Étape 6 : Vecteurs normaux (tournés de 90°)
    nx = -dy_norm
    ny = dx_norm

    # Étape 7 : Décaler chaque point dans la direction normale
    offset_x = curve_spline[0] - nx * thickness
    offset_y = curve_spline[1] - ny * thickness

    # Étape 8 : Ré-interpoler la courbe décalée pour garantir la lissité
    # tck_offset, u_offset = splprep([offset_x, offset_y], s=0)
    # offset_curve = splev(t_uniform, tck_offset)

    return np.array([offset_x, offset_y]).T  # (n_points, 2)


def find_node_coordinates(axon_trajectory):
    """
        Determine the coordinates of axon nodes for a given trajectory.
        Node spacing is determined by the fiber diameter
        
        Inputs:
            fiberD = µm, fiber diameter
            axon_trajectory = n x 3 array containing the line segments describing the axon trajectory in µm

        Outputs:
            array of x,y,z coordinates of nodes with the proper spacing along the provided trajectory
        
    """
    dx = 117                   # node spacing
    ii, jj = 0, 0                         # counters for stepping through points of axon trajectories
    xx, yy, zz = list(), list(), list()     # "interpolated" node locations for NEURON
    P0 = axon_trajectory[ii,:3]              # initialize point, P0
    P1 = axon_trajectory[ii+1,:3]            # initialize point, P1

    # define initial point of axon, i.e. first node
    xx.append(P0[0])
    yy.append(P0[1])
    zz.append(P0[2]) 

    while True:
        P = [xx[jj],yy[jj],zz[jj]] # last node
        PP1 = np.sqrt(sum((P1 - P)**2)) # distance of new segment point to last node
        if (PP1 >= dx): # check if it's longer than internode distance
            P0P1 = np.sqrt(sum((P1 - P0)**2)) # distance of before last segment to last segment
            tt = (P0P1 - PP1 + dx)/P0P1
            xx.append((1 - tt) * P0[0] + tt * P1[0])
            yy.append((1 - tt) * P0[1] + tt * P1[1])
            zz.append((1 - tt) * P0[2] + tt * P1[2])
            jj += 1
        else:
            ii += 1                               # update P0 and P1
            if (ii == (np.shape(axon_trajectory)[0] - 1)):
                break
            P0 = P1
            P1 = axon_trajectory[ii+1,:3]
            # P0P1 = np.sqrt(sum((P1 - P0)**2))
            # tt = (dx - PP1)/P0P1
            # xx.append((1 - tt) * P0[0] + tt * P1[0])
            # yy.append((1 - tt) * P0[1] + tt * P1[1])
            # zz.append((1 - tt) * P0[2] + tt * P1[2])
            # jj += 1

    return np.array([xx, yy, zz]).T


def get_y_axonal_extension(x, section, curve_points):
    return math.sqrt((curve_points[x, 0] - section.x3d(1))**2 + (curve_points[x, 1] - section.y3d(1))**2)

def get_y_axonal_extension_2(x, x_sec, y_sec, curve_points):
    return math.sqrt((curve_points[x, 0] - x_sec)**2 + (curve_points[x, 1] - y_sec)**2)


def get_coords_intr_2(yi, xc, yc, shape_points):
    """
    Parameters :
        - yi : float, coordonnée y intrinsèque
        - xc : float, coordonnée x cartésienne
        - yc : float, coordonnée y cartésienne
        - shape_points : np.array of len (N, 2), coordonnées des points constituants le S
    Return :
        - xi : float, coordonnée x intrinsèque

    On cherche l'indice i pour lequel dist((xc, yc), shape_points[i]) est minime
    """
    # Trouver le candidat minimisant la distance à (xc, yc)
    candidates = build_axonal_extension_5(shape_points,  yi)
    min_dist = float('inf')
    best_j = candidates[0]
    
    for j in range(len(candidates)):
        dx = xc - candidates[j, 0]
        dy = yc - candidates[j, 1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < min_dist:
            min_dist = dist
            best_j = j
    
    # interpolate
    if min_dist == 0:
        # Retourner l'indice normalisé (0 à 1)
        return best_j / (len(shape_points)-1)
    else:
        dx0 = xc - candidates[best_j-1, 0]
        dy0 = yc - candidates[best_j-1, 1]

        dx1 = xc - candidates[best_j+1, 0]
        dy1 = yc - candidates[best_j+1, 1]

        dist0 = math.sqrt(dx0*dx0 + dy0*dy0)
        dist1 = math.sqrt(dx1*dx1 + dy1*dy1)

        if dist0 == min(dist0, dist1):
            dx = candidates[best_j, 0] - candidates[best_j-1, 0]
            dy = candidates[best_j, 1] - candidates[best_j-1, 1]

            dist = math.sqrt(dx*dx + dy*dy)
            return (best_j + (dist/min_dist))/(len(shape_points)-1)
        else:
            dx = candidates[best_j, 0] - candidates[best_j+1, 0]
            dy = candidates[best_j, 1] - candidates[best_j+1, 1]

            dist = math.sqrt(dx*dx + dy*dy)
            return (best_j - (dist/min_dist))/(len(shape_points)-1)
