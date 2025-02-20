import numpy as np
import math

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
        y_pyrs = np.random.uniform(0.25, 0.6, size=(n_pyrs, 1))
    else:
        y_pyrs = np.random.uniform(0, 1, size=(n_pyrs, 1))
    # translate to cartesian coordinates
    pyr_coords = []
    for i in range(n_pyrs):
        coords, deg = get_coords(x_pyrs[i], y_pyrs[i], thickness, original_curve, paced_curve, idx, control_points)
        pyr_coords.extend(np.array([coords[0,0], coords[0,1], deg]))


    return np.array(pyr_coords).reshape(-1, 3), np.hstack((x_pyrs, y_pyrs)) # cartesian coordinates, intrinsic coordinates


def inh_placement(n_inh, region_min, region_max, cell_type, thickness, original_curve, paced_curve, idx, control_points):
    inh_dist_x = (region_max - region_min)/(n_inh+1)
    if cell_type == 'olm' or cell_type == 'hipp':
        y_inhs = np.random.uniform(0, 0.5, size=(n_inh, 1))
    if cell_type == 'basket':
        y_inhs = np.random.uniform(0.3, 0.7, size=(n_inh, 1))

    x_inhs = []
    for i in range(n_inh):
        x_inhs.append(region_min + (i+1)*inh_dist_x)

    # translate to cartesian coordinates
    inhs_coords = []
    for i in range(n_inh):
        coords, deg = get_coords(x_inhs[i], y_inhs[i], thickness, original_curve, paced_curve, idx, control_points)
        inhs_coords.extend(np.array([coords[0,0], coords[0,1], deg]))

    return np.array(inhs_coords).reshape(-1, 3), np.hstack((np.array(x_inhs).reshape(-1,1)*0.01, y_inhs)) # cartesian coordinates, intrinsic coordinates


def build_axonal_extension(paced_curve, thickness):
    Ds = np.zeros(np.shape(paced_curve)) # lower boundary
    slopes = [coefficient_dir(np.array([paced_curve[i+1], paced_curve[i]]).reshape(-1,2)) for i in range(len(paced_curve) - 1)]
    idx = [i for i in range(len(slopes)-1) if np.abs(slopes[i] - slopes[i+1]) > 25]

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


def get_y_axonal_extension(x, section, curve_points):
    return math.sqrt((curve_points[x, 0] - section.x3d(1))**2 + (curve_points[x, 1] - section.y3d(1))**2)



