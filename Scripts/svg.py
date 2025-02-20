import re
# import bezier
import numpy as np
from lxml import etree
from matplotlib.path import Path


def get(filename, name):
    """
    Read a given element from an SVG file
    """
    root = etree.parse(filename).getroot()
    return root.xpath("//*[@id='%s']" % name)[0].get("d")


def path(filename, name):
    """
    Read and convert an SVG path command into a matplotlib path representation
    """
    verts, codes, cubic_points, nodes = convert(get(filename, name))
    return Path(verts,codes), cubic_points, nodes


def convert(path):
    """
    Parse and convert an SVG path command into a matplotlib path representation

    Parameters
    ----------
    path : string
        A valid SVG path command
    """

    # First we separate tokens inside the path
    tokens = []
    COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
    COMMAND_RE = re.compile("([MmZzLlHhVvCcSsQqTtAa])")
    FLOAT_RE = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    for x in COMMAND_RE.split(path):
        if x in COMMANDS:
            tokens.append(x)
        for token in FLOAT_RE.findall(x):
            tokens.append(float(token))

    # Then we process and convert commands
    # (Note that not all commands have been implemented)
    commands = { 'M': [Path.MOVETO, 2], 'm': [Path.MOVETO, 2],
                 'L': [Path.LINETO, 2], 'l': [Path.LINETO, 2], 
                 # 'V': [Path.LINETO, 1], 'v': [Path.LINETO, 1],
                 # 'H': [Path.LINETO, 1], 'h': [Path.LINETO, 1], 
                 'C': [Path.CURVE4, 6], 'c': [Path.CURVE4, 6],
                 # 'S': [Path.CURVE4, 4], 's': [Path.CURVE4, 4],
                 'Q': [Path.CURVE3, 4], 'q': [Path.CURVE3, 4],
                 # 'T': [Path.CURVE3, 2], 't': [Path.CURVE3, 2],
                 # 'A': [None,        7], 'a': [None,        7],
                 'Z': [Path.CLOSEPOLY, 0], 'z': [Path.CLOSEPOLY, 0] }
    index = 0
    codes, verts = [], []
    cubic_bezier_points = [] 
    last_vertice = np.array([[0, 0]])
    nodes = []
    while index < len(tokens):
        # Token is a command
        # (else, we re-use last command because
        #   SVG allows to omit command when it is the same as the last command)
        if isinstance(tokens[index], str):
            last_command = tokens[index]
            code, n = commands[last_command]
            index += 1
        if n > 0:
            vertices = np.array(tokens[index:index+n]).reshape(n//2,2)
            nodes.append(vertices[-1])
            if last_command.islower():
                vertices += last_vertice
            if last_command.lower() == 'c':
                bezier_points = np.concatenate((np.reshape(last_vertice, (-1, 2)), vertices), axis=0)
                cubic_bezier_points.extend(bezier_points.tolist())
            last_vertice = vertices[-1]
            codes.extend([code,]*len(vertices))
            verts.extend(vertices.tolist())
            index += n
        else:
            codes.append(code)
            verts.append(last_vertice.tolist())

        # A 'M/m' follows by several vertices means implicit 'L/l' for
        # subsequent vertices
        if last_command == 'm':
            last_command, code = 'l', Path.LINETO
        elif last_command == 'M':
            last_command, code = 'L', Path.LINETO

    # verts = np.array(verts)
    # verts[:,1] = -verts[:,1]
    return verts, codes, cubic_bezier_points, np.array(nodes).reshape(-1,2)


