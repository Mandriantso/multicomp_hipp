import sys
import os
from neuron import h, hclass
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)
h.nrn_load_dll(os.path.join(dirname, '../Mods/nrnmech.dll'))
h.load_file("stdgui.hoc")

h.load_file(os.path.join(dirname, '../Cells/Tomko_ca1_pyr.hoc'))
h.load_file(os.path.join(dirname,'../Cells/class_pvbasketcell.hoc'))
h.load_file(os.path.join(dirname,'../Cells/class_olmcell.hoc'))
# h.load_file(os.path.join(dirname,'../GC.hoc'))
# h.load_file(os.path.join(dirname,'../HC.hoc'))

# create subclass for pyramidal cells to prepare for xtra
class PyrCell(hclass(h.CA1_PC_Tomko)):
	def grindway(self):
		for section in self.all:
			if h.ismembrane('xtra', sec=section):
				
				nn = section.n3d()
				xx = h.Vector(nn)
				yy = h.Vector(nn)
				zz = h.Vector(nn)
				length = h.Vector(nn)
				
				for i in range(nn):
					xx.x[i] = section.x3d(i)
					yy.x[i] = section.y3d(i)
					zz.x[i] = section.z3d(i)
					length.x[i] = section.arc3d(i)
					
				length.div(length.x[nn-1])
				
				r = h.Vector(section.nseg + 2)
				r.indgen(1/section.nseg)
				r.sub(1/(2 * section.nseg))
				r.x[0] = 0
				r.x[section.nseg + 1] = 1
				
				xint = h.Vector(section.nseg+2)
				yint = h.Vector(section.nseg+2)
				zint = h.Vector(section.nseg+2)
				xint.interpolate(r, length, xx)
				yint.interpolate(r, length, yy)
				zint.interpolate(r, length, zz)
				
				for i in range(1, section.nseg+1):
					xr = r.x[i]
					section(xr).x_xtra = xint.x[i]
					section(xr).y_xtra = yint.x[i]
					section(xr).z_xtra = zint.x[i]


# create subclass for basket cells to prepare for xtra
class BasketCell(hclass(h.pvbasketcell)):
	def grindway(self):
		for section in self.all:
			if h.ismembrane('xtra', sec=section):
				
				nn = section.n3d()
				xx = h.Vector(nn)
				yy = h.Vector(nn)
				zz = h.Vector(nn)
				length = h.Vector(nn)
				
				for i in range(nn):
					xx.x[i] = section.x3d(i)
					yy.x[i] = section.y3d(i)
					zz.x[i] = section.z3d(i)
					length.x[i] = section.arc3d(i)
					
				length.div(length.x[nn-1])
				
				r = h.Vector(section.nseg + 2)
				r.indgen(1/section.nseg)
				r.sub(1/(2 * section.nseg))
				r.x[0] = 0
				r.x[section.nseg + 1] = 1
				
				xint = h.Vector(section.nseg+2)
				yint = h.Vector(section.nseg+2)
				zint = h.Vector(section.nseg+2)
				xint.interpolate(r, length, xx)
				yint.interpolate(r, length, yy)
				zint.interpolate(r, length, zz)
				
				for i in range(1, section.nseg+1):
					xr = r.x[i]
					section(xr).x_xtra = xint.x[i]
					section(xr).y_xtra = yint.x[i]
					section(xr).z_xtra = zint.x[i]


class OLMCell(hclass(h.olmcell)):
	def grindway(self):
		for section in self.all:
			if h.ismembrane('xtra', sec=section):
				
				nn = section.n3d()
				xx = h.Vector(nn)
				yy = h.Vector(nn)
				zz = h.Vector(nn)
				length = h.Vector(nn)
				
				for i in range(nn):
					xx.x[i] = section.x3d(i)
					yy.x[i] = section.y3d(i)
					zz.x[i] = section.z3d(i)
					length.x[i] = section.arc3d(i)
					
				length.div(length.x[nn-1])
				
				r = h.Vector(section.nseg + 2)
				r.indgen(1/section.nseg)
				r.sub(1/(2 * section.nseg))
				r.x[0] = 0
				r.x[section.nseg + 1] = 1
				
				xint = h.Vector(section.nseg+2)
				yint = h.Vector(section.nseg+2)
				zint = h.Vector(section.nseg+2)
				xint.interpolate(r, length, xx)
				yint.interpolate(r, length, yy)
				zint.interpolate(r, length, zz)
				
				for i in range(1, section.nseg+1):
					xr = r.x[i]
					section(xr).x_xtra = xint.x[i]
					section(xr).y_xtra = yint.x[i]
					section(xr).z_xtra = zint.x[i]


# class GranuleCell(hclass(h.GranuleCell)):
# 	def grindway(self):
# 		for section in self.all:
# 			if h.ismembrane('xtra', sec=section):
				
# 				nn = section.n3d()
# 				xx = h.Vector(nn)
# 				yy = h.Vector(nn)
# 				zz = h.Vector(nn)
# 				length = h.Vector(nn)
				
# 				for i in range(nn):
# 					xx.x[i] = section.x3d(i)
# 					yy.x[i] = section.y3d(i)
# 					zz.x[i] = section.z3d(i)
# 					length.x[i] = section.arc3d(i)
					
# 				length.div(length.x[nn-1])
				
# 				r = h.Vector(section.nseg + 2)
# 				r.indgen(1/section.nseg)
# 				r.sub(1/(2 * section.nseg))
# 				r.x[0] = 0
# 				r.x[section.nseg + 1] = 1
				
# 				xint = h.Vector(section.nseg+2)
# 				yint = h.Vector(section.nseg+2)
# 				zint = h.Vector(section.nseg+2)
# 				xint.interpolate(r, length, xx)
# 				yint.interpolate(r, length, yy)
# 				zint.interpolate(r, length, zz)
				
# 				for i in range(1, section.nseg+1):
# 					xr = r.x[i]
# 					section(xr).x_xtra = xint.x[i]
# 					section(xr).y_xtra = yint.x[i]
# 					section(xr).z_xtra = zint.x[i]


# class HIPPCell(hclass(h.HIPPCell)):
# 	def grindway(self):
# 		for section in self.all:
# 			if h.ismembrane('xtra', sec=section):
				
# 				nn = section.n3d()
# 				xx = h.Vector(nn)
# 				yy = h.Vector(nn)
# 				zz = h.Vector(nn)
# 				length = h.Vector(nn)
				
# 				for i in range(nn):
# 					xx.x[i] = section.x3d(i)
# 					yy.x[i] = section.y3d(i)
# 					zz.x[i] = section.z3d(i)
# 					length.x[i] = section.arc3d(i)
					
# 				length.div(length.x[nn-1])
				
# 				r = h.Vector(section.nseg + 2)
# 				r.indgen(1/section.nseg)
# 				r.sub(1/(2 * section.nseg))
# 				r.x[0] = 0
# 				r.x[section.nseg + 1] = 1
				
# 				xint = h.Vector(section.nseg+2)
# 				yint = h.Vector(section.nseg+2)
# 				zint = h.Vector(section.nseg+2)
# 				xint.interpolate(r, length, xx)
# 				yint.interpolate(r, length, yy)
# 				zint.interpolate(r, length, zz)
				
# 				for i in range(1, section.nseg+1):
# 					xr = r.x[i]
# 					section(xr).x_xtra = xint.x[i]
# 					section(xr).y_xtra = yint.x[i]
# 					section(xr).z_xtra = zint.x[i]


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.
	from : https://stackoverflow.com/a/35501485

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def plot_2D(h, ax, section, color, real_width=False, d=0):
	'''
		Parameters:
			- d : additional diameter -> increase diameter of section by value of d. Can be used for hovering
	'''
	n3d = int(h.n3d(sec=section))
	xyd = []
	for i in range(0,n3d):
		xyd.append([h.x3d(i,sec=section),h.y3d(i,sec=section),h.diam3d(i,sec=section)+d])
	xyd = np.array(xyd)
	if real_width:
		ax.plot(xyd[:,0], xyd[:,1], color=color, linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
	else:
		ax.plot(xyd[:,0], xyd[:,1], color=color, linewidth=int(xyd[0,2]))


def plot_2D_custom(h, ax, section, colors, d=0):
	'''
		Parameters:
			- d : additional diameter -> increase diameter of section by value of d. Can be used for hovering
	'''
	n3d = int(h.n3d(sec=section))
	xyd = []
	for i in range(0,n3d):
		xyd.append([h.x3d(i,sec=section),h.y3d(i,sec=section),h.diam3d(i,sec=section)+d])
	xyd = np.array(xyd)
	if len(colors) > 1 and len(colors) == len(xyd)-1:
		#assert len(colors) == len(xyd)-1, (f"colors length must be 1 or {len(xyd)-1} not {len(colors)}")
		for i in range(len(xyd) - 2):
			ax.plot(xyd[i:i+2,0], xyd[i:i+2,1], color=colors[i], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
		ax.plot(xyd[-2:,0], xyd[-2:,1], color=colors[-1], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
	else:
		ax.plot(xyd[:,0], xyd[:,1], color=colors[0], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))


def plot_sectionlist_2D(h, ax, sections, color, real_width=False, exception=[], d=0):
	for sec in sections:
		if sec not in exception:
			plot_2D(h, ax, sec, color, real_width, d)
