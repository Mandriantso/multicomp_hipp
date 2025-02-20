from neuron import h
from neuron.units import ms, mV
import numpy as np

h.nrn_load_dll('Mods/nrnmech.dll')
h.load_file("stdgui.hoc")

class Cell():
    def __init__(self, gid: int,
                 x: float=0.,
                 y: float=0.,
                 z: float=0.,
                 theta: float=0.,) -> None:
        self._gid = gid

        # morphology
        self._setup_morphology() # create and connect sections + basic_shape()
        self._setup_dimensions() # sections dimensions
        self._setup_subsections() # section lists
        self._set_nseg() # number of seg per section
        h.define_shape()
        self._insert_channels()
        self._insert_mechanisms()
        self._setup_biophysics()

        # position
        self.x = self.y = self.z = 0
        self._set_position(x, y, z)
        self._rotate_z(theta)

        # recording vectors
        self.spike_times = h.Vector()

        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self._spike_detector.threshold = 0
        self._spike_detector.record(self.spike_times)

        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)
        self.soma_i = h.Vector().record(self.soma(0.5)._ref_i_membrane)

        # presynaptic and postsynaptic neurons gid list
        self._presyn_list = []
        self._postsyn_list = []

        self._ncs = []
        self._inputs_list = []

        # synapses list
        self._syn_list = h.List() # list of neuron objects

        # create synapses
        self._create_synapses()

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)
    
    def _set_nseg(self):
        for sec in self.all:
            sec.nseg = 1 + 2*int(sec.L/40)

    def _update_shape(self):
        h.define_shape()

    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
        self.x, self.y, self.z = x, y, z
        h.define_shape()

    def _rotate_x(self, theta):
        # translation to bring soma to (0, 0, 0)
        x_op = 0 - self.soma.x3d(0) - self.soma.diam/2
        y_op = 0 - self.soma.y3d(0) - self.soma.L/2
        z_op = 0 - self.soma.z3d(0)

        self._set_position(self.x + x_op, self.y + y_op, self.z + z_op)

        for sec in self.all:
            for i in range(sec.n3d()):
                y = sec.y3d(i)
                z = sec.z3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                yprime = y * c - z * s
                zprime = y * s + z * c
                sec.pt3dchange(i, sec.x3d(i), yprime, zprime, sec.diam3d(i))

        # translation to go back to original soma position
        self._set_position(self.x - x_op, self.y - y_op, self.z - z_op)
    
    def _rotate_y(self, theta):
        # translation to bring soma to (0, 0, 0)
        x_op = 0 - self.soma.x3d(0) - self.soma.diam/2
        y_op = 0 - self.soma.y3d(0) - self.soma.L/2
        z_op = 0 - self.soma.z3d(0)

        self._set_position(self.x + x_op, self.y + y_op, self.z + z_op)

        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                z = sec.z3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c + z * s
                zprime = -x * s + z * c
                sec.pt3dchange(i, xprime, sec.y3d(i), zprime, sec.diam3d(i))

        # translation to go back to original soma position
        self._set_position(self.x - x_op, self.y - y_op, self.z - z_op)


    def _rotate_z(self, theta):
        # translation to bring soma to (0, 0, 0)
        x_op = 0 - self.soma.x3d(0) - self.soma.diam/2
        y_op = 0 - self.soma.y3d(0) - self.soma.L/2
        z_op = 0 - self.soma.z3d(0)

        self._set_position(self.x + x_op, self.y + y_op, self.z + z_op)

        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))
        
        # translation to go back to original soma position
        self._set_position(self.x - x_op, self.y - y_op, self.z - z_op)

    def _insert_mechanisms(self):
        for sec in self.all:
            sec.insert("extracellular")
            # sec.insert("xtra") # can't use with intracellular stimulation

    def _connect2target(self, sourcesec, target):
        pass

    def _grindway(self):
        for sec in self.all:
            if h.ismembrane('xtra', sec=sec):
                nn = sec.n3d()
                xx = h.Vector(nn)
                yy = h.Vector(nn)
                zz = h.Vector(nn)
                length = h.Vector(nn)

                for i in range(nn):
                    xx.x[i] = sec.x3d(i)
                    yy.x[i] = sec.y3d(i)
                    zz.x[i] = sec.z3d(i)
                    length.x[i] = sec.arc3d(i)
                    
                length.div(length.x[nn-1])
                
                r = h.Vector(sec.nseg + 2)
                r.indgen(1/sec.nseg)
                r.sub(1/(2 * sec.nseg))
                r.x[0] = 0
                r.x[sec.nseg + 1] = 1
                
                xint = h.Vector(sec.nseg+2)
                yint = h.Vector(sec.nseg+2)
                zint = h.Vector(sec.nseg+2)
                xint.interpolate(r, length, xx)
                yint.interpolate(r, length, yy)
                zint.interpolate(r, length, zz)
                
                for i in range(1, sec.nseg+1):
                    xr = r.x[i]
                    sec(xr).x_xtra = xint.x[i]
                    sec(xr).y_xtra = yint.x[i]
                    sec(xr).z_xtra = zint.x[i]


class PyramidalCell(Cell):
    name = "PyramidalCell"
    def __init__(self, gid_soma: int, gid_axon: int, x: float = 0, y: float = 0, z: float = 0, theta: float = 0) -> None:
        super().__init__(gid_soma, x, y, z, theta)
        self.syn_dist = 500
        self._gid_axon = gid_axon

        # recording vectors axon
        self.spike_times_axon = h.Vector()

        self._spike_detector_axon = h.NetCon(self.axon(1)._ref_v, None, sec=self.axon)
        self._spike_detector_axon.threshold = 0
        self._spike_detector_axon.record(self.spike_times_axon)

        self.axon_v = h.Vector().record(self.axon(1)._ref_v)

    def _setup_morphology(self):
        # create sections
        self.soma = h.Section(name="soma", cell=self)

        self.bdend1 = h.Section(name="bdend1", cell=self)
        self.bdend2 = h.Section(name="bdend2", cell=self)
        self.bdend3 = h.Section(name="bdend3", cell=self)
        self.bdend4 = h.Section(name="bdend4", cell=self)

        self.adend1 = h.Section(name="adend1", cell=self)
        self.adend2 = h.Section(name="adend2", cell=self)
        self.adend3 = h.Section(name="adend3", cell=self)
        self.adend4 = h.Section(name="adend4", cell=self)
        self.adend5 = h.Section(name="adend5", cell=self)
        self.adend6 = h.Section(name="adend6", cell=self)
        self.adend7 = h.Section(name="adend7", cell=self)
        self.adend8 = h.Section(name="adend8", cell=self)
        self.adend9 = h.Section(name="adend9", cell=self)

        self.axon = h.Section(name="axon", cell=self)

        # connect sections
        self.adend1.connect(self.soma(1), 0)
        self.adend2.connect(self.adend1(1), 0)
        self.adend3.connect(self.adend2(1), 0)

        self.adend4.connect(self.adend3(1), 0)
        self.adend5.connect(self.adend4(1), 0)
        self.adend6.connect(self.adend5(1), 0)

        self.adend7.connect(self.adend3(1), 0)
        self.adend8.connect(self.adend7(1), 0)
        self.adend9.connect(self.adend8(1), 0)

        self.bdend1.connect(self.soma(0), 0)
        self.bdend2.connect(self.bdend1(1), 0)

        self.bdend3.connect(self.soma(0), 0)
        self.bdend4.connect(self.bdend3(1), 0)

        self.axon.connect(self.soma(0), 0)

        # set basic shape
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0., 0., 0., 10., sec=self.soma) 
        h.pt3dadd(0., 5., 0., 10., sec=self.soma) 
        h.pt3dadd(0., 10., 0., 10., sec=self.soma) 

        h.pt3dclear(sec=self.adend1)
        h.pt3dadd(0., 10., 0., 4., sec=self.adend1) 
        h.pt3dadd(0., 60., 0., 4., sec=self.adend1) 
        h.pt3dadd(0., 110., 0., 14., sec=self.adend1) 

        h.pt3dclear(sec=self.adend2)
        h.pt3dadd(0., 110., 0., 3., sec=self.adend2) 
        h.pt3dadd(0., 160., 0., 3., sec=self.adend2) 
        h.pt3dadd(0., 210., 0., 3., sec=self.adend2) 

        h.pt3dclear(sec=self.adend3)
        h.pt3dadd(0., 210., 0., 2., sec=self.adend3) 
        h.pt3dadd(0., 310., 0., 2., sec=self.adend3) 
        h.pt3dadd(0., 410., 0., 2., sec=self.adend3)

        h.pt3dclear(sec=self.adend4)
        h.pt3dadd(0., 410., 0., 2., sec=self.adend4) 
        h.pt3dadd(35.5, 445.5, 0., 2., sec=self.adend4) 
        h.pt3dadd(71., 481., 0., 2., sec=self.adend4)

        h.pt3dclear(sec=self.adend5)
        h.pt3dadd(71., 481., 0., 1.5, sec=self.adend5) 
        h.pt3dadd(106.5, 516.5, 0., 1.5, sec=self.adend5) 
        h.pt3dadd(142., 552., 0., 1.5, sec=self.adend5)

        h.pt3dclear(sec=self.adend6)
        h.pt3dadd(142., 552., 0., 1., sec=self.adend6) 
        h.pt3dadd(159.7, 569.7, 0., 1., sec=self.adend6) 
        h.pt3dadd(177.4, 587.4, 0., 1., sec=self.adend6)

        h.pt3dclear(sec=self.adend7)
        h.pt3dadd(0., 410., 0., 2., sec=self.adend7) 
        h.pt3dadd(-35.5, 445.5, 0., 2., sec=self.adend7) 
        h.pt3dadd(-71., 481., 0., 2., sec=self.adend7)

        h.pt3dclear(sec=self.adend8)
        h.pt3dadd(-71., 481., 0., 1.5, sec=self.adend8) 
        h.pt3dadd(-106.5, 516.5, 0., 1.5, sec=self.adend8) 
        h.pt3dadd(-142., 552., 0., 1.5, sec=self.adend8)

        h.pt3dclear(sec=self.adend9)
        h.pt3dadd(-142., 552., 0., 1., sec=self.adend9) 
        h.pt3dadd(-159.7, 569.7, 0., 1., sec=self.adend9) 
        h.pt3dadd(-177.4, 587.4, 0., 1., sec=self.adend9)

        h.pt3dclear(sec=self.bdend1)
        h.pt3dadd(0., 0., 0., 2., sec=self.bdend1) 
        h.pt3dadd(35.5, -35.5, 0., 2., sec=self.bdend1) 
        h.pt3dadd(71, -71, 0., 2., sec=self.bdend1)

        h.pt3dclear(sec=self.bdend2)
        h.pt3dadd(71., -71., 0., 1.5, sec=self.bdend2) 
        h.pt3dadd(142., -142., 0., 1.5, sec=self.bdend2) 
        h.pt3dadd(212.4, -212.4, 0., 1.5, sec=self.bdend2)

        h.pt3dclear(sec=self.bdend3)
        h.pt3dadd(0., 0., 0., 2., sec=self.bdend3) 
        h.pt3dadd(-35.5, -35.5, 0., 2., sec=self.bdend3) 
        h.pt3dadd(-71., -71., 0., 2., sec=self.bdend3)

        h.pt3dclear(sec=self.bdend4)
        h.pt3dadd(-71., -71., 0., 1.5, sec=self.bdend4) 
        h.pt3dadd(-142., -142., 0., 1.5, sec=self.bdend4) 
        h.pt3dadd(-212.4, -212.4, 0., 1.5, sec=self.bdend4)

        h.pt3dclear(sec=self.axon)
        h.pt3dadd(0., 0., 0., 1., sec=self.axon) 
        h.pt3dadd(0., -75., 0., 1., sec=self.axon) 
        h.pt3dadd(0., -150., 0., 1., sec=self.axon)

    def _setup_dimensions(self):
        self.soma.L = 20
        self.soma.diam = 10

        self.adend1.L = 100
        self.adend1.diam = 4

        self.adend2.L = 100
        self.adend2.diam = 3

        self.adend3.L = 200
        self.adend3.diam = 2

        self.adend4.L = 100.41
        self.adend4.diam = 2

        self.adend5.L = 100.41
        self.adend5.diam = 1.5

        self.adend6.L = 50.06
        self.adend6.diam = 1

        self.adend7.L = 100.41
        self.adend7.diam = 2

        self.adend8.L = 100.41
        self.adend8.diam = 1.5

        self.adend9.L = 50.06
        self.adend9.diam = 1

        self.bdend1.L = 100.41
        self.bdend1.diam = 2

        self.bdend2.L = 199.97
        self.bdend2.diam = 1.5

        self.bdend3.L = 100.41
        self.bdend3.diam = 2

        self.bdend4.L = 199.97
        self.bdend4.diam = 1.5

        self.axon.L = 150
        self.axon.diam = 1

    def _setup_subsections(self):
        self.all = h.SectionList()
        self.all.append(self.soma)
        self.all.append(self.bdend1)
        self.all.append(self.bdend2)
        self.all.append(self.bdend3)
        self.all.append(self.bdend4)
        self.all.append(self.adend1)
        self.all.append(self.adend2)
        self.all.append(self.adend3)
        self.all.append(self.adend4)
        self.all.append(self.adend5)
        self.all.append(self.adend6)
        self.all.append(self.adend7)
        self.all.append(self.adend8)
        self.all.append(self.adend9)
        self.all.append(self.axon)

        self.somatic = h.SectionList()
        self.somatic.append(self.soma)

        self.basal = h.SectionList()
        self.basal.append(self.bdend1)
        self.basal.append(self.bdend2)
        self.basal.append(self.bdend3)
        self.basal.append(self.bdend4)

        self.apical = h.SectionList()
        self.apical.append(self.adend1)
        self.apical.append(self.adend2)
        self.apical.append(self.adend3)
        self.apical.append(self.adend4)
        self.apical.append(self.adend5)
        self.apical.append(self.adend6)
        self.apical.append(self.adend7)
        self.apical.append(self.adend8)
        self.apical.append(self.adend9)

        self.axonal = h.SectionList()
        self.axonal.append(self.axon)

        self.rad_list = h.SectionList()
        self.rad_list.append(self.adend1)
        self.rad_list.append(self.adend2)
        self.rad_list.append(self.adend3)

        self.lm_list = h.SectionList()
        self.lm_list.append(self.adend4) # removed
        # self.lm_list.append(self.adend5)
        # self.lm_list.append(self.adend6)
        self.lm_list.append(self.adend7) # removed
        # self.lm_list.append(self.adend8)
        # self.lm_list.append(self.adend9)

        self.proximal = h.SectionList()
        self.proximal.append(self.bdend1) # removed
        self.proximal.append(self.bdend3) # removed
        self.proximal.append(self.adend1)

        self.dendrites = h.SectionList()
        self.dendrites.append(self.bdend1)
        self.dendrites.append(self.bdend2)
        self.dendrites.append(self.bdend3)
        self.dendrites.append(self.bdend4)
        self.dendrites.append(self.adend1)
        self.dendrites.append(self.adend2)
        self.dendrites.append(self.adend3)
        self.dendrites.append(self.adend4)
        self.dendrites.append(self.adend5)
        self.dendrites.append(self.adend6)
        self.dendrites.append(self.adend7)
        self.dendrites.append(self.adend8)
        self.dendrites.append(self.adend9)

    def _insert_channels(self):
        for sec in self.somatic:
            sec.insert("ch_HCNp")
            sec.insert("ch_Navp")
            sec.insert("ch_Kdrp")
            sec.insert("ch_KvAproxp")
            sec.insert("pas")

        for sec in self.basal:
            sec.insert("ch_Navp")
            sec.insert("ch_Kdrp")
            sec.insert("ch_KvAproxp")
            sec.insert("pas")

        for sec in self.apical:
            sec.insert("pas")
            if sec.diam > 0.5 and h.distance(self.soma(0.5), sec(0.5) < 500):
                sec.insert("ch_HCNp")
                sec.insert("ch_Navp")
                sec.insert("ch_Kdrp")
                sec.insert("ch_KvAproxp")
                sec.insert("ch_KvAdistp")

        for sec in self.axonal:
            sec.insert("ch_Navaxonp")
            sec.insert("ch_Kdrp")
            sec.insert("ch_KvAproxp")
            sec.insert("pas")

    def _setup_biophysics(self):
        # set variables values
        Vrest = -66

        Rm = 28000
        RmDend = Rm/2
        RmSoma = Rm
        RmAx = Rm

        Cm = 1
        CmSoma = Cm
        CmAx = Cm
        CmDend = Cm*2

        celsius = 34

        RaAll = 150
        RaSoma = 150
        RaAx = 50

        ekval = -90
        enaval = 55
        eHCNval = -30
        eleakval = Vrest # not lower than ekval

        gNav = 0.032 # Nav conductance in mho/cm2
        gNavaxon = 0.064 # axon multiplier for Nav conductance
        gKdr = 0.003 # Kdr conductance in mho/cm2
        gKvAdist = 0.008 # distal KvA conductance in mho/cm2
        gKvAprox = 0.008 # proximal KvA conductance in mho/cm2
        gHCN = 0.0006 # hcurrent conductance in mho/cm2 --> 6 pS/um2

        for sec in self.somatic:
            sec.Ra = RaSoma
            sec.cm = CmSoma
            for seg in sec:
                seg.ch_HCNp.gmax = gHCN
                seg.ch_HCNp.vhalfl = -82
                seg.ch_Navp.gmax = gNav
                seg.ch_Navp.ar2 = 1
                seg.ch_Kdrp.gmax = gKdr
                seg.ch_KvAproxp.gmax = gKvAprox
                seg.pas.e = eleakval
                seg.pas.g = 1/RmSoma

        for sec in self.basal:
            sec.Ra = RaAll
            sec.cm = CmDend
            for seg in sec:
                seg.ch_Navp.gmax = gNav
                seg.ch_Navp.ar2 = 1
                seg.ch_Kdrp.gmax = gKdr
                seg.ch_KvAproxp.gmax = gKvAprox
                seg.pas.e = eleakval
                seg.pas.g = 1/RmDend

        for sec in self.apical:
            sec.Ra = RaAll
            sec.cm = CmDend

            for seg in sec:
                seg.pas.e = eleakval
                seg.pas.g = 1/RmDend
            if sec.diam > 0.5 and h.distance(self.soma(0.5), sec(0.5)) < 500:
                for seg in sec:
                    seg.ch_HCNp.gmax = gHCN
                    seg.ch_Navp.ar2 = 0.8
                    seg.ch_Navp.gmax = gNav
                    seg.ch_Kdrp.gmax = gKdr
                    seg.ch_KvAproxp.gmax = 0
                    seg.ch_KvAdistp.gmax = 0

                    xdist = h.distance(self.soma(0.5), sec(seg.x))
                    if xdist > 500:
                        xdist = 500
                    seg.ch_HCNp.gmax = gHCN*(1+1.5*xdist/100)

                    if xdist > 100:
                        seg.ch_HCNp.vhalfl = -90
                        seg.ch_KvAdistp.gmax = gKvAdist*(1+xdist/100)
                    else:
                        seg.ch_HCNp.vhalfl = -82
                        seg.ch_KvAproxp.gmax = gKvAprox*(1+xdist/100)

        for sec in self.axonal:
            sec.Ra = RaAx
            sec.cm = CmAx
            for seg in sec:
                seg.ch_Navaxonp.gmax = gNavaxon
                seg.ch_Kdrp.gmax = gKdr
                seg.pas.e = eleakval
                seg.pas.g = 1/RmAx
                seg.ch_KvAproxp.gmax = gKvAprox * 0.2

        for sec in self.all:
            sec.v = Vrest
            if h.ismembrane("ch_Navaxonp", sec=sec) or h.ismembrane("ch_Navp", sec=sec):
                for seg in sec: seg.ena = enaval

            if h.ismembrane("ch_Kdrp", sec=sec) or h.ismembrane("ch_KvAproxp", sec=sec) or h.ismembrane("ch_KvAdistp", sec=sec):
                for seg in sec: seg.ek = ekval

            if h.ismembrane("ch_HCNp", sec=sec):
                for seg in sec: seg.ch_HCNp.e = eHCNval

    def _create_synapses(self):
        # from Schaffer collaterals
        for sec in self.rad_list:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 0.5 * ms
            syn_.tau2 = 3 * ms
            syn_.e = 0 * mV
            self._syn_list.append(syn_)

        # from Pyramidal cells
        for sec in self.proximal:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 0.5 * ms
            syn_.tau2 = 3 * ms
            syn_.e = 0 * mV
            self._syn_list.append(syn_)

        # from Basket cells
        for sec in self.somatic:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 1 * ms
            syn_.tau2 = 8 * ms
            syn_.e = -75 * mV
            self._syn_list.append(syn_)

        # from OLM cells
        for sec in self.lm_list:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 1 * ms
            syn_.tau2 = 8 * ms
            syn_.e = -75 * mV
            self._syn_list.append(syn_)
    

class BasketCell(Cell):
    name = "BasketCell"
    def __init__(self, gid: int, x: float = 0, y: float = 0, z: float = 0, theta: float = 0) -> None:
        super().__init__(gid, x, y, z, theta)
        self.syn_dist = 470

    def _setup_morphology(self):
        # create sections
        self.soma = h.Section(name="soma", cell=self)

        self.dend1 = h.Section(name="dend1", cell=self)
        self.dend2 = h.Section(name="dend2", cell=self)
        self.dend3 = h.Section(name="dend3", cell=self)
        self.dend4 = h.Section(name="dend4", cell=self)
        self.dend5 = h.Section(name="dend5", cell=self)
        self.dend6 = h.Section(name="dend6", cell=self)
        self.dend7 = h.Section(name="dend7", cell=self)
        self.dend8 = h.Section(name="dend8", cell=self)
        self.dend9 = h.Section(name="dend9", cell=self)
        self.dend10 = h.Section(name="dend10", cell=self)
        self.dend11 = h.Section(name="dend11", cell=self)
        self.dend12 = h.Section(name="dend12", cell=self)
        self.dend13 = h.Section(name="dend13", cell=self)
        self.dend14 = h.Section(name="dend14", cell=self)
        self.dend15 = h.Section(name="dend15", cell=self)
        self.dend16 = h.Section(name="dend16", cell=self)

        # connect sections
        self.dend1.connect(self.soma(1), 0)
        self.dend2.connect(self.dend1(1), 0)
        self.dend3.connect(self.dend2(1), 0)
        self.dend4.connect(self.dend3(1), 0)
        self.dend5.connect(self.dend4(1), 0)

        self.dend6.connect(self.soma(1), 0)
        self.dend7.connect(self.dend6(1), 0)
        self.dend8.connect(self.dend7(1), 0)
        self.dend9.connect(self.dend8(1), 0)
        self.dend10.connect(self.dend9(1), 0)

        self.dend11.connect(self.soma(0), 0)
        self.dend12.connect(self.dend11(1), 0)
        self.dend13.connect(self.dend12(1), 0)

        self.dend14.connect(self.soma(0), 0)
        self.dend15.connect(self.dend14(1), 0)
        self.dend16.connect(self.dend15(1), 0)

        # set basic shape
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0, 0, 0, 10, sec=self.soma) 
        h.pt3dadd(0, 10, 0, 10, sec=self.soma) 
        h.pt3dadd(0, 20, 0, 10, sec=self.soma) 

        h.pt3dclear(sec=self.dend1)
        h.pt3dadd(0., 20., 0., 4., sec=self.dend1)
        h.pt3dadd(19.4709, 66.053, 0., 4., sec=self.dend1)
        h.pt3dadd(38.9418, 112.106, 0., 4., sec=self.dend1)

        h.pt3dclear(sec=self.dend2)
        h.pt3dadd(38.9418, 112.106, 0., 3., sec=self.dend2)
        h.pt3dadd(58.4128, 158.159, 0., 3., sec=self.dend2)
        h.pt3dadd(77.8837, 204.212, 0., 3., sec=self.dend2)

        h.pt3dclear(sec=self.dend3)
        h.pt3dadd(77.8837, 204.212, 0., 2., sec=self.dend3)
        h.pt3dadd(116.826, 296.318, 0., 2., sec=self.dend3)
        h.pt3dadd(155.767, 388.424, 0., 2., sec=self.dend3)

        h.pt3dclear(sec=self.dend4)
        h.pt3dadd(155.767, 388.424, 0., 1.5, sec=self.dend4)
        h.pt3dadd(175.238, 434.477, 0., 1.5, sec=self.dend4)
        h.pt3dadd(194.709, 480.531, 0., 1.5, sec=self.dend4)

        h.pt3dclear(sec=self.dend5)
        h.pt3dadd(194.709, 480.531, 0., 1., sec=self.dend5)
        h.pt3dadd(214.18, 526.584, 0., 1., sec=self.dend5)
        h.pt3dadd(233.651, 572.637, 0., 1., sec=self.dend5)

        h.pt3dclear(sec=self.dend6)
        h.pt3dadd(0., 20., 0., 4., sec=self.dend6)
        h.pt3dadd(-19.4709, 66.053, 0., 4., sec=self.dend6)
        h.pt3dadd(-38.9418, 112.106, 0., 4., sec=self.dend6)

        h.pt3dclear(sec=self.dend7)
        h.pt3dadd(-38.9418, 112.106, 0., 3., sec=self.dend7)
        h.pt3dadd(-58.4128, 158.159, 0., 3., sec=self.dend7)
        h.pt3dadd(-77.8837, 204.212, 0., 3., sec=self.dend7)

        h.pt3dclear(sec=self.dend8)
        h.pt3dadd(-77.8837, 204.212, 0., 2., sec=self.dend8)
        h.pt3dadd(-116.826, 296.318, 0., 2., sec=self.dend8)
        h.pt3dadd(-155.767, 388.424, 0., 2., sec=self.dend8)

        h.pt3dclear(sec=self.dend9)
        h.pt3dadd(-155.767, 388.424, 0., 1.5, sec=self.dend9)
        h.pt3dadd(-175.238, 434.477, 0., 1.5, sec=self.dend9)
        h.pt3dadd(-194.709, 480.531, 0., 1.5, sec=self.dend9)

        h.pt3dclear(sec=self.dend10)
        h.pt3dadd(-194.709, 480.531, 0., 1., sec=self.dend10)
        h.pt3dadd(-214.18, 526.584, 0., 1., sec=self.dend10)
        h.pt3dadd(-233.651, 572.637, 0., 1., sec=self.dend10)

        h.pt3dclear(sec=self.dend11)
        h.pt3dadd(0., 0., 0., 2., sec=self.dend11)
        h.pt3dadd(-19.4709, -46.0531, 0., 2., sec=self.dend11)
        h.pt3dadd(-38.9418, -92.1061, 0., 2., sec=self.dend11)

        h.pt3dclear(sec=self.dend12)
        h.pt3dadd(-38.9418, -92.1061, 0., 1.5, sec=self.dend12)
        h.pt3dadd(-58.4128, -138.159, 0., 1.5, sec=self.dend12)
        h.pt3dadd(-77.8837, -184.212, 0., 1.5, sec=self.dend12)

        h.pt3dclear(sec=self.dend13)
        h.pt3dadd(-77.8837, -184.212, 0., 1., sec=self.dend13)
        h.pt3dadd(-97.3546, -230.265, 0., 1., sec=self.dend13)
        h.pt3dadd(-116.826, -276.318, 0., 1., sec=self.dend13)

        h.pt3dclear(sec=self.dend14)
        h.pt3dadd(0., 0., 0., 2., sec=self.dend14)
        h.pt3dadd(19.4709, -46.053, 0., 2., sec=self.dend14)
        h.pt3dadd(38.9419, -92.1061, 0., 2., sec=self.dend14)

        h.pt3dclear(sec=self.dend15)
        h.pt3dadd(38.9419, -92.1061, 0., 1.5, sec=self.dend15)
        h.pt3dadd(58.4128, -138.159, 0., 1.5, sec=self.dend15)
        h.pt3dadd(77.8837, -184.212, 0., 1.5, sec=self.dend15)

        h.pt3dclear(sec=self.dend16)
        h.pt3dadd(77.8837, -184.212, 0., 1., sec=self.dend16)
        h.pt3dadd(97.3546, -230.265, 0., 1., sec=self.dend16)
        h.pt3dadd(116.826, -276.318, 0., 1., sec=self.dend16)

    def _setup_dimensions(self):
        self.soma.L = 20
        self.soma.diam = 10

        self.dend1.L = 98.677
        self.dend1.diam = 4

        self.dend2.L = 99.883
        self.dend2.diam = 3

        self.dend3.L = 199.934
        self.dend3.diam = 2

        self.dend4.L = 99.986
        self.dend4.diam = 1.5

        self.dend5.L = 99.99
        self.dend5.diam = 1

        self.dend6.L = 98.677
        self.dend6.diam = 4

        self.dend7.L = 99.883
        self.dend7.diam = 3

        self.dend8.L = 199.934
        self.dend8.diam = 2

        self.dend9.L = 99.986
        self.dend9.diam = 1.5

        self.dend10.L = 99.99
        self.dend10.diam = 1

        self.dend11.L = 100
        self.dend11.diam = 2

        self.dend12.L = 100
        self.dend12.diam = 1.5

        self.dend13.L = 100
        self.dend13.diam = 1

        self.dend14.L = 100
        self.dend14.diam = 2

        self.dend15.L = 100
        self.dend15.diam = 1.5

        self.dend16.L = 100
        self.dend16.diam = 1

    def _setup_subsections(self):
        self.all = h.SectionList()
        self.all.append(self.soma)
        self.all.append(self.dend1)
        self.all.append(self.dend2)
        self.all.append(self.dend3)
        self.all.append(self.dend4)
        self.all.append(self.dend5)
        self.all.append(self.dend6)
        self.all.append(self.dend7)
        self.all.append(self.dend8)
        self.all.append(self.dend9)
        self.all.append(self.dend10)
        self.all.append(self.dend11)
        self.all.append(self.dend12)
        self.all.append(self.dend13)
        self.all.append(self.dend14)
        self.all.append(self.dend15)
        self.all.append(self.dend16)

        self.somatic = h.SectionList()
        self.somatic.append(self.soma)

        self.dendrites = h.SectionList()
        self.dendrites.append(self.dend1)
        self.dendrites.append(self.dend2)
        self.dendrites.append(self.dend3)
        self.dendrites.append(self.dend4)
        self.dendrites.append(self.dend5)
        self.dendrites.append(self.dend6)
        self.dendrites.append(self.dend7)
        self.dendrites.append(self.dend8)
        self.dendrites.append(self.dend9)
        self.dendrites.append(self.dend10)
        self.dendrites.append(self.dend11)
        self.dendrites.append(self.dend12)
        self.dendrites.append(self.dend13)
        self.dendrites.append(self.dend14)
        self.dendrites.append(self.dend15)
        self.dendrites.append(self.dend16)

        self.basal = h.SectionList()
        self.basal.append(self.dend11)
        self.basal.append(self.dend12)
        self.basal.append(self.dend13)
        self.basal.append(self.dend14)
        self.basal.append(self.dend15)
        self.basal.append(self.dend16)

        self.apical = h.SectionList()
        self.apical.append(self.dend1)
        self.apical.append(self.dend2)
        self.apical.append(self.dend3)
        self.apical.append(self.dend4)
        self.apical.append(self.dend5)
        self.apical.append(self.dend6)
        self.apical.append(self.dend7)
        self.apical.append(self.dend8)
        self.apical.append(self.dend9)
        self.apical.append(self.dend10)

        self.sr_med = h.SectionList()
        self.sr_med.append(self.dend3)
        self.sr_med.append(self.dend4)
        self.sr_med.append(self.dend8)
        self.sr_med.append(self.dend9)

        self.proximal = h.SectionList()
        self.proximal.append(self.dend1)
        self.proximal.append(self.dend2)
        self.proximal.append(self.dend6)
        self.proximal.append(self.dend7)
        self.proximal.append(self.dend11)
        self.proximal.append(self.dend14)

        self.proximal_apical = h.SectionList()
        self.proximal_apical.append(self.dend1)
        # self.proximal_apical.append(self.dend2)
        self.proximal_apical.append(self.dend6)
        # self.proximal_apical.append(self.dend7)

        self.distal_apical = h.SectionList()
        self.distal_apical.append(self.dend5)
        self.distal_apical.append(self.dend10)

    def _insert_channels(self):
        for sec in self.all:
            sec.insert("ch_KvA")
            sec.insert("ch_CavN")
            sec.insert("ch_CavL")
            sec.insert("ch_KCaS")
            sec.insert("ch_KvCaB")

        for sec in self.somatic:
            sec.insert("ch_Navaxonp")
            sec.insert("ch_Kdrfast")
            sec.insert("ch_leak")

        for sec in self.dendrites:
            sec.insert("ch_Navaxonp")
            sec.insert("ch_Kdrfast")
            sec.insert("ch_leak")

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100
            sec.cm = 1.4


        # set variables values
        Vrest = -65
        celsius = 34.

        Rm = 5555

        # calcium concentrations in mM
        ca_outside = 2
        ca_inside = 5.e-6
        catau = 10

        # reversal potentials
        ekval = -90
        enaval = 55
        eHCNval = -30
        ecaval = 8.314*(273.15+celsius)/(2*9.649e4)*np.log(ca_outside/ca_inside)*1000

        if (Vrest < ekval): Vrest = ekval
        if (Vrest > enaval): Vrest = enaval

        eleakval = Vrest

        # max ion channel conductances in mho/cm2
        gNav = 0.15
        gKdr = 0.013 # Delayed rectifier potassium
        gKvA = 0.00015 # Proximal A-type potassium
        gHCN = 0.00002 # HCN (hyperpolarization-activated cyclic nucleotide-gated channel)
        gCavN = 0.0008 # T-type calcium
        gCavL = 0.005 # L-type calcium
        gKvCaB = 0.0000002 # Big potassium channel: voltage and calcium gated 
        gKCaS = 0.000002 # Small potassium channel: calcium gated

        for sec in self.all:
            for seg in sec:
                seg.ch_KvA.gmax = gKvA
                seg.ch_CavN.gmax = gCavN
                seg.ch_CavL.gmax = gCavL
                seg.ch_KCaS.gmax = gKCaS
                seg.ch_KvCaB.gmax = gKvCaB

        for sec in self.somatic:
            for seg in sec:
                seg.ch_Navaxonp.gmax = gNav
                seg.ch_Kdrfast.gmax = gKdr
                seg.ch_leak.gmax = 1/Rm

        for sec in self.dendrites:
            for seg in sec:
                seg.ch_Navaxonp.gmax = gNav
                seg.ch_Kdrfast.gmax = gKdr
                seg.ch_leak.gmax = 1/Rm

        for sec in self.all:
            for seg in sec:
                seg.ena = enaval
                seg.ek = ekval
                seg.eca = ecaval
                seg.ch_leak.e = eleakval
                # seg.iconc_Ca.cao = ca_outside
        

    def _create_synapses(self):
        # synapses from Schaffer collaterals
        for sec in self.sr_med:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 0.5 * ms
            syn_.tau2 = 3 * ms
            syn_.e = 0 * mV
            self._syn_list.append(syn_)

        # synapses from Pyramidal cells connextions
        for sec in self.proximal_apical:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 0.5 * ms
            syn_.tau2 = 3 * ms
            syn_.e = 0 * mV
            self._syn_list.append(syn_)

        # synapses from basket cells connections
        for sec in self.somatic:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 1 * ms
            syn_.tau2 = 8 * ms
            syn_.e = -75 * mV
            self._syn_list.append(syn_)
        

       

class OLMCell(Cell):
    name = "OLMCell"
    def __init__(self, gid: int,
                 x: float=0.,
                 y: float=0.,
                 z: float=0.,
                 theta: float=np.pi) -> None:
        super().__init__(gid, x, y, z, theta)
        self.syn_dist = 705.13*1.5

    def _setup_morphology(self):
        # create sections
        self.soma = h.Section(name="soma", cell=self)
        self.dend1 = h.Section(name="dend1", cell=self)
        self.dend2 = h.Section(name="dend2", cell=self)
        self.axon = h.Section(name="axon", cell=self)

        # connect sections
        self.dend1.connect(self.soma(1), 0)
        self.dend2.connect(self.soma(1), 0)
        self.axon.connect(self.soma(0), 0)

        # set basic shape
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0., 0., 0., 10., sec=self.soma)
        h.pt3dadd(0., 10., 0., 10., sec=self.soma)
        h.pt3dadd(0., 20., 0., 10., sec=self.soma)

        h.pt3dclear(sec=self.dend1)
        h.pt3dadd(0., 20., 0., 3., sec=self.dend1)
        h.pt3dadd(100., 120., 0., 3., sec=self.dend1)
        h.pt3dadd(177., 197., 0., 3., sec=self.dend1)

        h.pt3dclear(sec=self.dend2)
        h.pt3dadd(0., 20., 0., 3., sec=self.dend2)
        h.pt3dadd(-100., 120., 0., 3., sec=self.dend2)
        h.pt3dadd(-177., 197., 0., 3., sec=self.dend2)

        h.pt3dclear(sec=self.axon)
        h.pt3dadd(0., 0., 0., 1.5, sec=self.axon)
        h.pt3dadd(0., -75., 0., 1.5, sec=self.axon)
        h.pt3dadd(0., -150., 0., 1.5, sec=self.axon)

    def _setup_dimensions(self):
        self.soma.L = 20
        self.soma.diam = 10

        self.dend1.L = 250
        self.dend1.diam = 3

        self.dend2.L = 250
        self.dend2.diam = 3

        self.axon.L = 150
        self.axon.diam = 1.5

    def _setup_subsections(self):
        self.all = h.SectionList()
        self.all.append(self.soma)
        self.all.append(self.dend1)
        self.all.append(self.dend2)
        self.all.append(self.axon)

        self.somatic = h.SectionList()
        self.somatic.append(self.soma)

        self.basal = h.SectionList()
        self.basal.append(self.dend1)
        self.basal.append(self.dend2)

        self.axonal = h.SectionList()
        self.axonal.append(self.axon)

    def _insert_channels(self):
        # soma
        self.soma.insert("ch_KvAolm")
        self.soma.insert("ch_HCNolm")
        self.soma.insert("ch_Kdrfast")
        self.soma.insert("ch_Nav")
        self.soma.insert("ch_leak")

        # dendrites
        for sec in self.basal:
            sec.insert("ch_KvAolm")
            sec.insert("ch_Kdrfast")
            sec.insert("ch_Nav")
            sec.insert("ch_leak")

        # axon
        self.axon.insert("ch_Kdrfast")
        self.axon.insert("ch_Nav")
        self.axon.insert("ch_leak")

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 150 # Axial resistance in Ohm * cm
            sec.cm = 1.3 # Membrane capacitance in micro Farads / cm^2

        # set variables values
        Rm = 20000 * 5
        gH = 0.0005

        gKvAsoma = 0.0165 * 0.3
        gKvAdend = 0.004 * 0.7

        # *2.3 gives 2x frequency & no depol. block
        gKvEaxon =  0.05104*2.3
        gKvEsoma =  0.0319*2.3
        gKvEdend =  2*0.023*2.3
        
        gNasoma = 0.0107
        gNadend = 2*0.0117
        gNaaxon = 0.01712
        
        eleak = -67	* mV

        # soma
        for seg in self.soma:
            seg.ch_KvAolm.gmax = gKvAsoma
            seg.ch_HCNolm.gmax = gH
            seg.ch_Kdrfast.gmax = gKvEsoma
            seg.ch_Nav.gmax = gNasoma
            seg.ch_leak.gmax = 1/Rm
            seg.ch_leak.e = eleak

        # dendrites
        for sec in self.basal:
            for seg in sec:
                seg.ch_KvAolm.gmax = gKvAdend
                seg.ch_Kdrfast.gmax = gKvEdend
                seg.ch_Nav.gmax = gNadend
                seg.ch_leak.gmax = 1/Rm
                seg.ch_leak.e = eleak

        # axon
        for seg in self.axon:
            seg.ch_Kdrfast.gmax = gKvEaxon
            seg.ch_Nav.gmax = gNaaxon
            seg.ch_leak.gmax = 1/Rm
            seg.ch_leak.e = eleak

    def _create_synapses(self):
        # synapses from pyramidal cells connections
        for sec in self.basal:
            syn_ = h.Exp2Syn(sec(0.5))
            syn_.tau1 = 0.5 * ms
            syn_.tau2 = 3 * ms
            syn_.e = 0 * mV
            self._syn_list.append(syn_)


       

    





