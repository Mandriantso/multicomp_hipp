TITLE Ohmic external noisy Current
:
: Ohmic external noisy current that can be set on a per-section basis
:
UNITS {
    (mcA) = (microamp)
    (mcF) = (microfarad)
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (S) = (siemens)
    (mS) = (millisiemens)
}

: INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX Inoise
    NONSPECIFIC_CURRENT inoise
    RANGE mean, sigma, myseed, tau, inoise
}

PARAMETER {
    mean = 0. (nA)
    sigma = 0.4 (nA)
    tau = 15 (ms)
    myseed = 1
    :dt (ms)
}

ASSIGNED {
    :inoise  (nA)
    : dt (ms)
}

STATE { inoise}

INITIAL {
    :inoise = 0
    :iext = 0 
    set_seed(myseed)
}

BREAKPOINT {
    SOLVE states METHOD euler
    :inoise = iext
}

DERIVATIVE states {   
	: noise()      
	inoise' = (mean - inoise)/tau + sigma * sqrt(2/tau) * normrand(0, 1)/sqrt(dt) 
}

PROCEDURE noise() {
    : printf("%f", dt )
    : iext = - mean + sigma * normrand(0, 1) * sqrt(1/dt)  
    : noise = sigma * sqrt(2/tau) * normrand(0, 1)/sqrt(dt) 

}

