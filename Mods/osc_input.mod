: $Id: izap.mod,v 1.4 2015/05/28 02:42:00 ted Exp ted $

COMMENT
izap.mod

Delivers an oscillating current that starts at t = del >= 0.
The frequency of the oscillation increases linearly with time
from f0 at t == del to f1 at t == del + dur, 
where both del and dur are > 0.

Uses event delivery system to ensure compatibility with adaptive integration.

Original implementation 12/4/2008 NTC

Modified to be strictly positive 14/03/2025 MA
ENDCOMMENT

NEURON {
  POINT_PROCESS Iosc
  RANGE delay, dur, amp, f, i
  ELECTRODE_CURRENT i
}

UNITS {
  (nA) = (nanoamp)
  PI = (pi) (1)
}

PARAMETER {
  delay (ms)
  dur (ms)
  f (1/s)  : frequency is in Hz
  amp (nA)
}

ASSIGNED {
  i (nA)
  on (1)
}

INITIAL {
  i = 0
  on = 0

  if (delay<0) { delay=0 }
  if (dur<0) { dur=0 }

  : do nothing if dur == 0
  if (dur>0) {
    net_send(delay, 1)  : to turn it on and start frequency ramp
  }
}

COMMENT
The angular velocity in radians/sec is w = 2*PI*f, 
where f is the instantaneous frequency in Hz.

Assume for the moment that the frequency ramp starts at t = 0.
f = f0 + (f1 - f0)*t/dur

Then the angular displacement is
theta = 2*PI * ( f0*t + (f1 - f0)*(t^2)/(2*dur) ) 
      = 2*PI * t * (f0 + (f1 - f0)*t/(2*dur))
But the ramp starts at t = del, so just substitute t-del for every occurrence of t
in the formula for theta.
ENDCOMMENT

BREAKPOINT {
  if (on==0) {
    i = 0
  } else {
    i = amp * (sin( 2*PI * (t-delay-(1/(4*f*0.001))) * f * (0.001) )+1)/2 
  }
}

NET_RECEIVE (w) {
  : respond only to self-events with flag > 0
  if (flag == 1) {
    if (on==0) {
      on = 1  : turn it on
      net_send(dur, 1)  : to stop frequency ramp, freezing frequency at f1
    } else {
      on = 0  : turn it off
    }
  }
}
