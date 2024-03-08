from numpy import exp,sin,cos,tan,random,mgrid,ogrid,linspace,sqrt,pi
import numpy as np
from mayavi import mlab

def peaks(x,y):
    return 3.0*(1.0-x)**2*exp(-(x**2) - (y+1.0)**2) - 10*(x/5.0 - x**3 - y**5) * exp(-x**2-y**2) - 1.0/3.0*exp(-(x+1.0)**2 - y**2)
y,x = np.mgrid[0:5:70j,0:5:70j]
z=peaks(x,y)
mlab.mesh(x,y,z)
mlab.colorbar()
mlab.show()


