import numpy
from mayavi.mlab import *
from mayavi import mlab

t = numpy.linspace(0, 4*numpy.pi, 20)
cos = numpy.cos
sin = numpy.sin

x = sin(2*t)
y = cos(t)
z = cos(2*t)
s = numpy.array(numpy.ones(20)) 



mlab.figure(1, bgcolor=(1, 1, 1))
pts = points3d(x, y, z, s, scale_factor=.25, color=(0, 1, 0))
#mlab.clf()
#pts = mlab.points3d(x, y, z, 1.5*s.max() - s, scale_factor = 0.015, resolution=10)
##pts.mlab_source.dataset.lines = np.array(connections)
#
#pts.glyph.glyph.clamping = False
#
#mlab.view(49, 31.5, 52.8, (4.2, 37.3, 20.6))
mlab.show()

