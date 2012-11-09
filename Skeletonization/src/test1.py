# Author: Prabhu Ramachandran <prabhu at aero dot iitb dot ac dot in>
# Copyright (c) 2007, Enthought, Inc.
# License: BSD style.

import numpy as np
from numpy import array
from tvtk.api import tvtk
from mayavi.scripts import mayavi2

# The numpy array data.
#points = array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], 'f')
corners = np.matrix(np.zeros((8, 3)))
radius = 1.
center = [0., 0., 0.]
corners[0:4, 2] = center[2] - radius
corners[4:, 2] = center[2] + radius
corners[::2, 0] = center[0] - radius
corners[1::2, 0] = center[0] + radius
corners[np.ix_([2,3,6,7],[1])] = center[1] - radius
corners[np.ix_([0,1,4,5],[1])] = center[1] + radius
points = corners
squares = np.array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])

# The TVTK dataset.
mesh = tvtk.PolyData(points = points, polys = squares)
#mesh.point_data.scalars = temperature
#mesh.point_data.scalars.name = 'Temperature'

# Uncomment the next two lines to save the dataset to a VTK XML file.
#w = tvtk.XMLPolyDataWriter(input=mesh, file_name='polydata.vtp')
#w.write()

# Now view the data.
@mayavi2.standalone
def view():
                
    from mayavi.sources.vtk_data_source import VTKDataSource
    from mayavi.modules.surface import Surface
    

    mayavi.new_scene()
    src = VTKDataSource(data = mesh)
    mayavi.add_source(src)
    s = Surface()    
    mayavi.add_module(s)

if __name__ == '__main__':
    view()