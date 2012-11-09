import numpy as np

#import enthought.mayavi as mayavi

# The TVTK dataset.
import numpy
import mayavi.mlab as mlab
from tvtk.api import tvtk
from mayavi.scripts import mayavi2

# Create some random data
x, y, z = np.array([0., 1.]), np.array([0., 1.]), np.array([0., 1.])
val = np.array([1.0, 0.5])

# Plot and show in mayavi2
pts = mlab.points3d(x, y, z)
mlab.show()

exit()
@mayavi2.standalone
def drawCubeMesh():
    
    from mayavi.sources.vtk_data_source import VTKDataSource
    from enthought.mayavi.modules.outline import Outline
    from enthought.mayavi.modules.grid_plane import GridPlane
    from mayavi.modules.surface import Surface

    center = [0., 0., 0.]
#    mlab.points3d(center[0], center[1], center[2])
    
    corners = np.matrix(np.zeros((8, 3)))
    print corners
    print corners[0:4, 2] 
    radius = 1.
    corners[0:4, 2] = center[2] - radius
    corners[4:, 2] = center[2] + radius
    corners[::2, 0] = center[0] - radius
    corners[1::2, 0] = center[0] + radius
    corners[np.ix_([2,3,6,7],[1])] = center[1] - radius
    corners[np.ix_([0,1,4,5],[1])] = center[1] + radius
    print corners

#    mlab.points3d(corners[:, 0], corners[:, 1], corners[:, 2])
#    mlab.show()
    squares = np.array([[0,1,2,3], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])
    mesh = tvtk.PolyData(points = np.array(corners, 'f'), polys = squares)
    ##mesh.point_data.scalar_mul()        
    #mesh.point_data.scalars = self.rgb
    #mesh.point_data.scalars.name = 'rgb'
    ##mesh.point_data.scalars = self.intensity
    ##mesh.point_data.scalars.name = 'Intensity'
    #                
    #verts = np.arange(0, corners.shape[0], 1)
    #verts.shape = (corners.shape[0], 1)
    #mesh.verts = verts
    
    
    #
    mayavi.new_scene() 
    d = VTKDataSource()
    d.data = mesh
    mayavi.add_source(d)    
    s = Surface()
    mayavi.add_module(s)
    s.actor.property.set(representation = 'p', point_size = 2)
    
if __name__ == "__main__":
    
    drawCubeMesh()