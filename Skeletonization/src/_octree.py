from numpy import array, random, linspace, pi, ravel, cos, sin, empty, zeros, ix_
from tvtk.api import tvtk
import PointCloud

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi import mlab

class Point:    
    def __init__(self, position, id):
        
        self.position = position # (x, y, z)
        self.id = id
        self.code = 0;
        
class Bounds:
    
    def __init__(self, center, radius):
        
        self.center = center
        self.radius = radius
    

class OctreeNode():
    
    def __init__(self, center, radius, theshold, maxDepth):
                
        self.center = center
        self.radius = radius        
        self.threshold = theshold
        self.maxDepth = maxDepth
        
        self.children = [None, None, None, None, None, None, None, None]
        self.boundsOffsetTable = array([[-0.5, -0.5, -0.5],
                                  [+0.5, -0.5, -0.5],
                                  [-0.5, +0.5, -0.5],
                                  [+0.5, +0.5, -0.5],
                                  [-0.5, -0.5, +0.5],
                                  [+0.5, -0.5, +0.5],
                                  [-0.5, +0.5, +0.5],
                                  [+0.5, +0.5, +0.5]])
    
    def Build(self, points):
        
        # Build the octree
        if points.size() < self.threshold:
            
            self.nPoints = points.size()
            # copy points ? TODO
            self.points = points
            return
                
        childPointsCount = zeros(8)
        childPointsIndices = [[], [], [], [], [], [], [], []] 
                
        for p, i in self.points, range(0, self.nPoints):
                                
#            Here, we need to know which child each point belongs to. To
#            do this, we build an index into the _child[] array using the
#            relative position of the point to the center of the current
#            node

            code = 0
            if p[0] > self.center[0]:
                code |= 1
                 
            if p[1] > self.center[1]:
                code |= 2
                 
            if p[2] > self.center[2]:
                code |= 4;
                 
#            We'll need to keep track of how many points get stuck in each
#            child so we'll just keep track of it here, since we have the
#            information handy.

            childPointsCount[code] += 1
            childPointsIndices[code].append(i)
            
            
        for i in range (0, 8):
            
            if childPointsCount[i] < self.threshold:                
                continue
            
            center = self.center + self.boundsOffsetTable[i] * self.radius
            radius = self.radius/2.
            
            self.children[i] = OctreeNode(center, radius, self.theshold, self.maxDepth)
            childPoints = points[ix_(childPointsIndices[code], )]            
            self.children[i].Build(childPoints)          
            
    def Render(self):    
        
        dataset = self.Outline()
        mlab.figure(bgcolor = (1, 1, 1), fgcolor = (0, 0, 0), figure = dataset.class_name[3:])
        surf = mlab.pipeline.surface(dataset, opacity = .01)
        mlab.pipeline.surface(mlab.pipeline.extract_edges(surf), color = (0, 0, 0), )
        
    def Outline(self):
        
        corners = array(zeros((9, 3)))
        corners[0:4, 2] = self.center[2] - self.radius
        corners[4:, 2] = self.center[2] + self.radius
        corners[::2, 0] = self.center[0] - self.radius
        corners[1::2, 0] = self.center[0] + self.radius
        corners[ix_([2,3,6,7],[1])] = self.center[1] - self.radius
        corners[ix_([0,1,4,5],[1])] = self.center[1] + self.radius                
        squares = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])        
        mesh = tvtk.PolyData(points = corners, polys = squares)
        return mesh
'''
Octree of 3D points
'''
class Octree:
    
    def __init__(self, maxDepth, threshold):
        
        self.maxDepth = maxDepth
        self.threshold = threshold
        
        self.nPoints = 0    
        
            
    def Build(self, points):
        
        self.nPoints = points.shape[0]
        self.root = OctreeNode(points)
            
    @mlab.show
    def Render(self):
        
        self.octree.Render()
            
if __name__ == "__main__":
    
    pointCloud = PointCloud.PointCloud()
    octree = Octree(10, 100)
    octree.Build(points)
