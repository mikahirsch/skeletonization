#Data representation in Mayavi http://docs.enthought.com/mayavi/mayavi/data.html
#Code was adapted to python from c++: http://www.flipcode.com/archives/Octree_Implementation.shtml
# Graph visualization in python : http://www.philipbjorge.com/, http://www.youtube.com/watch?v=FjY3EPR369o
from numpy import array, random, linspace, pi, ravel, cos, sin, empty, zeros, ix_, arange, ones
from tvtk.api import tvtk
import PointCloud
from random import randint, random
from time import time, clock

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi import mlab

class Point:    

    def __init__(self, position, index):
        
        self.position = position # (x, y, z)
        self.index = index
        self.code = 0;
        
class Bounds:
    
    def __init__(self, center, radius):
        
        self.center = center
        self.radius = radius
    
class OctreeNode():
    
    def __init__(self, center, radius, theshold, maxDepth):

        self.isLeaf = False                
        self.center = center
        self.radius = radius        
        self.threshold = theshold
        self.maxDepth = maxDepth
#        self.color = (1., 0., 0.)        
#        self.color = (randint(0,255), randint(0, 255), randint(0, 255))
        self.color = (random(), random(), random())
        #print "self.color = ", self.color 
        
        self.children = [None, None, None, None, None, None, None, None]
#        self.boundsOffsetTable = array([[-0.5, -0.5, -0.5],
#                                        [+0.5, -0.5, -0.5],
#                                        [-0.5, +0.5, -0.5],
#                                        [+0.5, +0.5, -0.5],
#                                        [-0.5, -0.5, +0.5],
#                                        [+0.5, -0.5, +0.5],
#                                        [-0.5, +0.5, +0.5],
#                                        [+0.5, +0.5, +0.5]])
        self.boundsOffsetTable = array([[-0.5, -0.5, -0.5],
                                        [+0.5, -0.5, -0.5],
                                        [-0.5, -0.5, +0.5],
                                        [+0.5, -0.5, +0.5],
                                        [-0.5, +0.5, -0.5],
                                        [+0.5, +0.5, -0.5],
                                        [-0.5, +0.5, +0.5],
                                        [+0.5, +0.5, +0.5]])
    
    def Build(self, points, currentDepth, octree):
        
        # Build the octree
        if points.shape[0] <= self.threshold or currentDepth >= self.maxDepth:
            
            self.isLeaf = True
            self.nPoints = points.shape[0]
            # copy points ? TODO
            self.points = points
            # Prepare for rendering
            self.index = octree.leafCount
            self.OutlineVerticesAndFacets()
            self.Centroid()      
            octree.leaves.append(self)       
            octree.leafCount += 1                    
            self.code = 0# In which octant are we? 
            return
                
        childPointsCount = zeros(8)
        childPointsIndices = [[], [], [], [], [], [], [], []] 
                
        for i in range(0, points.shape[0]):
                                
#            Here, we need to know which child each point belongs to. To
#            do this, we build an index into the _child[] array using the
#            relative position of the point to the center of the current
#            node

            p = points[i]

            code = 0
            if p[0] > self.center[0]:# Right half
                code |= 1
                 
            if p[1] > self.center[1]:# Bottom half
                code |= 2
                 
            if p[2] > self.center[2]:# Front Half
                code |= 4;
                 
#            We'll need to keep track of how many points get stuck in each
#            child so we'll just keep track of it here, since we have the
#            information handy.

            print 'code = ', code
            childPointsCount[code] += 1
            childPointsIndices[code].append(i)
                        
        for i in range (0, 8):
            
            if childPointsCount[i] == 0:                
                continue
            
            center = self.center + self.boundsOffsetTable[i] * self.radius
            radius = self.radius/2.
            
            self.children[i] = OctreeNode(center, radius, self.threshold, self.maxDepth)
            childPoints = points[ix_(childPointsIndices[i], )]            
            self.children[i].Build(childPoints, currentDepth + 1, octree)
            
    def Centroid(self): 
        # Calculate the centroid of the cell
        self.centroid = array(zeros(3))  
        self.centroid[0] = sum(self.points[:, 0]/self.nPoints) 
        self.centroid[1] = sum(self.points[:, 1]/self.nPoints)
        self.centroid[2] = sum(self.points[:, 2]/self.nPoints)
    
            
    def Render(self, vertices, facets, centroids):
        
        if self.isLeaf:
            #print 'Number of points in leaf:', self.nPoints             
            # aggregate vertices and facets
            vertices[self.index * 8:self.index * 8 + 8] = self.vertices
            facets[self.index * 6:self.index * 6 + 6] = self.facets                                                
            centroids[self.index] = self.centroid
            return
         
        else:        
            for child in self.children:
                if child != None:
                    child.Render(vertices, facets, centroids)                        
        
    def Render2(self):
        
        if self.isLeaf:
            print 'Number of points in leaf:', self.nPoints 
            if self.nPoints < 10:
                return 
            # Draw outline of node
            dataset = self.Outline()
                        
            surf = mlab.pipeline.surface(dataset, opacity = .001)
            mlab.pipeline.surface(mlab.pipeline.extract_edges(surf), color = (0, 0, 0))
            if self.renderPoints:            
                self.RenderPoints()
            return
         
        else:        
            for child in self.children:
                if child != None:
                    child.Render()
        
    def Cloud(self):
        
        cloud = tvtk.PolyData(points = array(self.points, 'f'))
#        cloud.point_data.scalars = self.rgb
#        cloud.point_data.scalars.name = 'rgb'
                        
        verts = arange(0, self.points.shape[0], 1)
        verts.shape = (self.points.shape[0], 1)
        cloud.verts = verts
        
        return cloud
    
    def RenderPoints(self, data = None): # In a different language I would force this method to implement a certain interface, let's call it ITraverseMethod... The data parameter is all I have left...
                                
        cloud = self.Cloud()
        #mlab.figure(bgcolor = (1, 1, 1))
        mlab.pipeline.surface(cloud, color = self.color)
        
    #def RenderOutline(self):
        
        
    def Outline(self):
                       
        corners = array(zeros((9, 3)))
        # Y
        corners[0:4, 1] = self.center[2] - self.radius
        corners[4:, 1] = self.center[2] + self.radius
        # X
        corners[::2, 0] = self.center[0] - self.radius
        corners[1::2, 0] = self.center[0] + self.radius
        # Z
        corners[ix_([2,3,6,7],[2])] = self.center[1] - self.radius
        corners[ix_([0,1,4,5],[2])] = self.center[1] + self.radius                
        squares = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])        
        mesh = tvtk.PolyData(points = corners, polys = squares)
        return mesh
    
    def OutlineVerticesAndFacets(self):
        
        self.vertices= array(zeros((8, 3)))
        # Y
        self.vertices[0:4, 2] = self.center[2] - self.radius
        self.vertices[4:, 2] = self.center[2] + self.radius
        # X
        self.vertices[::2, 0] = self.center[0] - self.radius    
        self.vertices[1::2, 0] = self.center[0] + self.radius
        # Z
        self.vertices[ix_([2,3,6,7],[1])] = self.center[1] - self.radius
        self.vertices[ix_([0,1,4,5],[1])] = self.center[1] + self.radius
                        
        self.facets = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])
        self.facets += self.index * 8
'''
Octree of 3D points
'''
class Octree:
    
    def __init__(self, maxDepth, threshold):
        
        self.maxDepth = maxDepth
        self.threshold = threshold
        self.renderPoints = True
        self.nPoints = 0    
        self.renderVerticesAndFacets = True
        self.renderCentroids = False
        self.leaves= []
                 
    def Build(self, points):
                
        self.nPoints = points.shape[0]
        self.root = OctreeNode(points)
        
    def BuildFromPointCloud(self, pointCloud):
        
        self.nPoints = pointCloud.Points().shape[0]
        self.root = OctreeNode(pointCloud.BoundingCube().center, pointCloud.BoundingCube().radius, self.threshold, self.maxDepth)
        self.leafCount = 0
        self.root.Build(pointCloud.Points(), 0, self)
        self.points = pointCloud.Points()        
        
    def TraverseLeaves(self, function, data = None):
        
        for node in self.leaves:
            function(node, data)
                                             
#    def Traverse(self, function, data): # Traverses all nodes and performs function on each node
#        
#        function(self, data)
#        for
        

    def RenderPoints(self, useColors = False):
        
        if not self.renderPoints:
            return         
    
        if useColors:
            self.TraverseLeaves(OctreeNode.RenderPoints)
         
        else:               
            cloud = tvtk.PolyData(points = array(self.points, 'f'))                        
            verts = arange(0, self.points.shape[0], 1)
            verts.shape = (self.points.shape[0], 1)
            cloud.verts = verts        
            mlab.pipeline.surface(cloud, color = (0, 1, 0))            
            
    @mlab.show
    def Render(self, optimized):

                        
        t1clock = clock()        
        mlab.figure(bgcolor = (1, 1, 1), fgcolor = (0, 0, 0))#, figure = dataset.class_name[3:])
        vertices = array(zeros((self.leafCount*8, 3)))
        facets = array(zeros((self.leafCount*6, 4)))
        centroids = array(zeros((self.leafCount, 3)))
        self.root.Render(vertices, facets, centroids)
        
        if self.renderVerticesAndFacets:
            dataset = tvtk.PolyData(points = vertices, polys = facets)
            surf = mlab.pipeline.surface(dataset, opacity = .001)            
            mlab.pipeline.surface(mlab.pipeline.extract_edges(surf), color = (0, 0, 0))
            print 'Rendering took ', clock()-t1clock
        
        if self.renderCentroids:
            scalars = array(ones(self.leafCount)) * 0.01
            pts = mlab.points3d(centroids[:, 0], centroids[:, 1], centroids[:, 2], scalars, scale_factor=.25, color=(1, 0, 0))#, s, color = (1, 0, 0), )        
                
        self.RenderPoints(useColors = True)        
            
if __name__ == "__main__":
    
    pointCloud = PointCloud.PointCloud()
    #pointCloud.ReadPtsFile('.\\Data\Plants\\Geranium\\Geranium2.pts')
    #pointCloud.ReadPtsFile('.\\Data\Shovel\\norgbtest.pts')    
    pointCloud.ReadPlyFile('.\\Data\\bunny\\bunny.ply')    
    octree = Octree(1, 100)
    octree.renderCentroids = True
    octree.renderVerticesAndFacets = True
    octree.renderPoints = True
    
    t1clock = clock()    
    octree.BuildFromPointCloud(pointCloud)        
    print 'It took ', clock()-t1clock, 'to build the octree'    
    octree.Render()    
    
