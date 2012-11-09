#http://docs.enthought.com/mayavi/mayavi/data.html
from numpy import array, random, linspace, pi, ravel, cos, sin, empty, zeros, ix_, arange, ones
from tvtk.api import tvtk
import PointCloud
from random import randint, random
from time import time, clock
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi import mlab

import logging
from chaco.shell.commands import figure
logging.basicConfig()


def PrintMatrix(M):
    for l in M:
        print l

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
        self.color = (0., 0., 0.)
        self.parent = None
        self.level = -1
        self.neighbors = []        
        self.neighborsDirections = []
        self.octree = None
        self.index = -1

        self.outline_rendered = False
                 
#       self.color = (randint(0,255), randint(0, 255), randint(0, 255))
        #self.color = (random(), random(), random())
        #print "self.color = ", self.color 
        
        self.children = [None, None, None, None, None, None, None, None]
        self.boundsOffsetTable = array([[-0.5, -0.5, -0.5],
                                        [+0.5, -0.5, -0.5],
                                        [-0.5, +0.5, -0.5],
                                        [+0.5, +0.5, -0.5],
                                        [-0.5, -0.5, +0.5],
                                        [+0.5, -0.5, +0.5],
                                        [-0.5, +0.5, +0.5],
                                        [+0.5, +0.5, +0.5]])        
    
    def Build(self, points, currentDepth, octree):
                
        octree.nodesPerLevel[currentDepth-1] += 1
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
            octree.leafCount += 1
            octree.leaves.append(self)
            #self.octree = octree
            #self.depth = 
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
            
            if childPointsCount[i] == 0:                
                continue
            
            center = self.center + self.boundsOffsetTable[i] * self.radius
            radius = self.radius/2.
            
            self.children[i] = OctreeNode(center, radius, self.threshold, self.maxDepth)            
            childPoints = points[ix_(childPointsIndices[i], )]            
            self.children[i].Build(childPoints, currentDepth + 1, octree)
            self.children[i].code = i            
            self.children[i].parent = self            
            self.children[i].color = octree.colors[i]        
    
    
    def UpdateNeighbors(self):
#        
#        # Neighbors of same size and same parent
#        neighbors = [[], [0], [0], [1, 2], [0], [1, 4], [2, 4], [3, 5, 6]]
#        neighborsDirectionsItoJ = [[], [(-1, 0, 0)], [(0, -1, 0)], [(0, -1, 0), (-1, 0, 0)], [(0, 0, -1)], 
#                                   [(0, 0, -1), (-1, 0, 0)], [(0, 0, -1), (0, -1, 0)], [(0, 0, -1), (0, -1, 0), (-1, 0, 0)]]
#        neighborsDirectionsJtoI = [[], [( 1, 0, 0)], [(0,  1, 0)], [(0,  1, 0), ( 1, 0, 0)], [(0, 0,  1)], 
#                                   [(0, 0,  1), ( 1, 0, 0)], [(0, 0,  1), (0,  1, 0)], [(0, 0,  1), (0,  1, 0), ( 1, 0, 0)]]
#        
        code = self.code
        parent = self.parent
#        
#        n = 0 
#        for j in neighbors[code]:
#            if parent.children[j] != None and parent.children[j].isLeaf:                
#                self.neighbors.append(parent.children[j].index)
#                self.neighborsDirections.append(neighborsDirectionsItoJ[code][n])                                        
#                parent.children[j].neighbors.append(self.index)
#                parent.children[j].neighborsDirections.append(neighborsDirectionsJtoI[code][n])
#                n += 1
                
        # Other neighbors        
        for i in range(0, self.index):             
            leaf = octree.leaves[i]
            if (self.index == 42) and (leaf.index == 38):
                k = 5
            selfToLeafDirection = array(zeros(3))            
            isNeighbor, selfToLeafDirection = self.IsNeighbor(leaf)
            if isNeighbor:
                self.neighbors.append(leaf.index)
                self.neighborsDirections.append(selfToLeafDirection)                                        
                leaf.neighbors.append(parent.children[code].index)
                leafToSelfDirection = array(selfToLeafDirection * -1)
                leaf.neighborsDirections.append(leafToSelfDirection)
    
    def IsNeighbor(self, leaf):
        
        directions = array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])        
        smaller = None
        if leaf.radius < self.radius:
            smaller = leaf
            bigger = self
        else:
            smaller = self
            bigger = leaf  
        
        # Go over six possible neighbors of small cube. If one of them is inside the big cube, big and small cube are neighbors (thank you Sivan Yogev)
        
        # Creating six neighbors
        for i in range(0, 6):
            shift = smaller.center + directions[i]*smaller.radius*2
            if bigger.ContainsPoint(shift):                
                direction = directions[i]
                return True, direction# Two cells can be neighbors only once...
        
        return False, None
    
    # Determine if point is contained in octree node bounding box
    def ContainsPoint(self, point):
        
        for i in range(0, 3):
            if (point[i] > (self.center[i] + self.radius)) or (point[i] < (self.center[i] - self.radius)):                
                return False
        
        return True 
        
    def UpdateNeighbors_old(self, code):
              
        if self.children[code] == None:
            return                      
        # Trivial (inside cube) neighbors
        neighbors = [[], [0], [0], [1, 2], [0], [1, 4], [2, 4], [3, 5, 6]]
        neighborsDirectionsItoJ = [[], [(-1, 0, 0)], [(0, -1, 0)], [(0, -1, 0), (-1, 0, 0)], [(0, 0, -1)], 
                                   [(0, 0, -1), (-1, 0, 0)], [(0, 0, -1), (0, -1, 0)], [(0, 0, -1), (0, -1, 0), (-1, 0, 0)]]
        neighborsDirectionsJtoI = [[], [( 1, 0, 0)], [(0,  1, 0)], [(0,  1, 0), ( 1, 0, 0)], [(0, 0,  1)], 
                                   [(0, 0,  1), ( 1, 0, 0)], [(0, 0,  1), (0,  1, 0)], [(0, 0,  1), (0,  1, 0), ( 1, 0, 0)]]

        n = 0 
        for j in neighbors[code]:
            if self.children[j] != None:
                self.children[code].neighbors.append(self.children[j].index)
                self.children[code].neighborsDirections.append(neighborsDirectionsItoJ[code][n])                                        
                self.children[j].neighbors.append(self.children[code].index)
                self.children[j].neighborsDirections.append(neighborsDirectionsJtoI[code][n])
                n += 1
                
        #leaf = OctreeNode()
        # Inheriting neighbors from daddy-o
#        directions = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
#        for leaf in self.neighbors:
#            # Go over six possible neighbors of small cube. If one of them is inside the big cube, big and small cube are neighbors (thank you Sivan Yogev)
#            if leaf.radius < self.children[code].radius:
#                # Creating six neighbors
#                for i in range(0, 6):
#                    center = 
                    
    
    def Centroid(self): 
        # Calculate the centroid of the cell
        self.centroid = zeros(3)  
        self.centroid[0] = sum(self.points[:, 0]/self.nPoints) 
        self.centroid[1] = sum(self.points[:, 1]/self.nPoints)
        self.centroid[2] = sum(self.points[:, 2]/self.nPoints)
            
    def InNeighbor(self, otherNode):
        
        return True
            
    def Render(self, renderToLevel, vertices, facets, centroids, render_points):
        
        if renderToLevel == -1:            
            if self.isLeaf:
                #print 'Number of points in leaf:', self.nPoints             
                # aggregate vertices and facets
                vertices[self.index * 8:self.index * 8 + 8] = self.vertices
                facets[self.index * 6:self.index * 6 + 6] = self.facets                                                
                centroids[self.index] = self.centroid        
                if render_points:
                    self.RenderPoints()
                return
             
            else:        
                for child in self.children:
                    if child != None:
                        child.Render(-1, vertices, facets, centroids, render_points)                        
        
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
    
    def RenderPoints(self):
                                
        cloud = self.Cloud()
        #mlab.figure(bgcolor = (1, 1, 1))
        mlab.pipeline.surface(cloud, color = self.color)
        
    def Outline(self):
        
        corners = array(zeros((9, 3)))
        # X
        corners[0:4, 2] = self.center[2] - self.radius
        corners[4:, 2] = self.center[2] + self.radius
        # Y
        corners[::2, 0] = self.center[0] - self.radius
        corners[1::2, 0] = self.center[0] + self.radius
        # Z
        corners[ix_([2,3,6,7],[1])] = self.center[1] - self.radius
        corners[ix_([0,1,4,5],[1])] = self.center[1] + self.radius                
        squares = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])        
        mesh = tvtk.PolyData(points = corners, polys = squares)
        return mesh
    
    def OutlineVerticesAndFacets(self):
        
        self.vertices= array(zeros((8, 3)))
        self.vertices[0:4, 2] = self.center[2] - self.radius
        self.vertices[4:, 2] = self.center[2] + self.radius
        self.vertices[::2, 0] = self.center[0] - self.radius
        self.vertices[1::2, 0] = self.center[0] + self.radius
        self.vertices[ix_([2,3,6,7],[1])] = self.center[1] - self.radius
        self.vertices[ix_([0,1,4,5],[1])] = self.center[1] + self.radius                
        self.facets = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])        
        self.facets += self.index * 8
'''
Octree of 3D points
'''
class Octree:
    
    def __init__(self, maxDepth, threshold):


        logging.basicConfig()        
        self.maxDepth = maxDepth
        self.threshold = threshold        
        self.nPoints = 0        
        self.leaves = []        
        #self.colors = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])#[(0., 1., 0.), (0., 0., 1.), (1., 0., 0.), (0., 0.5, 0.), (0., 1., 1.), (1., 1., 0.), (1., 0., 1.), (0.5, 0., 1.)]
        self.colors = [(0., 1., 0.), (0., 0., 1.), (1., 0., 0.), (0., 0.5, 0.), (0., 1., 1.), (1., 1., 0.), (1., 0., 1.), (0.5, 0., 1.)]
        self.trivialNeighbors = [[2, 3, 5], [1, 4, 6], [1, 4, 7], [2, 3, 8], [6, 7, 1], [5, 8, 2], [5, 8, 3], [6, 7, 4]]
        self.octreeGraphEdges = None
        self.octreeGraphVertices = None
        
        self.renderPoints = False    
        self.renderVerticesAndFacets = False        
        self.renderCentroids = False
        self.renderToLevel = -1
        self.nodesPerLevel = zeros(maxDepth)
        
        self.centroid_glyphs = None
        
        self.glyph_points = None
        self.ce = 0
        self.figure = None
        
    def CreateOctreeGraph(self):

        for leaf in self.leaves:                        
            leaf.UpdateNeighbors()
                
        self.octreeGraphVertices = array(zeros((self.leafCount, 3)))
        self.octreeGraphEdges = []
        for leaf in self.leaves:
            self.octreeGraphVertices[leaf.index, :] = leaf.centroid
            for n in leaf.neighbors:
                self.octreeGraphEdges.append([leaf.index, n])
        
        i = 7
        print i
                
    def RenderOctreeGraph(self):
        
        if self.octreeGraphEdges == None or self.octreeGraphVertices == None:
            return

        if len(self.octreeGraphVertices) < 1:
            return

        scalars = array(ones(self.leafCount)) * 0.05
        mlab.points3d(self.octreeGraphVertices[:, 0], self.octreeGraphVertices[:, 1], self.octreeGraphVertices[:, 2], scalars, scale_factor=.05, color = (1, 0, 0))#, s, color = (1, 0, 0), )
        print 'octreeGraphVertices'
#        print PrintMatrix(self.octreeGraphVertices)
        edges = array(self.octreeGraphEdges)
#        print edges
#        print 'len(self.octreeGraphVertices) = ', len(self.octreeGraphVertices)
#        print 'len 2= ', len(self.octreeGraphEdges)
#        #print array(self.octreeGraphEdges)
        dataset = tvtk.PolyData(points = self.octreeGraphVertices, polys = edges)
        surf = mlab.pipeline.surface(dataset, opacity = .001)            
        mlab.pipeline.surface(mlab.pipeline.extract_edges(surf), color = (1., 0., 0.))                
                 
    def Build(self, points):
                
        self.nPoints = points.shape[0]
        self.root = OctreeNode(points)
        
    def BuildFromPointCloud(self, pointCloud):
        
        self.nPoints = pointCloud.Points().shape[0]
        self.root = OctreeNode(pointCloud.BoundingCube().center, pointCloud.BoundingCube().radius, self.threshold, self.maxDepth)
        self.leafCount = 0
        self.root.Build(pointCloud.Points(), 0, self)
        self.points = pointCloud.Points()

    def RenderPoints(self):
                
        if not self.renderPoints:
            return         
        cloud = tvtk.PolyData(points = array(self.points, 'f'))                        
        verts = arange(0, self.points.shape[0], 1)
        verts.shape = (self.points.shape[0], 1)
        cloud.verts = verts        
        
#        cloud.point_data.scalars = self.
#        cloud.point_data.scalars.name = 'rgb'
        
        mlab.pipeline.surface(cloud, color = (0, 0, 0))            
            
    @mlab.show
    def Render(self):
        
        t1clock = clock()                    
        self.figure = mlab.figure(bgcolor = (1, 1, 1), fgcolor = (0, 0, 0))#, figure = dataset.class_name[3:])
        self.figure.scene.disable_render = True
        nodesToRender = self.leafCount
        if self.renderToLevel != -1:
            nodesToRender = 0
            for i in range(0, self.renderToLevel):
                nodesToRender += self.nodesPerLevel[i]
            
        vertices = array(zeros((nodesToRender*8, 3)))
        facets = array(zeros((nodesToRender *6, 4)))
        self.centroids = array(zeros((nodesToRender , 3)))                
        self.root.Render(self.renderToLevel, vertices, facets, self.centroids, render_points = False)
        
        
        # Print vertices
        # Print facets
        if self.renderVerticesAndFacets:
            dataset = tvtk.PolyData(points = vertices, polys = facets)
            surf = mlab.pipeline.surface(dataset, opacity = .001)            
            mlab.pipeline.surface(mlab.pipeline.extract_edges(surf), color = (0, 0, 0))
            
            #print 'len 111 = ', len(vertices)
            #print 'len 222 = ', len(facets)
            #print 'Rendering took ', clock() - t1clock
                    
        if self.renderCentroids:
            print 'self.leafCount = ', self.leafCount
            scalars = array(ones(self.leafCount)) * 0.05
            self.centroid_glyphs =  mlab.points3d(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], scalars, scale_factor=.05, color = (1, 0, 0))#, s, color = (1, 0, 0), )
            
#        outline = mlab.outline(name = str('1'), line_width = 3, color = (0, 1, 0))
#        outline.outline_mode = 'full'        
#        center = self.leaves[0].center
#        radius = self.leaves[0].radius
#        print 'center = ', center
#        print 'radius= ', radius
#        outline.bounds = (center[0] - radius, center[0] + radius,
#                          center[1] - radius, center[1] + radius,
#                          center[2] - radius, center[2] + radius)        

        # temp - rendering points                
        self.RenderPoints()
        
        self.RenderOctreeGraph()   
                        
        mlab.title('Click on centroid')
                        
        picker = self.figure.on_mouse_pick(self.picker_callback)
        # Decrease the tolerance, so that we can more easily select a precise
        # point.
        picker.tolerance = 0.01
        
        self.figure.scene.disable_render = False
        
        # Here, we grab the points describing the individual glyph, to figure
        # out how many points are in an individual glyph.
        self.glyph_points = self.centroid_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()
        
        
    def picker_callback(self, picker):
        """ Picker callback: this get called when on pick events.
        """
        if self.centroid_glyphs == None:
            return 

        if (picker.actor in self.centroid_glyphs.actor.actors) and (self.centroid_glyphs != None):
            # Find which data point corresponds to the point picked:
            # we have to account for the fact that each data point is
            # represented by a glyph with several points
            point_id = picker.point_id/self.glyph_points.shape[0]
            
            print 'point_id =', point_id
             
            # If the no points have been selected, we have '-1'
            if point_id != -1:
                
                current_leaf = self.leaves[point_id]
                center = current_leaf .center
                radius = current_leaf .radius
                
                
                  
                if current_leaf.outline_rendered:
                    # remove outline
                    self.figure.remove_traits(str(point_id))
                else:     
                    # Retrieve the coordinnates coorresponding to that data
                    # point
                    x, y, z = self.centroids[point_id, 0], self.centroids[point_id, 1], self.centroids[point_id, 2]
                    print x, y, z 
                    # Move the outline to the data point.
    #                self.outline.bounds = (x - 0.0001, x + 0.0001,
    #                                       y - 0.0001, y + 0.0001,
    #                                       z - 0.0001, z + 0.0001)
    #                self.outline.bounds = (self.centroids[self.ce, 0] - 0.01, self.centroids[self.ce, 0] + 0.01,
    #                                       self.centroids[self.ce, 1] - 0.01, self.centroids[self.ce, 1] + 0.01,
    #                                       self.centroids[self.ce, 2] - 0.01, self.centroids[self.ce, 2] + 0.01)
    
                    
                    outline = mlab.outline(name = str(point_id), line_width = 3, color = (0, 1, 0))                
                    
                    outline.outline_mode = 'full'                
                    print 'center = ', center
                    print 'radius= ', radius
                    outline.bounds = (center[0] - radius, center[0] + radius,
                                      center[1] - radius, center[1] + radius,
                                      center[2] - radius, center[2] + radius)
                    print 'self.outline.bounds = ', outline.bounds

                current_leaf.outline_rendered = not current_leaf.outline_rendered
                 

            
if __name__ == "__main__":

    
    pointCloud = PointCloud.PointCloud()
#    pointCloud.ReadPtsFile('.\\Data\Plants\\Geranium\\Geranium2.pts')
#    pointCloud.ReadPtsFile('.\\Data\Shovel\\norgbtest.pts')    
#    pointCloud.ReadPlyFile('.\\Data\\bunny\\bunny.ply')
    pointCloud.ReadAscFile('.\\Data\\Giraffe\\GiraffeTail.pts')    
    octree = Octree(5, 0)
    octree.renderPoints = False
    octree.renderVerticesAndFacets = False 
    octree.renderCentroids = True
    octree.renderToLevel = -1
    
    t1clock = clock()    
    octree.BuildFromPointCloud(pointCloud)        
    print 'It took ', clock()-t1clock, 'to build the octree'
    octree.CreateOctreeGraph()    
    octree.Render()
    
