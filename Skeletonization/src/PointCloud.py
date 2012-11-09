import numpy as np
#import NamedTuples
#from pylab import *
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt

from numpy import array, zeros, ix_ 
from tvtk.api import tvtk
from mayavi.scripts import mayavi2
import mayavi
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi import mlab

class BoundingCube:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius        

class PointCloud:
    
    def __init__(self):
        
        self.backgroundColor = (0., 0., 0.)
        self.xyz = None
        self.rgb = None
        self.intensity = None
        
    def ReadAscFile(self, fileName):
        
        self.fileName = fileName
        
        parametersTypes = np.dtype({'names':['x', 'y', 'z']
                               , 'formats':['float', 'float', 'float']})
        
            
        imported_array = np.genfromtxt(self.fileName, dtype = parametersTypes, filling_values = (0, 0, 0))
    
        self.xyz = imported_array [['x', 'y', 'z']].view(float).reshape(len(imported_array ),-1)                    
        self.boundingCube = None
        
    def ReadPtsFile(self, fileName):
        
        self.fileName = fileName
        
        parametersTypes = np.dtype({'names':['x', 'y', 'z', 'i', 'r','g','b']
                               , 'formats':['float', 'float', 'float', 'int', 'uint8', 'uint8', 'uint8']})
        
#        parametersTypes = np.dtype({'names':['x', 'y', 'z', 'i']
#                               , 'formats':['float', 'float', 'float', 'int']})
              
        imported_array = np.genfromtxt(self.fileName, dtype = parametersTypes, skip_header = 1, filling_values = (0, 0, 0))
        #imported_array = numpy.loadtxt(fileName, skiprows=1)
    
        self.xyz = imported_array [['x', 'y', 'z']].view(float).reshape(len(imported_array ),-1)
        self.intensity = imported_array [['i']].view(int).reshape(len(imported_array ),-1)
        self.intensity_max = self.intensity.max()
        self.intensity_min = self.intensity.min()      
        self.rgb = imported_array [['r', 'g', 'b']].view('uint8').reshape(len(imported_array ),-1)
                
        #self.sphericalCoords = self.cart2Sph(self.xyz, maxRange)                
        self.boundingCube = None
        
    def ReadPlyFile(self, fileName):
        
        self.fileName = fileName
        
        parametersTypes = np.dtype({'names':['x', 'y', 'z', 'c', 'i']
                               , 'formats':['float', 'float', 'float', 'float', 'float']})
              
        imported_array = np.genfromtxt(self.fileName, dtype = parametersTypes)#, skip_header = 1)
        #imported_array = numpy.loadtxt(fileName, skiprows=1)
    
        self.xyz = imported_array [['x', 'y', 'z']].view(float).reshape(len(imported_array ),-1)
        self.intensity = imported_array [['i']].view(float).reshape(len(imported_array ),-1)
        self.confidence = imported_array [['c']].view(float).reshape(len(imported_array ),-1)
        self.intensity_max = self.intensity.max()
        self.intensity_min = self.intensity.min()              
                
        #self.sphericalCoords = self.cart2Sph(self.xyz, maxRange)                
        self.boundingCube = None
        
    def Points(self):
        
        return self.xyz
    
    def RGB(self):
        
        return self.rgb
        
    def BoundingCube(self):
    
        if self.boundingCube == None:
            self.CalcCubicBounds()
        
        return self.boundingCube
    
    def CalcCubicBounds(self):
        
        # What we'll give to the caller        

        # Determine min/max of the given set of points

        minPoint = array(self.xyz[0])
        maxPoint = array(self.xyz[0])        

        nPoints = self.xyz.shape[0]
        for i in range(0, nPoints):            
            if self.xyz[i, 0] < minPoint[0]:
                minPoint[0] = self.xyz[i, 0]
                
            if self.xyz[i, 1] < minPoint[1]:
                minPoint[1] = self.xyz[i, 1]
                
            if self.xyz[i, 2] < minPoint[2]:
                minPoint[2] = self.xyz[i, 2]
                
            if self.xyz[i, 0] > maxPoint[0]:
                maxPoint[0] = self.xyz[i, 0]
                
            if self.xyz[i, 1] > maxPoint[1]:
                maxPoint[1] = self.xyz[i, 1]
                
            if self.xyz[i, 2] > maxPoint[2]:
                maxPoint[2] = self.xyz[i, 2]

        # The radius of the volume (dimensions in each direction)
        print 'maxPoint = ', maxPoint 
        print 'minPoint = ', minPoint
         

        radius = maxPoint - minPoint
        print "radius = ", radius

        # Find the center of this space
        center = minPoint + radius * 0.5
        print "center = ", center
        
        # We want a CUBIC space. By this, I mean we want a bounding cube, not
        # just a bounding box. We already have the center, we just need a
        # radius that contains the entire volume. To do this, we find the
        # maximum value of the radius' X/Y/Z components and use that

        maxRadius = max(radius)
        print "maxRadius = ", maxRadius 
        
        self.boundingCube = BoundingCube(center, maxRadius/2.)
        #self.boundingCube.minPoint = minPoint
        #self.boundingCube.maxPoint = maxPoint
        
        # calculate 8 corners of cube:
        #self.vertices = points = np.empty([8,3])                          
    
    def Center(self):
                
        if self.boundingCube == None:
            self.CalcCubicBounds()
            
        return self.boundingCube.center        
        
    def cart2Sph(self, xyz, maxRange = None):
        '''
        Trasfrom xyz to spherical coordinates
        
        Arguments: 
             - xyz : array of points [x y z] (array)
             - maxRange (optional) : threshold above which all ranges will be set to 999
        Returns: 
            - spherical coordinates array: [rho phi theta]
        
        '''
        ptsnew = np.zeros(xyz.shape)#numpy.hstack((xyz, numpy.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        
        rho = np.sqrt(xy + xyz[:,2]**2) # rho - distance to point
        if maxRange:
            rho[rho > maxRange] = 999 # above range threshold is considered as no-data  
        
        phi = np.arctan2(xyz[:,2], np.sqrt(xy)) # phi - elevation angle
         
        theta = np.arctan2(xyz[:,1], xyz[:,0]) # theta - elevation angle
        theta = np.pi - theta
        theta[theta < 0 ] = 2 * np.pi + theta[theta < 0]

        print np.rad2deg(theta)
        print np.rad2deg(phi)
        
        ptsnew[:,0] = rho
        ptsnew[:,1] = np.rad2deg(phi)
        ptsnew[:,2] = np.rad2deg(theta)
              
        return ptsnew#numpy.hsplit(ptsnew, 2)[1]
       

    def Outline(self):
        
        center = self.boundingCube.center
        radius = self.boundingCube.radius
        corners = array(zeros((9, 3)))        
        corners[0:4, 2] = center[2] - radius
        corners[4:, 2] = center[2] + radius
        corners[::2, 0] = center[0] - radius
        corners[1::2, 0] = center[0] + radius
        corners[ix_([2,3,6,7],[1])] = center[1] - radius
        corners[ix_([0,1,4,5],[1])] = center[1] + radius
        corners[8,:] = center
        points = corners
        squares = array([[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [1,3,7,5], [0,2,6,4]])
        
        mesh = tvtk.PolyData(points = points, polys = squares)
        return mesh
    
    def Cloud(self):
        
        points = tvtk.PolyData(points = array(self.xyz, 'f'))
        if self.rgb != None:
            points.point_data.scalars = self.rgb
            points.point_data.scalars.name = 'rgb'
        #mesh.point_data.scalars = self.intensity
        #mesh.point_data.scalars.name = 'Intensity'
                        
        verts = np.arange(0, self.xyz.shape[0], 1)
        verts.shape = (self.xyz.shape[0], 1)
        points.verts = verts
        
        return points        


    # Now view the data.
    #@mayavi2.standalone   
    @mlab.show    
    def Render(self):
                        
        mlab.figure(bgcolor = (1, 1, 1))        
        
        points = self.Cloud()
        outline = self.Outline()
        
        # Draw outline
        outline_surf = mlab.pipeline.surface(outline, opacity=.2)
        mlab.pipeline.surface(mlab.pipeline.extract_edges(outline_surf), color = (1, 0, 0))
        
        # Draw points
        mlab.pipeline.surface(points)       

if __name__ == "__main__":
        
        plantPoints = PointCloud()
        plantPoints.ReadPtsFile('.\Data\Plants\Geranium\Geranium2.pts')
        #plantPoints.ReadPtsFile('.\Data\Shovel\Site_58_bronze_shovel.pts')
        #plantPoints.ReadPtsFile('.\\Data\Shovel\\norgbtest.pts')                
        plantPoints.backgroundColor = (0., 0., 0.)
        #plantPoints.ReadPlyFile(".\\data\\bunny\\bunny.ply")
            
        plantPoints.CalcCubicBounds()
        plantPoints.Render()