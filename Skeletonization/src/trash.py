class BoundingCube:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius        
import vtk
from numpy import random
from mayavi.scripts import mayavi2
import mayavi

class BoundingCube:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius        

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def AddPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def ClearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        
    def Read 


pointCloud = VtkPointCloud()

#for k in xrange(1000):
 #   point = 20*(random.rand(3)-0.5)
    #pointCloud.addPoint(point)
    

#pointCloud.addPoint([0,0,0])
#pointCloud.addPoint([0,0,0])
#pointCloud.addPoint([0,0,0])
#pointCloud.addPoint([0,0,0])

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(pointCloud.vtkActor)
renderer.SetBackground(.2, .3, .4)
renderer.ResetCamera()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
renderWindow.Render()
renderWindowInteractor.Start()


        #from enthought.mayavi.modules.outline import Outline
        #from mayavi.modules.surface import Surface        
#        mayavi.new_scene() 
#        d = VTKDataSource()
#        d.data = mesh
#        mayavi.add_source(d)
#        #mayavi.add_module(Outline())
#        s = Surface()
#        mayavi.add_module(s)
#        s.actor.property.set(representation = 'p', point_size = 2)
#        # You could also use glyphs to render the points via the Glyph module.
              
#        Draw outline of point cloud             
#        Draw Center
#        s = [0.1]
#        center = self.boundingCube.center
#        print center[0]
#        pts = mlab.points3d([center[0]], [center[1]], [center[2]], s, color = (1, 0, 0), )
#        pts.glyph.glyph.clamping = False
                                
                                
                                #, fgcolor = (0, 0, 0))#, figure = outline.class_name[3:])