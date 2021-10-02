#!/usr/bin/python
import numpy
import math
import pyxel

#Window dimensions and framerate
width = 255
height = 255
fps=60

#Vertices of the object
CubeMesh=[
#Cube
[-0.5, 0., 0.5 ],[0.5, 0., 0.5 ],[0.5, 1., 0.5 ],
[-0.5, 0., 0.5 ],[0.5, 1., 0.5 ],[-0.5, 1., 0.5 ],
[0.5, 0., 0.5 ],[0.5, 0., -0.5 ],[0.5, 1., -0.5 ],
[0.5, 0., 0.5 ],[0.5, 1., -0.5 ],[0.5, 1., 0.5 ],
[0.5, 0., -0.5 ],[-0.5, 0., -0.5 ],[0.5, 1., -0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 1., -0.5 ],[0.5, 1., -0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 0., 0.5 ],[-0.5, 1., 0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 1., 0.5 ],[-0.5, 1., -0.5 ],
[-0.5, 1., 0.5 ],[0.5, 1., 0.5 ],[0.5, 1., -0.5 ],
[-0.5, 1., 0.5 ],[0.5, 1., -0.5 ],[-0.5, 1., -0.5 ],
[-0.5, 0., 0.5 ],[-0.5, 0., -0.5 ],[0.5, 0., -0.5 ],
[-0.5, 0., 0.5 ],[0.5, 0., -0.5 ],[0.5, 0., 0.5 ],
]


#Adding a homogenous coordinate (w=1) to all vertices
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertices to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T

#Adjusting object vertices
for i in range(len(CubeMesh)):
    homogenous(CubeMesh[i])               # Adding a homogenous coordinate
    CubeMesh[i] = transpose(CubeMesh[i])  # Changing a row vector to a column vector


#Transforming the object from local space to world space
def modelToWorld(vertex,x,y,z):
    global meshX,meshY,meshZ,size
    TranslationMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    return numpy.dot(TranslationMatrix,vertex)


#Global variables for the camera movement
xcam, ycam, zcam = 0, 0, 0
camXangle, camYangle, camZangle = 0, 0, 0
ViewMatrix = numpy.array([[1, 0, 0, 0],
                          [0, 1, 0, -1],
                          [0, 0, 1, -4],
                          [0, 0, 0, 1]])

#Updating the current camera matrix by keyboard inputs
def updateView():
    global ViewMatrix, xcam, ycam, zcam, camXangle, camYangle, camZangle
    CamyRotationMatrix = numpy.array([[math.cos(math.radians(camYangle)), 0, math.sin(math.radians(camYangle)), 0], [0, 1, 0, 0],[-math.sin(math.radians(camYangle)), 0, math.cos(math.radians(camYangle)), 0],[0, 0, 0, 1]])
    CamxRotationMatrix = numpy.array([[1, 0, 0, 0], [0, math.cos(math.radians(camXangle)), -math.sin(math.radians(camXangle)), 0],[0, math.sin(math.radians(camXangle)), math.cos(math.radians(camXangle)), 0],[0, 0, 0, 1]])
    CamzRotationMatrix = numpy.array([[math.cos(math.radians(camZangle)), -math.sin(math.radians(camZangle)), 0, 0], [math.sin(math.radians(camZangle)), math.cos(math.radians(camZangle)), 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
    CamTranslationMatrix = numpy.array([[1, 0, 0, 0 + xcam], [0, 1, 0, 0 + ycam], [0, 0, 1, 0 + zcam], [0, 0, 0, 1]])
    ViewMatrix = numpy.dot(CamxRotationMatrix, ViewMatrix)
    ViewMatrix = numpy.dot(CamyRotationMatrix, ViewMatrix)
    ViewMatrix = numpy.dot(CamzRotationMatrix, ViewMatrix)
    ViewMatrix = numpy.dot(CamTranslationMatrix, ViewMatrix)
    xcam, ycam, zcam = 0, 0, 0
    camXangle, camYangle, camZangle = 0, 0, 0

ProjectionMatrix = numpy.array([[1.2,0,0,0], [0,1.2,0,0],[0,0,-1.04,-0.41],[0,0,-1,0]])
#ProjectionMatrix = numpy.array([[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,-1,0]])
#Transforming from world space to view space
def worldToView(vertex):
    global ModelView
    ModelView = numpy.dot(ViewMatrix,vertex)
    return ModelView

#Transforming from view space to clip space
def viewToClip(vertex):
    return numpy.dot(ProjectionMatrix,vertex)

#Dividing by W
def perspectiveDivision(vertex):
    for j in range(4):
        if vertex[3] < 0.01 and -0.01<vertex[3]:
            vertex[3] = 0.01
        vertex[j]=vertex[j]/vertex[3]
    return vertex

#Transforming to window size
def viewportTransformation(vertex):
    vertex[0] = (vertex[0] * 0.5 + 0.5) * width
    vertex[1] = (vertex[1] * 0.5 + 0.5) * height
    return vertex

#Rounding
def roundPixel(vertex):
    vertex[0]=  round(float(vertex[0][0]))
    vertex[1] = round(float(vertex[1][0]))
    return vertex

def rasterize(triangle):
    pyxel.line(triangle[0][0][0],height-triangle[0][1][0],triangle[1][0][0],height-triangle[1][1][0],15)
    pyxel.line(triangle[1][0][0],height-triangle[1][1][0],triangle[2][0][0],height-triangle[2][1][0],15)
    pyxel.line(triangle[2][0][0],height-triangle[2][1][0],triangle[0][0][0],height-triangle[0][1][0],15)

def TransformTriangle(Triangle):
    for i in range(3):
            Triangle[i] = modelToWorld(Triangle[i],0,0,0)
            Triangle[i] = worldToView(Triangle[i])  # Moving the world relative to our camera ("moving" the camera)
            Triangle[i] = viewToClip(Triangle[i])  # Applying projection
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
    rasterize(Triangle)

def update():
        global CubeMesh
        # An empty triangle
        Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Fill the screen with black color (clear it)
        pyxel.cls(0)
        # Update the view matrix according to the current camera rotation and position
        updateView()
        for i in range(len(CubeMesh)):
            Triangle[i % 3][0] = (CubeMesh[i][0])
            Triangle[i % 3][1] = (CubeMesh[i][1])
            Triangle[i % 3][2] = (CubeMesh[i][2])
            Triangle[i % 3][3] = (CubeMesh[i][3])
            if i % 3 == 2:
                TransformTriangle(Triangle)
#Key bindings
def quit():
    global xcam,ycam,zcam,camXangle,camYangle,camZangle
    if pyxel.btnp(pyxel.KEY_T):
        pyxel.quit()
    if pyxel.btn(pyxel.KEY_W):
        zcam+=0.05
    if pyxel.btn(pyxel.KEY_S):
        zcam-=0.05
    if pyxel.btn(pyxel.KEY_A):
        xcam+=0.05
    if pyxel.btn(pyxel.KEY_D):
        xcam-=0.05
    if pyxel.btn(pyxel.KEY_Q):
        ycam-=0.05
    if pyxel.btn(pyxel.KEY_E):
        ycam+=0.05
    if pyxel.btn(pyxel.KEY_I):
        camXangle -= 2
    if pyxel.btn(pyxel.KEY_K):
        camXangle += 2
    if pyxel.btn(pyxel.KEY_J):
        camYangle -= 2
    if pyxel.btn(pyxel.KEY_L):
        camYangle += 2
    if pyxel.btn(pyxel.KEY_U):
        camZangle -= 2
    if pyxel.btn(pyxel.KEY_O):
        camZangle +=2

pyxel.init(width=width, height=height,fps=fps,caption="3D Engine",palette=[0x000000,0xFF0000, 0x00FF00, 0x0000FF, 0x333333, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xAAAAAA, 0xBBBBBB, 0xCCCCCC, 0xDDDDDD, 0xEEEEEE, 0xFFFFFF])

pyxel.run(update, quit)

