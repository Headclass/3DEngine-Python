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
meshX=0
meshY=0
meshZ=0
size=1
def modelToWorld(vertex,x,y,z):
    global meshX,meshY,meshZ,size
    xangle = math.radians(x)
    yangle = math.radians(y)
    zangle = math.radians(z)
    xRotationMatrix = numpy.array([[1, 0, 0, 0], [0, math.cos(xangle), -math.sin(xangle), 0], [0, math.sin(xangle), math.cos(xangle), 0],[0, 0, 0, 1]])
    yRotationMatrix = numpy.array([[math.cos(yangle), 0, math.sin(yangle), 0], [0, 1, 0, 0], [-math.sin(yangle), 0, math.cos(yangle), 0],[0, 0, 0, 1]])
    zRotationMatrix = numpy.array([[math.cos(zangle), -math.sin(zangle), 0, 0], [math.sin(zangle), math.cos(zangle), 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
    TranslationMatrix = numpy.array([[1, 0, 0, meshX], [0, 1, 0, meshY], [0, 0, 1, meshZ], [0, 0, 0, 1]])
    SizeMatrix = numpy.array([[size, 0, 0, 0], [0, size, 0, 0], [0, 0, size, 0], [0, 0, 0, 1]])
    ModelMatrix = numpy.dot(xRotationMatrix, SizeMatrix)
    ModelMatrix = numpy.dot(yRotationMatrix, ModelMatrix)
    ModelMatrix = numpy.dot(zRotationMatrix, ModelMatrix)
    ModelMatrix = numpy.dot(TranslationMatrix, ModelMatrix)
    return numpy.dot(ModelMatrix,vertex)


#Global variables for the camera movement
xcam, ycam, zcam = 0, 0, 0
camXangle, camYangle, camZangle = 0, 0, 0
ViewMatrix = numpy.array([[1, 0, 0, 0],
                          [0, 1, 0, -1],
                          [0, 0, 1, -4],
                          [0, 0, 0, 1]])

test_vector=numpy.array([0,0,0,1]).T

curCamX=0
curCamY=1
curCamZ=4
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

        pyxel.text(5, 5, "Location", 15)
        pyxel.text(5, 15,"X:", 15)
        pyxel.text(5, 25, "Y:", 15)
        pyxel.text(5, 35, "Z:", 15)
        pyxel.text(15, 15,str(numpy.dot(numpy.linalg.inv(ViewMatrix),test_vector)[0])[:4], 1)
        pyxel.text(15, 25, str(numpy.dot(numpy.linalg.inv(ViewMatrix), test_vector)[1])[:4], 2)
        pyxel.text(15, 35, str(numpy.dot(numpy.linalg.inv(ViewMatrix), test_vector)[2])[:4], 3)

        pyxel.text(55, 5, "Direction", 15)
        pyxel.text(55, 15,"X:", 15)
        pyxel.text(55, 25, "Y:", 15)
        pyxel.text(55, 35, "Z:", 15)
        pyxel.text(65, 15,str(numpy.dot(numpy.linalg.inv(ViewMatrix),numpy.array([0,0,-1,0]).T)[0])[:4], 1)
        pyxel.text(65, 25, str(numpy.dot(numpy.linalg.inv(ViewMatrix), numpy.array([0,0,-1,0]).T)[1])[:4], 2)
        pyxel.text(65, 35, str(numpy.dot(numpy.linalg.inv(ViewMatrix), numpy.array([0,0,-1,0]).T)[2])[:4], 3)

#Key bindings
def quit():
    if pyxel.btnp(pyxel.KEY_T):
        pyxel.quit()
    global xcam,ycam,zcam,camXangle,camYangle,camZangle,index,mode,turnedoff,curCamX,curCamY,curCamZ,mesh,meshX,meshY,meshZ,size

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
    if pyxel.btn(pyxel.KEY_KP_4):
        meshX +=0.05
    if pyxel.btn(pyxel.KEY_KP_6):
        meshX -=0.05
    if pyxel.btn(pyxel.KEY_KP_8):
        meshZ +=0.05
    if pyxel.btn(pyxel.KEY_KP_5):
        meshZ -=0.05
    if pyxel.btn(pyxel.KEY_KP_7):
        meshY +=0.05
    if pyxel.btn(pyxel.KEY_KP_9):
        meshY -=0.05
    if pyxel.btn(pyxel.KEY_KP_3):
        size +=0.05
    if pyxel.btn(pyxel.KEY_KP_1):
        size -=0.05

pyxel.init(width=width, height=height,fps=fps,caption="3D Engine",palette=[0x000000,0xFF0000, 0x00FF00, 0x0000FF, 0x333333, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xAAAAAA, 0xBBBBBB, 0xCCCCCC, 0xDDDDDD, 0xEEEEEE, 0xFFFFFF])

pyxel.run(update, quit)

