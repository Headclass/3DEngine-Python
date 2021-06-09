#!/usr/bin/python
import numpy
import math
import pyxel

#Window dimensions and framerate
width = 255
height = 255
fps=60

#Vertices of the object
House=[
#Ground
[-2, 0, -2],[2, 0, -2],[-2, 0, 2],[-2, 0, 2],[2, 0, -2],[2, 0, 2],
#House
[-0.5, 0., 0.5 ],[0.5, 0., 0.5 ],[0.5, 1., 0.5 ],
[-0.5, 0., 0.5 ],[0.5, 1., 0.5 ],[-0.5, 1., 0.5 ],
[0.5, 0., 0.5 ],[0.5, 0., -0.5 ],[0.5, 1., -0.5 ],
[0.5, 0., 0.5 ],[0.5, 1., -0.5 ],[0.5, 1., 0.5 ],
[0.5, 0., -0.5 ],[-0.5, 0., -0.5 ],[0.5, 1., -0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 1., -0.5 ],[0.5, 1., -0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 0., 0.5 ],[-0.5, 1., 0.5 ],
[-0.5, 0., -0.5 ],[-0.5, 1., 0.5 ],[-0.5, 1., -0.5 ],
[-0.5, 1., 0.5 ],[0.5, 1., 0.5 ],[0, 2., 0 ],
[0.5, 1., 0.5 ],[0.5, 1., -0.5 ],[0, 2., 0 ],
[0.5, 1., -0.5 ],[-0.5, 1., -0.5 ],[0, 2., 0 ],
[-0.5, 1., -0.5 ],[-0.5, 1., 0.5 ],[0, 2., 0 ]
]

icosahedron=[[0.850651, 0.0, 0.525731], [0.850651, 0.0, -0.525731], [0.525731, 0.850651, 0.0], [0.850651, 0.0, 0.525731], [0.525731, -0.850651, 0.0], [0.850651, 0.0, -0.525731], [-0.850651, 0.0, -0.525731], [-0.850651, 0.0, 0.525731], [-0.525731, 0.850651, 0.0], [-0.850651, 0.0, 0.525731], [-0.850651, 0.0, -0.525731], [-0.525731, -0.850651, 0.0], [0.525731, 0.850651, 0.0], [-0.525731, 0.850651, 0.0], [0.0, 0.525731, 0.850651], [-0.525731, 0.850651, 0.0], [0.525731, 0.850651, 0.0], [0.0, 0.525731, -0.850651], [0.0, -0.525731, -0.850651], [0.0, 0.525731, -0.850651], [0.850651, 0.0, -0.525731], [0.0, 0.525731, -0.850651], [0.0, -0.525731, -0.850651], [-0.850651, 0.0, -0.525731], [0.525731, -0.850651, 0.0], [-0.525731, -0.850651, 0.0], [0.0, -0.525731, -0.850651], [-0.525731, -0.850651, 0.0], [0.525731, -0.850651, 0.0], [0.0, -0.525731, 0.850651], [0.0, 0.525731, 0.850651], [0.0, -0.525731, 0.850651], [0.850651, 0.0, 0.525731], [0.0, -0.525731, 0.850651], [0.0, 0.525731, 0.850651], [-0.850651, 0.0, 0.525731], [0.525731, 0.850651, 0.0], [0.850651, 0.0, -0.525731], [0.0, 0.525731, -0.850651], [0.850651, 0.0, 0.525731], [0.525731, 0.850651, 0.0], [0.0, 0.525731, 0.850651], [-0.850651, 0.0, -0.525731], [-0.525731, 0.850651, 0.0], [0.0, 0.525731, -0.850651], [-0.525731, 0.850651, 0.0], [-0.850651, 0.0, 0.525731], [0.0, 0.525731, 0.850651], [0.850651, 0.0, -0.525731], [0.525731, -0.850651, 0.0], [0.0, -0.525731, -0.850651], [0.525731, -0.850651, 0.0], [0.850651, 0.0, 0.525731], [0.0, -0.525731, 0.850651], [-0.850651, 0.0, -0.525731], [0.0, -0.525731, -0.850651], [-0.525731, -0.850651, 0.0], [-0.850651, 0.0, 0.525731], [-0.525731, -0.850651, 0.0], [0.0, -0.525731, 0.850651]]


cubeMesh=House

#Axes (x,y,z)
axes = [[0, 0, 0],[0.7, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0.7, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0.7],[0, 0, 0]]


#Adding a homogenous coordinate (w=1) to all vertices
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertices to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T

#Adjusting object vertices
for i in range(len(House)):
    homogenous(House[i])               # Adding a homogenous coordinate
    House[i] = transpose(House[i])  # Changing a row vector to a column vector

#Adjusting object vertices
for i in range(len(icosahedron)):
    homogenous(icosahedron[i])               # Adding a homogenous coordinate
    icosahedron[i] = transpose(icosahedron[i])  # Changing a row vector to a column vector

#Adjusting axes vertices
for i in range(len(axes)):
    homogenous(axes[i])                   # Adding a homogenous coordinate
    axes[i] = transpose(axes[i])          # Changing a row vector to a column vector

def meshChange():
    global cubeMesh
    if mesh==0:
        cubeMesh=House
    else:
        cubeMesh=icosahedron

#Transforming the object from local space to world space
def modelToWorld(vertex,x,y,z):
    xangle = math.radians(x)
    yangle = math.radians(y)
    zangle = math.radians(z)
    xRotationMatrix = numpy.array([[1, 0, 0, 0], [0, math.cos(xangle), -math.sin(xangle), 0], [0, math.sin(xangle), math.cos(xangle), 0],[0, 0, 0, 1]])
    yRotationMatrix = numpy.array([[math.cos(yangle), 0, math.sin(yangle), 0], [0, 1, 0, 0], [-math.sin(yangle), 0, math.cos(yangle), 0],[0, 0, 0, 1]])
    zRotationMatrix = numpy.array([[math.cos(zangle), -math.sin(zangle), 0, 0], [math.sin(zangle), math.cos(zangle), 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
    TranslationMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ModelMatrix = numpy.dot(yRotationMatrix, xRotationMatrix)
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

index = 0
#Depending on the mode of view, a different projection matrix is used
def updatePespective():
    global ProjectionMatrix,index
    #Normal projection
    if index==0:
        ProjectionMatrix = numpy.array([[1.2,0,0,0], [0,1.2,0,0],[0,0,-1.04,-0.41],[0,0,-1,0]])
    #Wide angle projection
    if index==1:
        ProjectionMatrix = numpy.array([[0.33,0,0,0], [0,0.33,0,0],[0,0,-1.00,-0.20],[0,0,-1,0]])
    #Ortographic projection
    if index==2:
        ProjectionMatrix = numpy.array([[0.25,0,0,0], [0,0.25,0,0],[0,0,-0.22,-1.22],[0,0,0,1]])

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

#List of triangles to be sorted due to the painter's algorithm
triangles=[]
#List of surface normals
normals=[]
#List of midpoints of triangles
mids=[]
def drawTriangle(triangle,color,use):
    global triangles
    triangles.append([triangle,color,use,max(triangle[0][2][0],triangle[1][2][0],triangle[2][2][0])])

#Counter of triangles
counter=0
def rasterize(triangle,color,use,normal,mid):
    global counter

    #Incorrect flat shading / needs rework
    if mode == 1:
        cam_pos=numpy.dot(numpy.linalg.inv(ViewMatrix), test_vector)
        light=cam_pos-mid
        light=(light/numpy.linalg.norm(light))[0][:3]
        cur_normal=triangles[counter][4]*-1
        cur_normal=cur_normal/numpy.linalg.norm(cur_normal)
        final_value=abs(float(numpy.dot(cur_normal,light)))
        final=int(final_value*11)+4

        pyxel.tri(int(triangle[0][0][0]), int(height - triangle[0][1][0]), int(triangle[1][0][0]), int(height - triangle[1][1][0]),int(triangle[2][0][0]),int(height-triangle[2][1][0]),final)
        counter += 1

    else:
        if use!=2:
            pyxel.line(triangle[0][0][0],height-triangle[0][1][0],triangle[1][0][0],height-triangle[1][1][0],color)
        if use!=0:
            pyxel.line(triangle[1][0][0],height-triangle[1][1][0],triangle[2][0][0],height-triangle[2][1][0],color)
        if use!=1:
            pyxel.line(triangle[2][0][0],height-triangle[2][1][0],triangle[0][0][0],height-triangle[0][1][0],color)


def workTriangle(Triangle,j,color,axis):
    global normals,turnedoff,mids
    out = 0
    place = 3
    inside = 0
    #J is a rotation constant - we want the axes to remain stationary
    if axis:
        j=0
    if not axis:
        for i in range(3):
            Triangle[i] = modelToWorld(Triangle[i], 0, j, 0)
    else:
        for i in range(3):
            Triangle[i] = modelToWorld(Triangle[i], 0, 0, 0)
    first = (Triangle[1] - Triangle[0])[:3].T
    second = (Triangle[2] - Triangle[0])[:3].T
    midX=(Triangle[0][0]+Triangle[1][0]+Triangle[2][0])/3
    midY=(Triangle[0][1]+Triangle[1][1]+Triangle[2][1])/3
    midZ=(Triangle[0][2]+Triangle[1][2]+Triangle[2][2])/3
    midW=(Triangle[0][3]+Triangle[1][3]+Triangle[2][3])/3
    normal = numpy.cross(first, second)
    mid=numpy.array([midX,midY,midZ,midW]).T

    for i in range(3):
        Triangle[i] = worldToView(Triangle[i])  # Moving the world relative to our camera ("moving" the camera)
        Triangle[i] = viewToClip(Triangle[i])  # Applying projection

    #Finding out which vertices are before the near plane
    if -Triangle[0][3] > Triangle[0][2]:
        out+=1
        inside=0
    else:
        place = 0
    if -Triangle[1][3] > Triangle[1][2]:
        out += 1
        inside = 1
    else:
        place = 1
    if -Triangle[2][3] > Triangle[2][2]:
        out += 1
        inside = 2
    else:
        place = 2

    #Near plane clipping
    if out==0:
        for i in range(3):
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
        normals.append(normal)
        mids.append(mid)
        drawTriangle(Triangle,color,3)
    if out==1:
        place = inside
        Swap1=Triangle[(place+1)%3]
        Swap2 = Triangle[(place + 2) % 3]
        one=[0,0,0,0]
        two=[0,0,0,0]

        t01=(  -Triangle[(place+1)%3][3]-Triangle[(place+1)%3][2] ) / ( Triangle[place][3] - Triangle[(place+1)%3][3] + Triangle[(place)][2] - Triangle[(place+1)%3][2]  )
        t02=(  -Triangle[(place+2)%3][3]-Triangle[(place+2)%3][2] ) / ( Triangle[place][3] - Triangle[(place+2)%3][3] + Triangle[(place)][2] - Triangle[(place+2)%3][2]  )

        one[0] = Triangle[(place + 1) % 3][0] + t01*(Triangle[place][0] - Triangle[(place+1)%3][0])
        one[1] = Triangle[(place + 1) % 3][1] + t01*(Triangle[place][1] - Triangle[(place+1)%3][1])
        one[2] = Triangle[(place + 1) % 3][2] + t01*(Triangle[place][2] - Triangle[(place+1)%3][2])
        one[3] = Triangle[(place + 1) % 3][3] + t01*(Triangle[place][3] - Triangle[(place+1)%3][3])

        two[0] = Triangle[(place + 2) % 3][0] + t02*(Triangle[place][0] - Triangle[(place+2)%3][0])
        two[1] = Triangle[(place + 2) % 3][1] + t02*(Triangle[place][1] - Triangle[(place+2)%3][1])
        two[2] = Triangle[(place + 2) % 3][2] + t02*(Triangle[place][2] - Triangle[(place+2)%3][2])
        two[3] = Triangle[(place + 2) % 3][3] + t02*(Triangle[place][3] - Triangle[(place+2)%3][3])

        Triangle[0] = numpy.array(two)
        Triangle[1] = numpy.array(one)
        Triangle[2] = Swap1
        Swap3=Swap1.copy()
        for i in range(3):
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
        normals.append(normal)
        mids.append(mid)
        drawTriangle(Triangle, color,1)
        Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        Triangle[0] = Swap3
        Triangle[1] = Swap2
        Triangle[2] = numpy.array(two)
        for i in range(3):
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
        normals.append(normal)
        mids.append(mid)
        drawTriangle(Triangle, color,1)

    if out==2:
        t01=(  -Triangle[(place+1)%3][3]-Triangle[(place+1)%3][2] ) / ( Triangle[place][3] - Triangle[(place+1)%3][3] + Triangle[(place)][2] - Triangle[(place+1)%3][2]  )
        t02=(  -Triangle[(place+2)%3][3]-Triangle[(place+2)%3][2] ) / ( Triangle[place][3] - Triangle[(place+2)%3][3] + Triangle[(place)][2] - Triangle[(place+2)%3][2]  )

        for i in range(4):
            Triangle[(place + 1) % 3][i] = Triangle[(place + 1) % 3][i] + t01 * (Triangle[place][i] - Triangle[(place + 1) % 3][i])
            Triangle[(place + 2) % 3][i] = Triangle[(place + 2) % 3][i] + t02 * (Triangle[place][i] - Triangle[(place + 2) % 3][i])

        for i in range(3):
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
        normals.append(normal)
        mids.append(mid)
        drawTriangle(Triangle, color,4)

    if out==3:
        pass
j=0
def update():
        global j,triangles, normals,counter,mids
        # An empty triangle
        Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Fill the screen with black color (clear it)
        pyxel.cls(0)
        # Update the view matrix according to the current camera rotation and position
        updateView()
        # Update the projection matrix according to the currently chosen perspective (normal, wide, ortographic)
        updatePespective()
        # Update the choice of mesh
        meshChange()

        for i in range(len(cubeMesh)):
            Triangle[i % 3][0] = (cubeMesh[i][0])
            Triangle[i % 3][1] = (cubeMesh[i][1])
            Triangle[i % 3][2] = (cubeMesh[i][2])
            Triangle[i % 3][3] = (cubeMesh[i][3])
            if i % 3 == 2:
                workTriangle(Triangle,j,13,0)
                Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        #If axes are shown in wireframe mode
        if mode == 0:
            for i in range(len(axes)):
                Triangle[i % 3][0] = (axes[i][0])
                Triangle[i % 3][1] = (axes[i][1])
                Triangle[i % 3][2] = (axes[i][2])
                Triangle[i % 3][3] = (axes[i][3])
                if i%3==2:
                        if i == 2:
                            workTriangle(Triangle,j,1,1)
                            Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                        if i == 5 :
                            workTriangle(Triangle,j, 2,1)
                            Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                        if i == 8 :
                            workTriangle(Triangle,j, 3,1)
                            Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        if not turnedoff:
            j+=1
        if j == 361:
            j=0
        for o in range(len(triangles)):
            triangles[o].append(normals[o])
            triangles[o].append(mids[o])
        #Sort the list in Z order for the painter algorithm
        triangles.sort(key=lambda item : item[3],reverse=True)
        for k in triangles:
            rasterize(k[0],k[1],k[2],k[4],k[5])

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
        triangles=[]
        normals=[]
        mids=[]
        counter=0

#Projection mode (normal, wide angle, ortographic)
mode = 0
#Rotation of the house
turnedoff=1
#Model choice
mesh=0
#Key bindings
def quit():
    if pyxel.btnp(pyxel.KEY_T):
        pyxel.quit()
    global xcam,ycam,zcam,camXangle,camYangle,camZangle,index,mode,turnedoff,curCamX,curCamY,curCamZ,mesh

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
    if pyxel.btnp(pyxel.KEY_P):
        index=(index+1)%3
    if pyxel.btnp(pyxel.KEY_H):
        mesh=(mesh+1)%2
    if pyxel.btnp(pyxel.KEY_B):
        mode=(mode+1)%2
    if pyxel.btnp(pyxel.KEY_C):
        turnedoff=(turnedoff+1)%2

pyxel.init(width, height,fps=fps,palette=[0x000000,0xFF0000, 0x00FF00, 0x0000FF, 0x333333, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xAAAAAA, 0xBBBBBB, 0xCCCCCC, 0xDDDDDD, 0xEEEEEE, 0xFFFFFF])

pyxel.run(update, quit)

