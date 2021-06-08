#!/usr/bin/python
import numpy
import math
import pyxel

#Window dimensions
width = 255
height = 255
fps=60

#Vertices of the object
cubeMesh=[
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

#Axes
axes = [[0, 0, 0],[0.7, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0.7, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0.7],[0, 0, 0]]


#Adding a homogenous coordinate (w=1) to all vertices
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertices to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T

for i in range(len(cubeMesh)):
    homogenous(cubeMesh[i])               # Adding a homogenous coordinate
    cubeMesh[i] = transpose(cubeMesh[i])  # Changing a row vector to a column vector

for i in range(len(axes)):
    homogenous(axes[i])                   # Adding a homogenous coordinate
    axes[i] = transpose(axes[i])          # Changing a row vector to a column vector

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


xcam, ycam, zcam = 0, 0, 0
camXangle, camYangle, camZangle = 0, 0, 0
ViewMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -4], [0, 0, 0, 1]])

curCamX=0
curCamY=0
curCamZ=0
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
def updatePespective():
    global ProjectionMatrix,index
    if index==0:
        ProjectionMatrix = numpy.array([[1.2,0,0,0], [0,1.2,0,0],[0,0,-1.04,-0.41],[0,0,-1,0]])
    if index==1:
        ProjectionMatrix = numpy.array([[0.33,0,0,0], [0,0.33,0,0],[0,0,-1.00,-0.20],[0,0,-1,0]])
    if index==2:
        ProjectionMatrix = numpy.array([[0.25,0,0,0], [0,0.25,0,0],[0,0,-0.22,-1.22],[0,0,0,1]])

def worldToView(vertex):
    global ModelView
    ModelView = numpy.dot(ViewMatrix,vertex)
    return ModelView

def viewToClip(vertex):
    return numpy.dot(ProjectionMatrix,vertex)

def perspectiveDivision(vertex):
    for j in range(4):
        vertex[j]=vertex[j]/vertex[3]
    return vertex

def viewportTransformation(vertex):
    vertex[0] = (vertex[0] * 0.5 + 0.5) * width
    vertex[1] = (vertex[1] * 0.5 + 0.5) * height
    return vertex

def roundPixel(vertex):
    vertex[0]=  round(float(vertex[0][0]))
    vertex[1] = round(float(vertex[1][0]))
    return vertex

triangles=[]
normals=[]
polygoncenters=[]
def drawTriangle(triangle,color,use):
    global triangles
    triangleXcenter=(triangle[0][0][0]+triangle[1][0][0]+triangle[2][0][0])/3
    triangleYcenter=(triangle[0][1][0]+triangle[1][1][0]+triangle[2][1][0])/3
    triangleZcenter=(triangle[0][2][0]+triangle[1][2][0]+triangle[2][2][0])/3
    triangles.append([triangle,color,use,max(triangle[0][2][0],triangle[1][2][0],triangle[2][2][0])])
    pass
counter=0
def rasterize(triangle,color,use):
    global counter

    if mode == 1:
        lookat=(ViewMatrix[:,2][:3].T)*-1
        lookat=lookat/numpy.linalg.norm(lookat)
        cur_normal=triangles[counter][4]*-1
        cur_normal=cur_normal/numpy.linalg.norm(cur_normal)
        asdf=abs(float(numpy.dot(cur_normal,lookat)))
        final=int(asdf*11)+4

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
    global normals,turnedoff
    out = 0
    place = 3
    inside = 0
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
    normal = numpy.cross(first, second)

    for i in range(3):
        Triangle[i] = worldToView(Triangle[i])  # Moving the world relative to our camera ("moving" the camera)
        Triangle[i] = viewToClip(Triangle[i])  # Applying projection
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
    if out==0:
        for i in range(3):
            Triangle[i] = perspectiveDivision(Triangle[i])  # Dividing by W to get to normalised device coordinates
            Triangle[i] = viewportTransformation(Triangle[i])  # Changing the normalised device coordinates to pixels on the screen
            Triangle[i] = roundPixel(Triangle[i])  # Rounding the resulting values to nearest pixel
        normals.append(normal)
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
        drawTriangle(Triangle, color,4)

    if out==3:
        pass
j=0
def update():
        global j,triangles, normals,counter
        # An empty triangle
        Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Fill the screen with black color (clear it)
        pyxel.cls(0)
        # Update the view matrix according to the current camera rotation and position
        updateView()
        # Update the projection matrix according to the currently chosen perspective (normal, wide, ortographic)
        updatePespective()
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
        #Sort the list in Z order for the painter algorithm
        triangles.sort(key=lambda item : item[3],reverse=True)
        for k in triangles:
            rasterize(k[0],k[1],k[2])
        triangles=[]
        normals=[]
        counter=0

mode = 0
turnedoff=1
def quit():
    if pyxel.btnp(pyxel.KEY_T):
        pyxel.quit()
    global xcam,ycam,zcam,camXangle,camYangle,camZangle,index,mode,turnedoff,curCamX,curCamY,curCamZ

    if pyxel.btn(pyxel.KEY_W):
        zcam+=0.05
        curCamZ+=0.05
    if pyxel.btn(pyxel.KEY_S):
        zcam-=0.05
        curCamZ -= 0.05
    if pyxel.btn(pyxel.KEY_A):
        xcam+=0.05
        curCamX += 0.05
    if pyxel.btn(pyxel.KEY_D):
        xcam-=0.05
        curCamX -= 0.05
    if pyxel.btn(pyxel.KEY_Q):
        ycam-=0.05
        curCamY -= 0.05
    if pyxel.btn(pyxel.KEY_E):
        ycam+=0.05
        curCamY += 0.05
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
        print()
        print(ViewMatrix)
        print()
    if pyxel.btnp(pyxel.KEY_B):
        mode=(mode+1)%2
    if pyxel.btnp(pyxel.KEY_C):
        turnedoff=(turnedoff+1)%2

pyxel.init(width, height,fps=fps,palette=[0x000000,0xFF0000, 0x00FF00, 0x0000FF, 0x333333, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xAAAAAA, 0xBBBBBB, 0xCCCCCC, 0xDDDDDD, 0xEEEEEE, 0xFFFFFF])

pyxel.run(update, quit)

