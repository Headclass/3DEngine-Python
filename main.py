import numpy
import math
import pyxel

#Window dimensions
width = 255
height = 255

#Vertexes of the object
cubeMesh=[
#Ground
[-2, 0, -2],[2, 0, -2],[-2, 0, 2],
[-2, 0, 2],[2, 0, -2],[2, 0, 2],

#House
[-0.5, 0., 0.5 ],
[0.5, 0., 0.5 ],
[0.5, 1., 0.5 ],

[-0.5, 0., 0.5 ],
[0.5, 1., 0.5 ],
[-0.5, 1., 0.5 ],

[0.5, 0., 0.5 ],
[0.5, 0., -0.5 ],
[0.5, 1., -0.5 ],

[0.5, 0., 0.5 ],
[0.5, 1., -0.5 ],
[0.5, 1., 0.5 ],

[0.5, 0., -0.5 ],
[-0.5, 0., -0.5 ],
[0.5, 1., -0.5 ],

[-0.5, 0., -0.5 ],
[-0.5, 1., -0.5 ],
[0.5, 1., -0.5 ],

[-0.5, 0., -0.5 ],
[-0.5, 0., 0.5 ],
[-0.5, 1., 0.5 ],

[-0.5, 0., -0.5 ],
[-0.5, 1., 0.5 ],
[-0.5, 1., -0.5 ],


[-0.5, 1., 0.5 ],
[0.5, 1., 0.5 ],
[0, 2., 0 ],

[0.5, 1., 0.5 ],
[0.5, 1., -0.5 ],
[0, 2., 0 ],

[0.5, 1., -0.5 ],
[-0.5, 1., -0.5 ],
[0, 2., 0 ],

[-0.5, 1., -0.5 ],
[-0.5, 1., 0.5 ],
[0, 2., 0 ]



]

#Axes
axes = [
    #x
[0, 0, 0],
[0.7, 0, 0],
[0, 0, 0],
    #y
[0, 0, 0],
[0, 0.7, 0],
[0, 0, 0],
    #z
[0, 0, 0],
[0, 0, 0.7],
[0, 0, 0]
]


#Adding a homogenous coordinate to all vertexes (w)
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertexes to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T


for i in range(len(cubeMesh)):
    homogenous(cubeMesh[i])               # Adding a homogenous coordinate
    cubeMesh[i] = transpose(cubeMesh[i])  # Changing a row vector to a column vector

for i in range(len(axes)):
    homogenous(axes[i])                   # Adding a homogenous coordinate
    axes[i] = transpose(axes[i])          # Changing a row vector to a column vector

#Line drawing algorithm
def drawLine(x1, y1, x2, y2, color):
    chan = 0
    dx = x2 - x1
    dy = y2 - y1
    if (dx != 0 and dy != 0):
        if (abs(dy) > abs(dx)):
            x1, y1 = y1,x1
            x2, y2 = y2, x2
            dx = x2 - x1
            dy = y2 - y1
            chan = 1
        if (x2 < x1):
            x1, x2=x2, x1
            dx = x2 - x1
            y1,y2 = y2,y1
            dy = y2 - y1

        k = (dy << 8) / dx
        y = y1 << 8
        for x in range(x1,x2+1):
            if (chan == 0):
                try:
                    pyxel.pset(x, int(y) >> 8,color)
                except:
                    return
                y += k

            else:
                try:
                    pyxel.pset(int(y) >> 8,x,color)
                except:
                    return
                y += k
    else:
        if (dy == 0 and dx != 0):
            if (x2 < x1):
                x1, x2=x2, x1
            for x in range(x1,x2+1):
                if y1 < 0 or y1 > height-1 or x < 0 or x > width-1:
                    return
                pyxel.pset(x, y1,color)

        if (dx == 0 and dy != 0):
            if (y2 < y1):
                y1,y2 = y2,y1
            for y in range(y1, y2 + 1):
                if x1 < 0 or x1 > height-1 or y < 0 or y > width-1:
                    return
                pyxel.pset(x1, y, color)

#SPACE CONVERSION

#Our cube is in its model space. We want to put it onto our scene, while rotating it a bit and moving it further away from the camera.
#model space->world space
#Applying the transformation to all of our vertexes
def modelToWorld(vertex,x,y,z):
    # Rotation angles
    xangle = math.radians(x)
    yangle = math.radians(y)
    zangle = math.radians(z)

    # Rotation matrices
    xRotationMatrix = numpy.array([[1, 0, 0, 0], [0, math.cos(xangle), -math.sin(xangle), 0], [0, math.sin(xangle), math.cos(xangle), 0],[0, 0, 0, 1]])
    yRotationMatrix = numpy.array([[math.cos(yangle), 0, math.sin(yangle), 0], [0, 1, 0, 0], [-math.sin(yangle), 0, math.cos(yangle), 0],[0, 0, 0, 1]])
    zRotationMatrix = numpy.array([[math.cos(zangle), -math.sin(zangle), 0, 0], [math.sin(zangle), math.cos(zangle), 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
    # Translation along the negative Z axis
    TranslationMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Combining the transformations into one model matrix
    ModelMatrix = numpy.dot(yRotationMatrix, xRotationMatrix)
    ModelMatrix = numpy.dot(zRotationMatrix, ModelMatrix)
    ModelMatrix = numpy.dot(TranslationMatrix, ModelMatrix)
    return numpy.dot(ModelMatrix,vertex)


#Now we want to move our camera
#We cannot move the camera itself, we need to move the world. So in order to move the camera 1 unit closer to the cube,
#we need to move the cube closer to the camera. Remember, the camera always points to the negative Z axis.
#world space->view space
#View matrix
#Applying the transformation to all of our vertexes
xcam, ycam, zcam = 0, 0, 0
camXangle, camYangle, camZangle = 0, 0, 0
ViewMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -4], [0, 0, 0, 1]])

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
ModelView = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def worldToView(vertex):
    global ModelView
    ModelView = numpy.dot(ViewMatrix,vertex)
    return ModelView
#Now we need to apply the projection matrix to create perspective.
#view space->clip space

#Projection matrix

def viewToClip(vertex):
    return numpy.dot(ProjectionMatrix,vertex)

#In order to turn the resulting coordinates into NDC, we need to divide by W`.
def perspectiveDivision(vertex):
    for j in range(4):
        vertex[j]=vertex[j]/vertex[3]
    return vertex

#Turning values from -1 to 1 into individual pixels on the screen
def viewportTransformation(vertex):
    vertex[0] = (vertex[0] * 0.5 + 0.5) * width
    vertex[1] = (vertex[1] * 0.5 + 0.5) * height
    return vertex

#Rounding the resulting values
def roundPixel(vertex):
    vertex[0]=  round(float(vertex[0][0]))
    vertex[1] = round(float(vertex[1][0]))
    return vertex


#Triangle rasterization
triangles=[]
normals=[]
def drawTriangle(triangle,color,use):
    global triangles
    triangles.append([triangle,color,use,max(triangle[0][2][0],triangle[1][2][0],triangle[2][2][0])])
    """
    if use!=2:
        #drawLine(int(triangle[0][0][0]), int(height - triangle[0][1][0]), int(triangle[1][0][0]), int(height - triangle[1][1][0]), color)
        pyxel.line(triangle[0][0][0],height-triangle[0][1][0],triangle[1][0][0],height-triangle[1][1][0],color)
    if use!=0:
        #drawLine(int(triangle[1][0][0]),int(height-triangle[1][1][0]),int(triangle[2][0][0]),int(height-triangle[2][1][0]),color)
        pyxel.line(triangle[1][0][0],height-triangle[1][1][0],triangle[2][0][0],height-triangle[2][1][0],color)
    if use!=1:
        #drawLine(int(triangle[2][0][0]),int(height-triangle[2][1][0]),int(triangle[0][0][0]),int(height-triangle[0][1][0]),color)
        pyxel.line(triangle[2][0][0],height-triangle[2][1][0],triangle[0][0][0],height-triangle[0][1][0],color)
    """
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
            #drawLine(int(triangle[0][0][0]), int(height - triangle[0][1][0]), int(triangle[1][0][0]), int(height - triangle[1][1][0]), color)
            pyxel.line(triangle[0][0][0],height-triangle[0][1][0],triangle[1][0][0],height-triangle[1][1][0],color)
        if use!=0:
            #drawLine(int(triangle[1][0][0]),int(height-triangle[1][1][0]),int(triangle[2][0][0]),int(height-triangle[2][1][0]),color)
            pyxel.line(triangle[1][0][0],height-triangle[1][1][0],triangle[2][0][0],height-triangle[2][1][0],color)
        if use!=1:
            #drawLine(int(triangle[2][0][0]),int(height-triangle[2][1][0]),int(triangle[0][0][0]),int(height-triangle[0][1][0]),color)
            pyxel.line(triangle[2][0][0],height-triangle[2][1][0],triangle[0][0][0],height-triangle[0][1][0],color)


def workTriangle(Triangle,j,color,axis):
    global normals
    out = 0
    place = 3
    inside = 0
    if axis:
        j=0
    for i in range(3):
        Triangle[i] = modelToWorld(Triangle[i], 0, j, 0)  # Moving our model to its place on world coordinates

    # normal
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

        Triangle[(place + 1) % 3][0] = Triangle[(place + 1) % 3][0] + t01*(Triangle[place][0] - Triangle[(place+1)%3][0])
        Triangle[(place + 1) % 3][1] = Triangle[(place + 1) % 3][1] + t01*(Triangle[place][1] - Triangle[(place+1)%3][1])
        Triangle[(place + 1) % 3][2] = Triangle[(place + 1) % 3][2] + t01*(Triangle[place][2] - Triangle[(place+1)%3][2])
        Triangle[(place + 1) % 3][3] = Triangle[(place + 1) % 3][3] + t01*(Triangle[place][3] - Triangle[(place+1)%3][3])

        Triangle[(place + 2) % 3][0] = Triangle[(place + 2) % 3][0] + t02*(Triangle[place][0] - Triangle[(place+2)%3][0])
        Triangle[(place + 2) % 3][1] = Triangle[(place + 2) % 3][1] + t02*(Triangle[place][1] - Triangle[(place+2)%3][1])
        Triangle[(place + 2) % 3][2] = Triangle[(place + 2) % 3][2] + t02*(Triangle[place][2] - Triangle[(place+2)%3][2])
        Triangle[(place + 2) % 3][3] = Triangle[(place + 2) % 3][3] + t02*(Triangle[place][3] - Triangle[(place+2)%3][3])

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
        pyxel.cls(0)
        updateView()
        updatePespective()
        for i in range(len(cubeMesh)):
            Triangle[i % 3][0] = (cubeMesh[i][0])
            Triangle[i % 3][1] = (cubeMesh[i][1])
            Triangle[i % 3][2] = (cubeMesh[i][2])
            Triangle[i % 3][3] = (cubeMesh[i][3])
            if i % 3 == 2:
                workTriangle(Triangle,j,13,0)
                Triangle = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
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

        j+=1
        if j == 360:
            j=0
        for o in range(len(triangles)):
            triangles[o].append(normals[o])
        triangles.sort(key=lambda item : item[3],reverse=True)
        for k in triangles:
            rasterize(k[0],k[1],k[2])
        triangles=[]
        normals=[]
        counter=0

mode = 0
def quit():
    if pyxel.btnp(pyxel.KEY_T):
        pyxel.quit()
    global xcam
    global ycam
    global zcam
    global camXangle
    global camYangle
    global camZangle
    global index
    global mode

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
        print(ViewMatrix)
    if pyxel.btnp(pyxel.KEY_B):
        mode=(mode+1)%2


pyxel.init(width, height,fps=60,palette=[0x000000,0xFF0000, 0x00FF00, 0x0000FF, 0x333333, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xAAAAAA, 0xBBBBBB, 0xCCCCCC, 0xDDDDDD, 0xEEEEEE, 0xFFFFFF])

pyxel.run(update, quit)

