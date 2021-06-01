import numpy
import math
import tkinter

#Window dimensions
width = 1000
height = 1000

#An empty canvas
cnv = tkinter.Canvas(bg='white', width=width, height=height)

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
                    cnv.create_line(x, int(y) >> 8,x+1,int(y) >> 8,fill=color)
                except:
                    return
                y += k

            else:
                try:
                    cnv.create_line(int(y) >> 8,x,(int(y) >> 8)+1,x,fill=color)
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
                cnv.create_line(x, y1, x+1, y1, fill=color)

        if (dx == 0 and dy != 0):
            if (y2 < y1):
                y1,y2 = y2,y1
            for y in range(y1, y2 + 1):
                if x1 < 0 or x1 > height-1 or y < 0 or y > width-1:
                    return
                cnv.create_line(x1, y, x1+1, y, fill=color)

#Triangle rasterization
def drawTriangle(triangle,color,width):
    cnv.create_line(triangle[0][0],height-triangle[0][1],triangle[1][0],height-triangle[1][1],fill=color,width=width)
    cnv.create_line(triangle[1][0],height-triangle[1][1],triangle[2][0],height-triangle[2][1],fill=color,width=width)
    cnv.create_line(triangle[2][0],height-triangle[2][1],triangle[0][0],height-triangle[0][1],fill=color,width=width)
    #drawLine(triangle[0][0],height-triangle[0][1],triangle[1][0],height-triangle[1][1],color)
    #drawLine(triangle[1][0],height-triangle[1][1],triangle[2][0],height-triangle[2][1],color)
    #drawLine(triangle[2][0],height-triangle[2][1],triangle[0][0],height-triangle[0][1],color)

#Adding a homogenous coordinate (w)
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertexes to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T


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
ViewMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -6], [0, 0, 0, 1]])


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

def worldToView(vertex):
    return numpy.dot(ViewMatrix,vertex)
#Now we need to apply the projection matrix to create perspective.
#view space->clip space

#Projection matrix
ProjectionMatrix = numpy.array([[0.8,0,0,0], [0,0.8,0,0],[0,0,-1.22,-2.22],[0,0,-1,0]])
#ProjectionMatrix = numpy.array([[0.25,0,0,0], [0,0.25,0,0],[0,0,-0.22,-1.22],[0,0,0,1]])
def viewToClip(vertex):
    return numpy.dot(ProjectionMatrix,vertex)

#In order to turn the resulting coordinates into NDC, we need to divide by W.
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
    vertex[0]=  int(round(vertex[0][0]))
    vertex[1] = int(round(vertex[1][0]))
    return vertex

#Vertexes of the object
cubeMesh=[
#Ground
[-2, 0, -2],[2, 0, -2],[-2, 0, 2],
[-2, 0, 2],[2, 0, -2],[2, 0, 2],

#House
[-0.5, 0., -0.5 ],
[-0.5, 1., -0.5 ],
[-0.5, 0, 0.5],

[-0.5, 1, -0.5],
[-0.5, 0., 0.5],
[-0.5, 1., 0.5],

[-0.5, 0., -0.5],
[-0.5, 1., -0.5],
[0.5, 0., -0.5],

[-0.5, 1., -0.5],
[0.5, 0., -0.5],
[0.5, 1., -0.5],

[0.5, 0., -0.5],
[0.5, 1., -0.5],
[0.5, 0., 0.5],

#Roof
[0.5, 1., -0.5],
[0.5, 0., 0.5],
[0.5, 1, 0.5],

[-0.5, 0., 0.5],
[-0.5, 1., 0.5],
[0.5, 0., 0.5],

[-0.5, 1., 0.5],
[0.5, 0., 0.5],
[0.5, 1., 0.5],

[-0.25, 0.0, 0.5],
[0.25, 0., 0.5],
[0.25, 0.5, 0.5],

[-0.25, 0., 0.5],
[0.25, 0.5, 0.5],
[-0.25, 0.5, 0.5],


#Door
[-0.5, 1., -0.5],
[-0.5, 1., 0.5],
[0., 2., 0.0],

[0.5, 1., -0.5],
[0.5, 1., 0.5],
[0., 2., 0.0],

[-0.5, 1., -0.5],
[0.5, 1., -0.5],
[0., 2., 0.0],

[-0.5, 1, 0.5],
[0.5, 1, 0.5],
[0., 2., 0.0],

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

#An empty triangle
Triangle=[[0,0,0],[0,0,0],[0,0,0]]

#Colors

#Triangle counter
cnv.pack()


for i in range(len(cubeMesh)):
    homogenous(cubeMesh[i])               # Adding a homogenous coordinate
    cubeMesh[i] = transpose(cubeMesh[i])  # Changing a row vector to a column vector

for i in range(len(axes)):
    homogenous(axes[i])               # Adding a homogenous coordinate
    axes[i] = transpose(axes[i])  # Changing a row vector to a column vector

changingMesh = cubeMesh.copy()
changingAxes = axes.copy()
j=0
def update():
        cnv.delete("all")
        updateView()
        global j,counter,colors
        counter=0
        outPoints=0
        for i in range(len(cubeMesh)):
            changingMesh[i]=modelToWorld(cubeMesh[i],0,0,0)              #Moving our model to its place on world coordinates
            changingMesh[i]=worldToView(changingMesh[i])               #Moving the world relative to our camera ("moving" the camera)
            changingMesh[i]=viewToClip(changingMesh[i])                #Applying projection
            if -changingMesh[i][3]>changingMesh[i][2]:
                outPoints+=1
            changingMesh[i] = perspectiveDivision(changingMesh[i])  # Dividing by W to get to normalised device coordinates
            changingMesh[i] = viewportTransformation(changingMesh[i])  # Changing the normalised device coordinates to pixels on the screen
            changingMesh[i] = roundPixel(changingMesh[i])  # Rounding the resulting values to nearest pixel
            Triangle[i%3][0] = int(changingMesh[i][0])
            Triangle[i%3][1] = int(changingMesh[i][1])
            Triangle[i%3][2] = int(changingMesh[i][2])
            if i%3==2:
                counter+=1
                if outPoints==0:
                    drawTriangle(Triangle,'black',1)
                outPoints=0


        outPoints = 0
        for i in range(len(axes)):
            changingAxes[i]=modelToWorld(axes[i],0,0,0)              #Moving our model to its place on world coordinates
            changingAxes[i]=worldToView(changingAxes[i])               #Moving the world relative to our camera ("moving" the camera)
            changingAxes[i]=viewToClip(changingAxes[i])                #Applying projection
            if -changingAxes[i][3]>changingAxes[i][2]:
                outPoints+=1
            changingAxes[i] = perspectiveDivision(changingAxes[i])  # Dividing by W to get to normalised device coordinates
            changingAxes[i] = viewportTransformation(changingAxes[i])  # Changing the normalised device coordinates to pixels on the screen
            changingMesh[i] = roundPixel(changingAxes[i])  # Rounding the resulting values to nearest pixel
            Triangle[i%3][0] = int(changingAxes[i][0])
            Triangle[i%3][1] = int(changingAxes[i][1])
            Triangle[i%3][2] = int(changingAxes[i][2])
            if i%3==2:
                if outPoints == 0:
                    if i == 2:
                        drawTriangle(Triangle,'red',3)
                    if i == 5 :
                        drawTriangle(Triangle, 'green',3)
                    if i == 8 :
                        drawTriangle(Triangle, 'blue',3)
                outPoints=0



        j+=1
        if j == 360:
            j=0
        cnv.after(10,update)

update()
def move(event):
    global xcam
    global ycam
    global zcam
    global camXangle
    global camYangle
    global camZangle

    if event.char=='w':
        zcam+=0.2
    if event.char == 's':
        zcam-=0.2
    if event.char=='a':
        xcam+=0.2
    if event.char == 'd':
        xcam-=0.2
    if event.char=='q':
        ycam-=0.2
    if event.char == 'e':
        ycam+=0.2

    if event.char == 'i':
        camXangle -= 4
    if event.char == 'k':
        camXangle += 4
    if event.char == 'j':
        camYangle -= 4
    if event.char == 'l':
        camYangle += 4
    if event.char == 'u':
        camZangle -= 4
    if event.char == 'o':
        camZangle +=4


cnv.bind_all('<Key>', move)

tkinter.mainloop()