from PIL import Image
import numpy
import math

#Window dimensions
width = 1000
height = 1000

#An empty canvas
img  = Image.new( mode = "RGB", size = (width, height) )
pix=img.load()

#Bresenham's algorithm for line rasterization
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
                    pix[x, int(y) >> 8] = color
                except:
                    return
                y += k

            else:
                try:
                    pix[int(y) >> 8,x] = color
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
                pix[x,y1] = color

        if (dx == 0 and dy != 0):
            if (y2 < y1):
                y1,y2 = y2,y1
            for y in range(y1, y2 + 1):
                if x1 < 0 or x1 > height-1 or y < 0 or y > width-1:
                    return
                pix[x1,y] = color

#Triangle rasterization
def drawTriangle(triangle,color):
    drawLine(triangle[0][0],triangle[0][1],triangle[1][0],triangle[1][1],color)
    drawLine(triangle[1][0],triangle[1][1],triangle[2][0],triangle[2][1],color)
    drawLine(triangle[2][0],triangle[2][1],triangle[0][0],triangle[0][1],color)


#Adding a homogenous coordinate (w)
def homogenous(vertex):
    vertex.append(1)

#Transforming row major vertexes to column major vertexes
def transpose(vertex):
    return numpy.array([vertex]).T

#SPACE CONVERSION

#Our cube is in its model space. We want to put it onto our scene, while rotating it a bit and moving it further away from the camera.
#model space->world space

#Model matrix

xangle = math.radians(69)
yangle = math.radians(39)
zangle = math.radians(0)

#Rotation around the Y axis
xRotationMatrix = numpy.array([[1, 0, 0, 0], [0,  math.cos(xangle), -math.sin(xangle), 0],[0, math.sin(xangle), math.cos(xangle), 0],[0,0,0,1]])
yRotationMatrix = numpy.array([[math.cos(yangle), 0, math.sin(yangle), 0], [0, 1, 0, 0],[-math.sin(yangle), 0, math.cos(yangle), 0],[0,0,0,1]])
zRotationMatrix = numpy.array([[math.cos(zangle), -math.sin(zangle), 0, 0], [math.sin(zangle), math.cos(zangle), 0, 0],[0, 0, 1, 0],[0,0,0,1]])
#Translation along the negative Z axis
TranslationMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, -6],[0,0,0,1]])
ModelMatrix = numpy.dot(yRotationMatrix,xRotationMatrix)
ModelMatrix = numpy.dot(zRotationMatrix,ModelMatrix)
ModelMatrix=numpy.dot(TranslationMatrix,ModelMatrix)
#Applying the transformation to all of our vertexes
def modelToWorld(vertex):
    return numpy.dot(ModelMatrix,vertex)


#Now we want to move our camera
#We cannot move the camera itself, we need to move the world. So in order to move the camera 1 unit closer to the cube,
#we need to move the cube closer to the camera. Remember, the camera always points to the negative Z axis.
#world space->view space

#View matrix
ViewMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 3],[0,0,0,1]])
#Applying the transformation to all of our vertexes
def worldToView(vertex):
    return numpy.dot(ViewMatrix,vertex)

#Mame umiestnenu kameru a mame umiestnenu kocku. Teraz potrebujeme vysledny obraz zobrazit na platno.
#Nasleduje maticovy prevod view space->clip space

#Projection matrix
ProjectionMatrix = numpy.array([[0.8,0,0,0], [0,0.8,0,0],[0,0,-1.22,-2.22],[0,0,-1,0]])
#ProjectionMatrix = numpy.array([[0.25,0,0,0], [0,0.25,0,0],[0,0,-0.22,-1.22],[0,0,0,1]])

def viewToClip(vertex):
    return numpy.dot(ProjectionMatrix,vertex)


#Teraz musime vydelit suradnice vsetkych bodov ich W hodnotami (Perspective division)
def perspectiveDivision(vertex):
    for j in range(4):
        vertex[j]=vertex[j]/vertex[3]
    return vertex

#Teraz musime previest na samotne pixely (Viewport transformation)
def viewportTransformation(vertex):
    vertex[0] = (vertex[0] * 0.5 + 0.5) * width
    vertex[1] = (vertex[1] * 0.5 + 0.5) * height
    return vertex

#A nakoniec sa zbavime skaredych desatinnych miest zaokruhlovanim na najblizsi pixel.
def roundPixel(vertex):
    vertex[0]=  int(round(vertex[0][0]))
    vertex[1] = int(round(vertex[1][0]))
    return vertex




#Vertexes of cube triangles
cubeMesh=[
                [-1,-1,-1],[1,-1,-1],[1,-1,1]  ,  [-1,-1,-1],[1,-1,1],[-1,-1,1], #TOP
                [1,-1,-1],[1,1,-1],[1,1,1]     ,  [1,-1,-1],[1,1,1],[1,-1,1],    #RIGHT
                [-1,1,-1],[-1,-1,-1],[-1,-1,1] ,  [-1,1,-1],[-1,-1,1],[-1,1,1],  #LEFT
                [1,1,-1],[-1,1,-1],[-1,1,1]    ,  [1,1,-1],[-1,1,1],[1,1,1],     #BOTTOM
                [-1,-1,1],[1,-1,1],[1,1,1]   ,  [-1,-1,1],[1,1,1],[-1,1,1],      #NEAR
                [1,-1,-1],[-1,-1,-1],[-1,1,-1]  ,  [1,-1,-1],[-1,1,-1],[1,1,-1]  #FAR
]


#An empty triangle
Triangle=[[0,0],[0,0],[0,0]]

#Colors
colors = [(255,0,0),(255,0,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(255,255,0),(255,255,0),(0,255,255),(0,255,255),(255,0,255),(255,0,255)]

#Triangle counter
counter=0
for i in range(len(cubeMesh)):
    homogenous(cubeMesh[i])                            #Adding a homogenous coordinate
    cubeMesh[i]=transpose(cubeMesh[i])                 #Changing a row vector to a column vector
    cubeMesh[i]=modelToWorld(cubeMesh[i])              #Moving our model to its place on world coordinates
    cubeMesh[i]=worldToView(cubeMesh[i])               #Moving the world relative to our camera ("moving" the camera)
    cubeMesh[i]=viewToClip(cubeMesh[i])                #Applying projection
    cubeMesh[i]=perspectiveDivision(cubeMesh[i])       #Dividing by W to get to normalised device coordinates
    cubeMesh[i]=viewportTransformation(cubeMesh[i])    #Changing the normalised device coordinates to pixels on the screen
    cubeMesh[i]=roundPixel(cubeMesh[i])                #Rounding the resulting values to nearest pixel
    Triangle[i%3][0] = int(cubeMesh[i][0])
    Triangle[i%3][1] = int(cubeMesh[i][1])
    if i%3==2: #mame trojuholnik
        drawTriangle(Triangle,colors[counter%12])
        counter+=1
img.show()