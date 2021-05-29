from PIL import Image
import numpy
import math

#Window dimensions
width = 400
height = 400

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
                pix[x, int(y) >> 8] = color
                y += k

            else:
                pix[int(y) >> 8,x] = color
                y += k
    else:
        if (dy == 0 and dx != 0):
            if (x2 < x1):
                x1, x2=x2, x1
            for x in range(x1,x2+1):
                pix[x,y1] = color

        if (dx == 0 and dy != 0):
            if (y2 < y1):
                y1,y2 = y2,y1
            for y in range(y1, y2 + 1):
                pix[x1,y] = color

#Triangle rasterization
def drawTriangle(triangle,color):
    drawLine(triangle[0][0],triangle[0][1],triangle[1][0],triangle[1][1],color)
    drawLine(triangle[1][0],triangle[1][1],triangle[2][0],triangle[2][1],color)
    drawLine(triangle[2][0],triangle[2][1],triangle[0][0],triangle[0][1],color)

#Triangle vertexes of a cube
cubeMesh = [
                [-1,-1,-1],[1,-1,-1],[1,-1,1]  ,  [-1,-1,-1],[1,-1,1],[-1,-1,1], #TOP
                [1,-1,-1],[1,1,-1],[1,1,1]     ,  [1,-1,-1],[1,1,1],[1,-1,1],    #RIGHT
                [-1,1,-1],[-1,-1,-1],[-1,-1,1] ,  [-1,1,-1],[-1,-1,1],[-1,1,1],  #LEFT
                [1,1,-1],[-1,1,-1],[-1,1,1]    ,  [1,1,-1],[-1,1,1],[1,1,1],     #BOTTOM
                [-1,-1,1],[1,-1,1],[1,1,1]   ,  [-1,-1,1],[1,1,1],[-1,1,1],      #NEAR
                [1,-1,-1],[-1,-1,-1],[-1,1,-1]  ,  [1,-1,-1],[-1,1,-1],[1,1,-1]  #FAR
]


#An empty triangle
Triangle=[[0,0],[0,0],[0,0]]

#Adding a homogenous coordinate (w)
for i in range(len(cubeMesh)):
    cubeMesh[i].append(1)

#Transforming row major vertexes to column major vertexes
for i in range(len(cubeMesh)):
    i=numpy.array([cubeMesh[i]]).T



#SPACE CONVERSION

#Our cube is in its model space. We want to put it onto our scene, while rotating it a bit and moving it further away from the camera.
#model space->world space

#Model matrix
angle = math.radians(36)

#Rotation about the Y axis
RotationMatrix = numpy.array([[math.cos(angle), 0, math.sin(angle), 0], [0, 1, 0, 0],[-math.sin(angle), 0, math.cos(angle), 0],[0,0,0,1]])
#Translation along the negative Z axis
TranslationMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, -3],[0,0,0,1]])
ModelMatrix = numpy.dot(TranslationMatrix,RotationMatrix)
#Applying the transformation to all of our vertexes
for i in range(len(cubeMesh)):
    cubeMesh[i]=numpy.dot(ModelMatrix,cubeMesh[i])


#Now we want to move our camera
#We cannot move the camera itself, we need to move the world. So in order to move the camera 1 unit closer to the cube,
#we need to move the cube closer to the camera. Remember, the camera always points to the negative Z axis.
#world space->view space

#View matrix
ViewMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0.2],[0,0,0,1]])
#Applying the transformation to all of our vertexes
for i in range(len(cubeMesh)):
    cubeMesh[i]=numpy.dot(ViewMatrix,cubeMesh[i])

print(cubeMesh)
#Mame umiestnenu kameru a mame umiestnenu kocku. Teraz potrebujeme vysledny obraz zobrazit na platno.
#Nasleduje maticovy prevod view space->clip space

#Projection matrix
ProjectionMatrix = numpy.array([[0.8,0,0,0], [0,0.8,0,0],[0,0,-1.22,-2.22],[0,0,-1,0]])
#ProjectionMatrix = numpy.array([[0.25,0,0,0], [0,0.25,0,0],[0,0,-0.22,-1.22],[0,0,0,1]])

for i in range(len(cubeMesh)):
    cubeMesh[i]=numpy.dot(ProjectionMatrix,cubeMesh[i])

print(cubeMesh)


#Teraz musime vydelit suradnice vsetkych bodov ich W hodnotami (Perspective division)
for i in range(len(cubeMesh)):
    for j in range(4):
        cubeMesh[i][j]=cubeMesh[i][j]/cubeMesh[i][3]
print(cubeMesh)

#Teraz musime previest na samotne pixely (Viewport transformation)
for i in range(len(cubeMesh)):
    cubeMesh[i][0]=(cubeMesh[i][0]*0.5+0.5)*width
    cubeMesh[i][1] = (cubeMesh[i][1] * 0.5 + 0.5) * height

#A nakoniec sa zbavime skaredych desatinnych miest zaokruhlovanim na najblizsi pixel.
for i in range(len(cubeMesh)):
    cubeMesh[i][0]=int(round(cubeMesh[i][0]))
    cubeMesh[i][1] = int(round(cubeMesh[i][1]))
print(cubeMesh)
#print(cubeMesh)
counter=0

colors = [(255,0,0),(255,0,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(255,255,0),(255,255,0),(0,255,255),(0,255,255),(255,0,255),(255,0,255)]
for i in range(len(cubeMesh)):
    Triangle[i%3][0] = int(cubeMesh[i][0])
    Triangle[i%3][1] = int(cubeMesh[i][1])
    if i%3==2: #mame trojuholnik
        drawTriangle(Triangle,colors[counter%12])
        counter+=1
img.show()
