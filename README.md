A simple 3D engine which renders vertices in real time using a movable camera. Used to learn and understand the transformation between spaces and the subsequent projection onto screen.

![image](https://user-images.githubusercontent.com/74743561/121445985-dc198400-c992-11eb-9768-007a3f1d0af3.png)


The goal was to minimize the use of external libraries and blackbox frameworks and to showcase exactly what happens when a vertex is projected onto the screen. 
# Moving the camera
Use WASD keys to move the camera forwards, backwards, as if it was a plane (going forwards means that the plane will go where it's nose is pointed) in contrast with an FPS game, where, even if pointing the gun at your feet, your character will walk forwards after pressing W.

Use EQ to elevate or lower the camera.
# Rotating the camera
Use IK keys to control the pitch.

Use JL keys to controll the yaw.

Use UO keys to control the roll.

#Moving the object
The object moves along the global axes.

Use 8 and 5 on the numpad to move the mesh along the Z axis.
Use 4 and 6 on the numpad to move the mesh along the X axis.
Use 7 and 9 on the numpad to move the mesh along the X axis.


# Other controls.
Use P key to switch between the projection matrices (small angle, wide angle, ortographic)

Use B key to switch between wireframe mode and shaded mode.

Use H key to switch between the meshes (house and icosahedron).

Use C key to turn the rotation of the mesh on/off.

Use T key to exit.
# Dependencies
Uses Pyxel, math, and numpy packages.
