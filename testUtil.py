from robotUtil import *

Theta = [pi/8,pi/2,pi/4]
Si =  [[0,0,-1,2,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0.1]]

print FixedJacobian(Theta, Si)
print BodyJacobian(Theta, Si)
