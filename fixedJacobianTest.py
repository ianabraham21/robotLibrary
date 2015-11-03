from math import pi
from robotUtil import *
S1 = [0,0,1,0,0,0]
S2 = [0,0,1,0,-1,0]
S3 = [0,0,1,0,-2,0]
S4 = [0,0,0,0,0,1]

M = [[1,0,0,2],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]]

theta0 = [pi/2,pi/2,pi/2,2]

Si = [S1,S2,S3,S4]

print FixedJacobian(theta0, Si)