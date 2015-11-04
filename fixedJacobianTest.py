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

from robotUtil import *
from math import pi
import numpy as np


B3 = [0,0,1,0,0,0]
B2 = [0,0,1,0,1,0]
B1 = [0,0,1,0,2,0]

Bi = [B1,B2]
M = [[1,0,0,2],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]]

Tsd = [[1,0,0,0],
	[0,1,0,1.5],
	[0,0,1,0],
	[0,0,0,1]]
print BodyJacobian([pi,pi],Bi)

print FixedJacobian(theta0, Si)