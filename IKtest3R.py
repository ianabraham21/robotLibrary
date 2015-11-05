from robotUtil import *
from math import pi
import numpy as np


B3 = [0,0,1,0,0,0]
B2 = [0,0,1,0,1,0]
B1 = [0,0,1,0,2,0]

S1 = [0,0,1,0,0,0]
S2 = [0,0,1,0,-1,0]
S3 = [0,0,1,0,-2,0]

Bi = [B1,B2,B3]
Si = [S1,S2,S3]

M = [[1,0,0,2],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]]

Tsd = [[1,0,0,1.5],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]]

#theta0 = [0.8,-0.1,0.1,-0.1,0.1,-0.01]
# works for both -> [ 0.02738007  0.06867218  0.09145575]
#  [ 0.06647238  0.03890556  0.01746679]

theta0 = np.absolute(0.1*np.random.rand(3))#[1,1,1,1,1,1]
print "Initial Joint Angles: ", theta0
Theta = IKinBody(Bi, M, Tsd, theta0)
Theta = IKinFixed(Si, M, Tsd, theta0)
size = np.shape(Theta)


'''
import csv
with open('test1.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'])
    for i in range(size[0]):
    	spamwriter.writerow(np.insert(Theta[i][:],0,i*0.1))
    	'''