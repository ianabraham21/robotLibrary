from robotUtil import *
from math import pi
import numpy as np


B3 = [0,0,1,0,0,0]
B2 = [0,0,1,0,1,0]
B1 = [0,0,1,0,2,0]

Bi = [B1,B2,B3]
M = [[1,0,0,2],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,0,1]]

Tsd = [[1,0,0,0],
	[0,1,0,1.5],
	[0,0,1,0],
	[0,0,0,1]]

#theta0 = [0.8,-0.1,0.1,-0.1,0.1,-0.01]
theta0 = [0.01,0.01,0.01]#np.absolute(0.1*np.random.rand(3))#[1,1,1,1,1,1]
print "Initial Joint Angles: ", theta0
Theta = IKinBody(Bi, M, Tsd, theta0)
print np.cos(Theta[-1][0])+np.cos(Theta[-1][0]+Theta[-1][1])
print np.sin(Theta[-1][0])+np.sin(Theta[-1][0]+Theta[-1][1])
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