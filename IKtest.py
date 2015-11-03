from robotUtil import *
from math import pi
import numpy as np
H1 = 89*10e-3
H2 = 95*10e-3
W2 = 82*10e-3
L2 = 392*10e-3
L1 = 425*10e-3
W1 = 109*10e-3
B6 = [0,0,1,0,0,0]
B5 = [0,-1,0,-W2,0,0]
B4 = [0,0,1,H2,0,0]
B3 = [0,0,1,H2,-L2,0]
B2 = [0,0,1,H2,-L2-L1,0]
B1 = [0,1,0,W1+W2,0,L1+L2]

Bi = [B1,B2,B3,B4,B5,B6]
M = [[1,0,0,-L2-L1],
	[0,0,-1,-W1-W2],
	[0,1,0,H1-H2],
	[0,0,0,1]]

Tsd = [[0,1,0,-0.6],[0,0,-1,0.1],[-1,0,0,0.1],[0,0,0,1]]

#theta0 = [0.8,-0.1,0.1,-0.1,0.1,-0.01]
theta0 = 2*pi*np.random.rand(6)-pi#[1,1,1,1,1,1]
print theta0
Theta = IKinBody(Bi, M, Tsd, theta0)
size = np.shape(Theta)

import csv
with open('test1.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'])
    for i in range(size[0]):
    	spamwriter.writerow(np.insert(Theta[i][:],0,i*0.1))