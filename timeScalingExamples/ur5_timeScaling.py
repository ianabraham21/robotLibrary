from robotUtil import *
from math import pi
import numpy as np
import matplotlib.pyplot as plt

saveData = True

# MACHINE CONSTANTS
H1 = 89*1e-3
H2 = 95*1e-3
W2 = 82*1e-3
L2 = 392*1e-3
L1 = 425*1e-3
W1 = 109*1e-3

B6 = [0,0,1,0,0,0]
B5 = [0,-1,0,-W2,0,0]
B4 = [0,0,1,H2,0,0]
B3 = [0,0,1,H2,-L2,0]
B2 = [0,0,1,H2,-L2-L1,0]
B1 = [0,1,0,W1+W2,0,L1+L2]

S6 = [0,-1,0,-(H2-H1),0,L1+L2]
S5 = [0,0,-1,W1,-L1-L2,0]
S4 = [0,-1,0,H1,0,L1+L2]
S3 = [0,-1,0,H1,0,L1]
S2 = [0,-1,0,H1,0,0]
S1 = [0,0,1,0,0,0]

Bi = [B1,B2,B3,B4,B5,B6]
Si = [S1,S2,S3,S4,S5,S6]

M = [[1,0,0,-L2-L1],
    [0,0,-1,-W1-W2],
    [0,1,0,H1-H2],
    [0,0,0,1]]

Tsd = [[0,1,0,-0.6],
    [0,0,-1,0.1],
    [-1,0,0,0.1],
    [0,0,0,1]]


theta0 = 0.1*np.ones(6)#
thetaf = pi/2*np.ones(6)
print "Initial Joint Angles: "
print theta0

trajectoryQ = JointTrajectory(theta0,thetaf,2.0,101,scaleMethod="Quintic")
trajectoryC = JointTrajectory(theta0,thetaf,2.0,101,scaleMethod="Cubic")
thetalistQ = [i[1] for i in trajectoryQ]
thetalistC = [i[1] for i in trajectoryC]
t = np.arange(0.0,2.0, 2.0/(101))
# plt.plot(t, thetalistQ)
# plt.plot(t, thetalistC)
# plt.show()


Xstart = FKinBody(M, Bi, theta0)
Xend = FKinBody(M, Bi, thetaf)

trajectory_Cart = CartesianTrajectory(Xstart, Xend, 2.0, 101, scaleMethod="Quintic")

size = np.shape(trajectory_Cart)
for i in range(size[0]): 
    Tsd = RpToTrans(trajectory_Cart[i][0],trajectory_Cart[i][1])
    theta0 = IKinBody(Bi, M, Tsd, theta0)


if saveData==True:
    size = np.shape(trajectoryQ)
    import csv
    with open('test1.csv','wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'])
        for i in range(size[0]):
            writeList = []
            spamwriter.writerow(np.insert(trajectoryQ[i][:],0,t[i]))













