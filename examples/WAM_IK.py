from robotUtil import *
from math import pi
import numpy as np

saveData = False

# MACHINE CONSTANTS
L1 = 550*1e-3
L2 = 300*1e-3
L3 = 60*1e-3
W1 = 45*1e-3

B7 = [0,0,1,0,0,0]
B6 = [0,1,0,L3,0,0]
B5 = [0,0,1,0,0,0]
B4 = [0,1,0,L2+L3,0,W1]
B3 = [0,0,1,0,0,0]
B2 = [0,1,0,L1+L2+L3,0,0]
B1 = [0,0,1,0,0,0]

S7 = [0,0,1,0,0,0]
S6 = [0,1,0,-L1-L2,0,0]
S5 = [0,0,1,0,0,0]
S4 = [0,1,0,-L1,0,W1]
S3 = [0,0,1,0,0,0]
S2 = [0,1,0,0,0,0]
S1 = [0,0,1,0,0,0]

Bi = [B1,B2,B3,B4,B5,B6,B7]
Si = [S1,S2,S3,S4,S5,S6,S7]

M = [[1,0,0,0],
    [0,1,0,0],
    [0,0,1,L1+L2+L3],
    [0,0,0,1]]

Tsd = [[1,0,0,0.4],
    [0,1,0,0],
    [0,0,1,0.4],
    [0,0,0,1]]


theta0 = 0.1*np.random.randn(7)#
#theta0 = [0.001,0.001,0.001,0.001,0.001,0.001]
# this one below works well
#theta0 = [-0.00263374, -0.00845871, -0.00511406, -0.00517887,  0.00351958,  0.00924158]
# also works well with IK
#theta0 = [-0.00686114  0.00922843  0.00602058 -0.00167514  0.00557784  0.00661908]
print "Initial Joint Angles: "
print theta0
Theta = IKinBody(Bi, M, Tsd, theta0)
Theta = IKinFixed(Si, M, Tsd, theta0)
if saveData==True:
    size = np.shape(Theta)
    import csv
    with open('WAM_test.csv','wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #spamwriter.writerow(['time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'])
    for i in range(size[0]):
        spamwriter.writerow(np.insert(Theta[i][:],0,i*0.1))
