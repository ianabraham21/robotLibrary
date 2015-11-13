from robotUtil import *
from math import pi
import numpy as np

saveData = False

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


theta0 = 0.01*np.random.randn(6)#
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
    import csv
    with open('test1.csv','wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'])
    for i in range(size[0]):
        spamwriter.writerow(np.insert(Theta[i][:],0,i*0.1))
