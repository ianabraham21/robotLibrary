from robotUtil import *
import numpy as np
import matplotlib.pyplot as plt
'''
File provides the ur5 configurations and screw axis
'''
def main():
    M1 = [[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0.089159],
        [0,0,0,1]]

    M2 = [[0,0,1,0.28],
        [0,1,0,0.13585],
        [-1,0,0,0.089159],
        [0,0,0,1]] 

    M3 = [[0,0,1,0.675],
        [0,1,0,0.01615],
        [-1,0,0,0.089159],
        [0,0,0,1]]

    M4 = [[-1,0,0,0.81725],
        [0,1,0,0.01615],
        [0,0,-1,0.089159],
        [0,0,0,1]]

    M5 = [[-1,0,0,0.81725],
        [0,1,0,0.10915],
        [0,0,-1,0.089159],
        [0,0,0,1]]

    M6 = [[-1,0,0,0.81725],
        [0,1,0,0.10915],
        [0,0,-1,-0.005491],
        [0,0,0,1]]

    Mi = [M1,M2,M3,M4,M5,M6]

    M01 = M1
    M12 = np.dot(TransInv(M1),M2)
    M23 = np.dot(TransInv(M2),M3)
    M34 = np.dot(TransInv(M3),M4)
    M45 = np.dot(TransInv(M4),M5)
    M56 = np.dot(TransInv(M5),M6)

    Mii = [M01,M12,M23,M34,M45,M56]

    S1 = [0,0,1,0,0,0]
    S2 = [0,1,0,-0.089159,0,0]
    S3 = [0,1,0,-0.089159,0,0.425]
    S4 = [0,1,0,-0.089159,0,0.81725]
    S5 = [0,0,-1,-0.10915,0.81725,0]
    S6 = [0,1,0,0.005491,0,0.81725]
    Si = [S1,S2,S3,S4,S5,S6]

    m0 = 4.0
    m1 = 3.7
    m2 = 8.393
    m3 = 2.275
    m4 = 1.219
    m5 = 1.219
    m6 = 0.1879

    G0 = np.diag([0.00443333156,0.00443333156,0.0072,m0,m0,m0])
    G1 = np.diag([0.0102674,0.0102674,0.00666,m1,m1,m1])
    G2 = np.diag([0.22689,0.22689,0.0151074,m2,m2,m2])
    G3 = np.diag([0.04944,0.04944,0.004095,m3,m3,m3])
    G4 = np.diag([0.11117,0.11117,0.21942,m4,m4,m4])
    G5 = np.diag([0.11117,0.11117,0.21942,m5,m5,m5])
    G6 = np.diag([0.01713,0.01713,0.033822,m6,m6,m6])

    Gi = [G0,G1,G2,G3,G4,G5,G6]

    robotDisc = {'Mi': Mi, 'Mii': Mii, 'Si': Si, 'Gi': Gi}

    theta = np.zeros(6)
    thetadot = np.zeros(6)
    thetaddot = np.zeros(6)

    thetadot[2] = 1
    # thetaddot[5] = 0

    tau = InverseDynamics(theta, thetadot, thetaddot, robotDisc)

    # thetaForward = ForwardDynamics(theta, thetadot, tau, robotDisc)
    print theta, thetadot
    print "CoriolisForces: \t", CoriolisForces(theta, thetadot, robotDisc)
    print "again \t", CoriolisForces(theta, thetadot, robotDisc)

    for i in range(10):
        # CoriolisForces(theta, thetadot, robotDisc)
        print i, "\t", CoriolisForces(theta, thetadot, robotDisc)

    np.set_printoptions(precision=3)
    print "inertia"
    print InertiaMatrix([0,0,0,0,0,0], robotDisc)


    # print "coriolis"
    # print CoriolisForces([0,0,0,0,0,0], [0,0,0,0,0,0], robotDisc)

    # print "Gravity Forces"
    # print GravityForces([0,0,0,0,0,0], 9.81, robotDisc)

    # print "E-E Force"
    # print EndEffectorForces([0,0,0,0,0,0],[0,0,0,0,0,0], robotDisc)


    theta0 = 0.1*np.ones(6)#
    theta0dot = np.zeros(6)
    thetaf = pi/2*np.ones(6)
    trajectoryQ = JointTrajectory(theta0,thetaf,1.0,1001,scaleMethod="Quintic")
    trajectoryC = JointTrajectory(theta0,thetaf,1.0,1001,scaleMethod="Cubic")

    tau = InverseDynamicsTrajectory(trajectoryQ, robotDisc)

    t = np.arange(0.0,1.0, 1.0/(1001))

    # for j in range(len(tau[0])):
    #     taui = []
    #     taui = [ti[j] for ti in tau]
    #     plt.plot(t, taui,label='%d'%j)
    # plt.legend(loc="upper left")
    # plt.show()

    trajN = ForwardDynamicsTrajectory(tau, theta0, theta0dot,trajectoryQ['dt'], robotDisc)
    print np.shape(trajN)
    for j in range(len(trajN[0])):
        taui = []
        taui = [ti[j] for ti in trajN]
        plt.plot(t, taui,label='%d'%j)
    plt.legend(loc="upper left")
    plt.show()  

main()