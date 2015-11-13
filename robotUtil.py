import numpy as np
from math import cos, acos, sin, tan, pi, sqrt

def matmult(*x):
    """
    Example Input:
    ([1,2,3], [3,2,1])
    Output:
    10
        """
    return reduce(np.dot, x)


def Magnitude(V):
    """
    Example Input: 
    V = [1,2,3]
    Output:
    3.74165738677
        """
    try:
        length = 0
        for i in range(len(V)):
            length += V[i] ** 2

        return length ** 0.5
    except:
        print 'Input is not a vector'


def Normalise(V):
    """
    Example Input: 
    V = [1,2,3]
    Output:
    [0.2672612419124244, 0.5345224838248488, 0.8017837257372732]
        """
    try:
        length = Magnitude(V)
        if length == 0.0:
            for i in range(len(V)):
                V[i] = 0.0

            return V
        for i in range(len(V)):
            V[i] = V[i] / length

        return V
    except:
        print 'Input is not a vector'


def Det_3(R):
    """
    Example Input: 
    R = [[0, 0,1],
        [1, 0, 0],
        [0, 1, 0]]
    Output:
    1.0
        """
    try:
        return np.linalg.det(R)
    except:
        print 'Matrix is not square'


def RotInv(R):
    """
    Example Input: 
    R = [[0, 0,1],
        [1, 0, 0],
        [0, 1, 0]]
    Output:
    [[0, 1, 0], 
    [0, 0, 1],
    [1, 0, 0]]
        """
    if Det_3(R) * 1.0 >= 0.95 and Det_3(R) * 1.0 <= 1.05:
        Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]]
    if (matmult(R, Rt) == np.eye(3)).all:
        return Rt


def VecToso3(V):
    """
    Example Input: 
    V = [1,2,3]
    Output:
    [[0, -3, 2],
     [3, 0, -1],
     [-2, 1, 0]]
        """
    try:
        if len(V) == 3:
            return [[0, -V[2], V[1]], [V[2], 0, -V[0]], [-V[1], V[0], 0]]
        print 'This vector is the wrong size'
    except:
        print 'This vector cannot be converted to a skew sym matrix'


def so3ToVec(R):
    """
    Example Input: 
    R = [[0, -3, 2],
     [3, 0, -1],
     [-2, 1, 0]]
    Output:
    [1, 2, 3]
        """
    try:
        if R[0][0] == 0 and R[1][1] == 0 and R[2][2] == 0 and R[0][1] == -R[1][0] and R[2][0] == -R[0][2] and R[1][2] == -R[2][1]:
            return [R[2][1], R[0][2], R[1][0]]
        print 'This is not a skew symetric matrix'
    except:
        print 'Input martix is the wrong shape'


def AxisAng3(V):
    """
    Example Input: 
    V = [1,2,3]
    Output:
    ([0.2672612419124244, 0.5345224838248488, 0.8017837257372732], -->unit rotation axis w
     3.7416573867739413) -->theta
        """
    if len(V) == 3:
        theta = Magnitude(V)
        if theta == 0:
            return ([0, 0, 0], 0)
        else:
            return (Normalise(V), theta)
    else:
        print 'This vector is the wrong size'


def MatrixExp3(V):
    """
    Example Input: 
    p = [0*0.524,0.866*0.524,0.5*0.524]
    Output:
    [[ 0.86583049 -0.25017423  0.43330176],
    [ 0.25017423  0.96645615  0.05809795],
    [-0.43330176  0.05809795  0.89937434]]
        """
    if len(V) == 3:
        w, theta = AxisAng3(V)
        return np.eye(3) + matmult(VecToso3(w), np.sin(theta)) + matmult(VecToso3(w), VecToso3(w)) * (1 - np.cos(theta))
    print 'This vector is the wrong size'


def MatrixLog3(R):
    """
    Example Input: 
    R = [[0, 0,1],[1, 0, 0],[0, 1, 0]]
    
    Output:
    [1.2091995761561456, 1.2091995761561456, 1.2091995761561456]
        """
    if len(R) == 3:
        try:
            Rtrace = R[0][0] + R[1][1] + R[2][2]
            theta = np.arccos((Rtrace - 1) / 2.0)
            w = 1 / (2 * np.sin(theta)) * np.array([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
            if any(map(np.isinf, w)) or any(map(np.isnan, w)):
                theta = 0
                w = 3 * [1 / np.sqrt(3)]
            return [w[0] * theta, w[1] * theta, w[2] * theta]
        except:
            print 'Matrix cannot be converted'

    else:
        print 'This matrix is the wrong size'


def RpToTrans(R, p):
    """
    Example Input: 
    R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    p = [1,2,5]
    Output:
    [[1, 0, 0, 1],
     [0, 0, -1, 2],
     [0, 1, 0, 5],
     [0, 0, 0, 1]]
        """
    try:
        return [[R[0][0],
          R[0][1],
          R[0][2],
          p[0]],
         [R[1][0],
          R[1][1],
          R[1][2],
          p[1]],
         [R[2][0],
          R[2][1],
          R[2][2],
          p[2]],
         [0,
          0,
          0,
          1]]
    except:
        print 'Input rotation matrix or position vector are the wrong size'


def TransToRp(T):
    """
    Example Input: 
    T = [[1,0,0,0],
         [0,0,-1,0],
         [0,1,0,3],
         [0,0,0,1]]
    Output:
    ([[1, 0, 0], -->R
     [0, 0, -1], -->R
     [0, 1, 0]], -->R 
    [0, 0, 3]) -->P
        """
    try:
        return ([[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]], [T[0][3], T[1][3], T[2][3]])
    except:
        print 'Input Transformation matrix is the wrong size'


def TransInv(T):
    """
    Example Input: 
    T = [[1,0,0,0],
         [0,0,-1,0],
         [0,1,0,3],
         [0,0,0,1]]
    Output:
    [[1, 0, 0, 0],
     [0, 0, 1, -3],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
        """
    try:
        R, p = TransToRp(T)
        Rt = RotInv(R)
        pt = np.dot(Rt, p)
        Tinv= [[Rt[0][0],Rt[0][1],Rt[0][2],-pt[0]],
          [Rt[1][0],Rt[1][1],Rt[1][2],-pt[1]],
          [Rt[2][0],Rt[2][1],Rt[2][2],-pt[2]],
          [0,0,0,1]]
        return Tinv
    except:
        print 'Input matrix is not an element of SE3'


def VecTose3(V):
    """
    Example Input: 
    V = [1,2,3,4,5,6]
    Output:
    [[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]]
        """
    if len(V) == 6:
        try:
            R = VecToso3([V[0], V[1], V[2]])
            return [[R[0][0],
              R[0][1],
              R[0][2],
              V[3]],
             [R[1][0],
              R[1][1],
              R[1][2],
              V[4]],
             [R[2][0],
              R[2][1],
              R[2][2],
              V[5]],
             [0,
              0,
              0,
              0]]
        except:
            print 'Input vector is not a spatial velocity representation'

    else:
        print 'Input vector is the wrong size'


def se3ToVec(Sp):
    """
    Example Input: 
    Sp =[[-1,0,0,0],  [0,1,0,6],  [0,0,-1,2],  [0,0,0,1]]
    Output:
    [0, 0, 0, 0, 6, 2]
        """
    try:
        return [Sp[2][1],
         Sp[0][2],
         Sp[1][0],
         Sp[0][3],
         Sp[1][3],
         Sp[2][3]]
    except:
        print 'Input is not an element of se3'


def Adjoint(T):
    """
    Example Input: 
    T = [[1,0,0,0], [0,0,-1,0], [0,1,0,3], [0,0,0,1]]
    Output:
    [[1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 3, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, -1],
    [-3, 0, 0, 0, 1, 0]]
        """
    try:
        R, p = TransToRp(T)
        #Rt = np.cross(p,R)
        Rt = np.dot(VecToso3(p), R)
        return [[R[0][0],
          R[0][1],
          R[0][2],
          0,
          0,
          0],
         [R[1][0],
          R[1][1],
          R[1][2],
          0,
          0,
          0],
         [R[2][0],
          R[2][1],
          R[2][2],
          0,
          0,
          0],
         [Rt[0][0],
          Rt[0][1],
          Rt[0][2],
          R[0][0],
          R[0][1],
          R[0][2]],
         [Rt[1][0],
          Rt[1][1],
          Rt[1][2],
          R[1][0],
          R[1][1],
          R[1][2]],
         [Rt[2][0],
          Rt[2][1],
          Rt[2][2],
          R[2][0],
          R[2][1],
          R[2][2]]]
    except:
        print 'Input T is not an element of SE3'


def ScrewToAxis(q, s, h):
    """
    Example Input: 
    q = [3,0,0]
    s = [0,0,1]
    h = 2
    Output:
    [[0], [0], [1], [0], [-3], [2]]
        """
    try:
        sq = np.cross(s, q)
        return [[s[0]],
         [s[1]],
         [s[2]],
         [-sq[0] + h * s[0]],
         [-sq[1] + h * s[1]],
         [-sq[2] + h * s[2]]]
    except:
        print 'Inputs are the wrong size. -->  qE3, sE3, h scaler'


def AxisAng6(V):
    """
    Example Input: 
    V = [1,0,0,1,2,3]
    Output:
    ([1.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1.0) --> First the srew axis and then the theta
        """
    if len(V) == 6:
        theta = Magnitude([V[0], V[1], V[2]])
        if theta == 0:
            theta = Magnitude([V[3], V[4], V[5]])
            return ([0,
              0,
              0,
              V[3] / theta * 1.0,
              V[4] / theta * 1.0,
              V[5] / theta * 1.0], theta)
        else:
            return ([V[0] / theta * 1.0,
              V[1] / theta * 1.0,
              V[2] / theta * 1.0,
              V[3] / theta * 1.0,
              V[4] / theta * 1.0,
              V[5] / theta * 1.0], theta)
    else:
        print 'This vector is the wrong size'


def MatrixExp6(T):
    """
    Example Input: 
    Stheta = [0,(1/(2**0.5)),(1/(2**0.5)),1,2,3]
    Output:
    [[0.54030230586813965, -0.59500983952938591, 0.59500983952938591, 1.1665263416243543],
    [0.59500983952938591, 0.77015115293406988, 0.22984884706593017, 2.4043198644125101],
    [-0.59500983952938591, 0.22984884706593017, 0.77015115293406988, 2.5956801355874903],
    [0, 0, 0, 1]]
        """
    s, th = AxisAng6(T)
    w = [s[0], s[1], s[2]]
    UL = np.eye(3) + matmult(VecToso3(w), np.sin(th)) + matmult(VecToso3(w), VecToso3(w)) * (1 - np.cos(th))
    UR = matmult(np.eye(3), th) + matmult(VecToso3(w), 1 - np.cos(th)) + matmult(VecToso3(w), VecToso3(w)) * (th - np.sin(th))
    UR = np.dot(UR, [s[3], s[4], s[5]])
    return [[UL[0][0],
      UL[0][1],
      UL[0][2],
      UR[0]],
     [UL[1][0],
      UL[1][1],
      UL[1][2],
      UR[1]],
     [UL[2][0],
      UL[2][1],
      UL[2][2],
      UR[2]],
     [0,
      0,
      0,
      1]]


def MatrixLog6(T):
    """
    Example Input: 
    T = [[1,0,0,0], [0,0,-1,0], [0,1,0,3], [0,0,0,1]]
    Output:
    [1.5707963267948966, 0.0, 0.0, 0.0, 2.3561944901923448, 2.3561944901923457]
        """
    R, p = TransToRp(T)

    Rtrace = R[0][0] + R[1][1] + R[2][2]
    if np.linalg.norm(R - np.eye(3))<10e-6:
        w = [0,0,0]
        v = p#Normalise(p)
        th = np.linalg.norm(p)
	return [w[0],w[1],w[2],v[0],v[1],v[2]]
    elif (Rtrace+1) < 10e-6:
        th = pi
        w = MatrixLog3(R)
        G = 1 / th * np.eye(3) - 0.5 * np.asarray(VecToso3(w)) + (1 / th - 1 / tan(th / 2.0) / 2.0) * matmult(VecToso3(w), VecToso3(w))
        v = np.dot(G, p)
    else:
        th = acos((Rtrace - 1) / 2.0)
        w = so3ToVec(1 / (2 * np.sin(th)) * np.subtract(R, RotInv(R)))
        G = 1 / th * np.eye(3) - 0.5 * np.asarray(VecToso3(w)) + (1 / th - 1 / tan(th / 2.0) / 2.0) * matmult(VecToso3(w), VecToso3(w))
        v = np.dot(G, p)

    return [w[0] * th,
     w[1] * th,
     w[2] * th,
     v[0] * th,
     v[1] * th,
     v[2] * th]


def FKinFixed(M, S, th):
    """
    Example Input: 
    M = [[-1,0,0,0], [0,1,0,6], [0,0,-1,2], [0,0,0,1]]
    S = [[0,0,1,4,0,0],[0,0,0,0,1,0],[0,0,-1,-6,0,-0.1]]
    th =[(pi/2.0),3,pi]
    Output:
    [[ -1.14423775e-17   1.00000000e+00   0.00000000e+00  -5.00000000e+00],
    [  1.00000000e+00   1.14423775e-17   0.00000000e+00   4.00000000e+00],
    [ 0.          0.         -1.          1.68584073],
    [ 0.  0.  0.  1.]]
        """
    T = np.eye(4)
    for i in range(len(S)):
        T = np.dot(T, MatrixExp6(np.asarray(S[i]) * th[i]))

    T = np.dot(T, M)
    return T


def FKinBody(M, S, th):
    """
    Example Input: 
    M = [[-1,0,0,0], [0,1,0,6], [0,0,-1,2], [0,0,0,1]]
    Sb = [[0,0,-1,2,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0.1]]
    th =[(pi/2.0),3,pi]
    Output:
    [[ -1.14423775e-17   1.00000000e+00   0.00000000e+00  -5.00000000e+00],
    [  1.00000000e+00   1.14423775e-17   0.00000000e+00   4.00000000e+00],
    [ 0.          0.         -1.          1.68584073],
    [ 0.  0.  0.  1.]]
        """
    T = np.eye(4)
    T = np.dot(M, T)
    for i in range(len(S)):
        T = np.dot(T, MatrixExp6(np.asarray(S[i]) * th[i]))

    return T


def FixedJacobian(Theta, Si, *argv):
    """
    Example inputs:
    Theta = [pi,pi/2,pi/4]
    Si =  [[0,0,-1,2,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0.1]], unit screw axis
    Output:
    No idea yet
    """
    Js = []
    if argv:
        n = argv[0]
    else:
        n = len(Theta)
    for i in range(n):
        if i == 0:
            Js.append(np.asarray(Si[i][:]))# * Theta[i])
        else:
            temp = MatrixExp6(np.asarray(Si[0][:]) * Theta[0])
            for j in range(1, i):
                temp = np.dot(temp, MatrixExp6(np.asarray(Si[j][:]) * Theta[j]))
            Js.append(np.dot(Adjoint(temp), Si[i][:]))

    return np.asarray(Js).T


def BodyJacobian(Theta, Bi, *argv):
    """
    Example inputs:
    Theta = [pi,pi/2,pi/4]
    Bi =  [[0,0,-1,2,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0.1]], unit screw axis
    Output:
    No idea yet
    """
    Jb = []
    if argv:
        n = argv[0]
    else:
        n = len(Theta)
    for i in range(n):
        if i == n-1:
            Jb.append(np.asarray(Bi[i][:]))# * Theta[i])
        else:
            temp = MatrixExp6(-1*np.asarray(Bi[n-1][:]) * Theta[n-1])
            for j in range(n-2, i, -1):
                temp = np.dot(temp, MatrixExp6(-1*np.asarray(Bi[j][:]) * Theta[j]))
            Jb.append(np.dot(Adjoint(temp), Bi[i][:]))
    return np.asarray(Jb).T

def IKinBody(Bi, M, Tsd, theta0, *argv):
    if argv:
        epsilon_v = argv[0]
        epsilon_w = argv[1]
    else:
        # set default tolerance
        epsilon_w = 0.01
        epsilon_v = 0.001

    # initialize parameters
    maxIter = 100
    i = 0
    Tsb = FKinBody( M, Bi, theta0 )
    Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
    wb = [Vb[0],Vb[1],Vb[2]]
    vb = [Vb[3],Vb[4],Vb[5]]
    theta0 = np.asarray(theta0)
    thetaStor = [theta0]
    
    # run the loop
    while (np.linalg.norm(wb) > epsilon_w or np.linalg.norm(vb) > epsilon_v) and i<maxIter:
        Jb = BodyJacobian(theta0, Bi)
        theta0 = theta0 + np.dot( np.linalg.pinv(Jb), Vb )
        i = i + 1
        Tsb = FKinBody( M, Bi, theta0 )
        Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
        wb = [Vb[0],Vb[1],Vb[2]]
        vb = [Vb[3],Vb[4],Vb[5]]
        thetaStor.append(theta0)
    print "Inverse Kinematics wrt Body: "
    print Tsb
    print np.asarray(Tsd)
    print "Iterations: "
    print i
    print "Final Configuration: "
    print theta0
    return thetaStor

def IKinFixed(Si, M, Tsd, theta0, *argv):
    if argv:
        epsilon_v = argv[0]
        epsilon_w = argv[1]
    else:
        # set default tolerance
        epsilon_w = 0.01
        epsilon_v = 0.01

    # initialize parameters
    maxIter = 100
    i = 0
    Tsb = FKinFixed( M, Si, theta0 )
    Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
    Vs = np.dot(Adjoint(Tsb),Vb)
    wb = [Vb[0],Vb[1],Vb[2]]
    vb = [Vb[3],Vb[4],Vb[5]]
    theta0 = np.asarray(theta0)
    thetaStor = [theta0]
    
    # run the loop
    while (np.linalg.norm(wb) > epsilon_w or np.linalg.norm(vb) > epsilon_v) and i<maxIter:
        Js = FixedJacobian(theta0,Si)
        theta0 = theta0 + np.dot( np.linalg.pinv(Js), Vs)
        i = i + 1
        Tsb = FKinFixed( M, Si, theta0 )
        Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
        Vs = np.dot(Adjoint(Tsb),Vb)
        wb = [Vb[0],Vb[1],Vb[2]]
        vb = [Vb[3],Vb[4],Vb[5]]
        thetaStor.append(theta0)
    print "Inverse Kinematics wrt Fixed:"
    print Tsb
    print np.asarray(Tsd)
    print "Iterations: "
    print i
    print "Final Configuration: "
    print theta0
    return thetaStor
