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
// hello

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
            if np.linalg.norm(R-np.eye(3)) < 1e-6:
                w = [0,0,0]
                theta = 0
            else:
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
        p = [T[0][3], T[1][3], T[2][3]]
        R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]]
        return (R,p)
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
            if theta == 0:
                return ([0,0,0,0,0,0],0)
            else:
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
        epsilon_w = 0.001
        epsilon_v = 0.005

    # initialize parameters
    maxIter = 100
    i = 0
    Tsb = FKinBody( M, Bi, theta0 )
    Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
    wb = [Vb[0],Vb[1],Vb[2]]
    vb = [Vb[3],Vb[4],Vb[5]]
    theta0 = np.asarray(theta0)
    # thetaStor = [theta0]
    
    # run the loop
    while (np.linalg.norm(wb) > epsilon_w or np.linalg.norm(vb) > epsilon_v) and i<maxIter:
        Jb = BodyJacobian(theta0, Bi)
        theta0 = theta0 + np.dot( np.linalg.pinv(Jb), Vb )
        i = i + 1
        Tsb = FKinBody( M, Bi, theta0 )
        Vb = MatrixLog6( np.dot(TransInv(Tsb), Tsd) )
        wb = [Vb[0],Vb[1],Vb[2]]
        vb = [Vb[3],Vb[4],Vb[5]]
        # thetaStor.append(theta0)

    return theta0

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


def CubicTimeScaling(T):
    '''
    Takes total travel time T and returns the corresponding s parameters
    '''
    if T <= 0:
        print "T has to be greater than or equal to 0 "
    else:
        return [0,0,3/(T**2),-2/(T**3)]

def QuinticTimeScaling(T):
    '''
    Takes total travel time T and returns the corresponding s parameters
    '''
    if T <= 0:
        print "T has to be greater than or equal to 0 "
    else:
        return [0,0,0,10/(T**3),-15/(T**4),6/(T**5)]

def JointTrajectory(thetaStart, thetaEnd, T, N, scaleMethod="Cubic"):
    '''
    Function takes initial joint thetastart and end thetaend, total elapsed time T,
    number of discrete points N (where N >= 2) and time scaling method(default is cubic)

    Note: thetaStart and thetaEnd are assumed to be a numpy array
    T is a floating point and N is an integer
    ''' 
    if scaleMethod == "Cubic":
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
        ds = lambda t: a[1] + 2*a[2]*t + 3*a[3]*t**2
        dds = lambda t: 2*a[2] + 6*a[3]*t
    elif scaleMethod == "Quintic":
        a = QuinticTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        ds = lambda t: a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
        dds = lambda t: 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
    else:
        print " Scaling method unknown: defaulting to Cubic"
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
        ds = lambda t: a[1] + 2*a[2]*t + 3*a[3]*t**2
        dds = lambda t: 2*a[2] + 6*a[3]*t

    
    t0 = 0.
    ts = T/(N-1)
    trajectory = {'s': [], 'ds': [], 'dds': [], 'T': T, 'dt':ts, 'N':N}
    trajectory['s'].append(thetaStart)
    trajectory['ds'].append(np.zeros(len(thetaStart)))
    trajectory['dds'].append(np.zeros(len(thetaStart)))
    #np.insert(trajectory[-1],0,t0)
    for i in range(1,N):
        t0 += ts
        thetaS = thetaStart + s(t0)*(thetaEnd - thetaStart)
        dthetaS = ds(t0)*(thetaEnd - thetaStart)
        ddthetaS = dds(t0)*(thetaEnd - thetaStart)
        trajectory['s'].append(thetaS)
        trajectory['ds'].append(dthetaS)
        trajectory['dds'].append(ddthetaS)
     #   np.insert(trajectory[-1],0,t0)

    return trajectory

def ScrewTrajectory(Xstart, Xend, T, N, scaleMethod="Cubic"):
    '''
    Function takes initial endeff config and end endeff config, total elapsed time T,
    number of discrete points N (where N >= 2) and time scaling method(default is cubic)
    Returns: List of SE(3) matrices of configs
     ''' 
    if scaleMethod == "Cubic":
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
    elif scaleMethod == "Quintic":
        a = QuinticTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    else:
        print " Scaling method unknown: defaulting to Cubic"
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3

    t0 = 0.
    ts = T/(N-1)
    trajectory = []
    trajectory.append(np.asarray(Xstart))
    for i in range(1,N):
        t0 += ts
        Xn = np.dot(Xstart, MatrixExp6( np.asarray(MatrixLog6(  np.dot(TransInv(Xstart),Xend)  ))*s(t0) ))
        trajectory.append(Xn)
    return trajectory

def CartesianTrajectory(Xstart, Xend, T, N, scaleMethod="Cubic"):
    if scaleMethod == "Cubic":
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
    elif scaleMethod == "Quintic":
        a = QuinticTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    else:
        print " Scaling method unknown: defaulting to Cubic"
        a = CubicTimeScaling(T)
        s = lambda t: a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3

    Rstart,pstart = TransToRp(Xstart)
    Rend,pend = TransToRp(Xend)
    t0 = 0.
    ts = T/(N-1)
    trajectory = []
    trajectory.append([np.asarray(Rstart),np.asarray(pstart)])

    for i in range(1,N):
        t0 += ts
        pn = np.asarray(pstart) + s(t0)*(np.asarray(pend) - np.asarray(pstart))
        Rn = np.dot(Rstart,MatrixExp3( np.asarray(MatrixLog3( np.dot(RotInv(Rstart),Rend)))*s(t0)  ))
        trajectory.append([Rn, pn])

    return trajectory

def LieBracket(V1,V2):
    omegahat = VecToso3(V1[0:3])
    vhat = VecToso3(V1[3:6])
    Va = np.dot(omegahat, V2[0:3])
    Vb = np.asarray(np.dot(vhat, V2[0:3])) + np.asarray(np.dot(omegahat, V2[3:6]))
    return np.array([Va[0],Va[1],Va[2],Vb[0],Vb[1],Vb[2]])

def adT(V,F):

    omegahat = VecToso3(V[0:3])
    vhat = VecToso3(V[3:6])
    Sa = -np.dot(omegahat, F[0:3]) - np.dot(vhat,F[3:6])
    Sb = -np.dot(omegahat, F[3:6])
    return np.array([Sa[0],Sa[1],Sa[2],Sb[0],Sb[1],Sb[2]])

def InverseDynamics(theta, thetadot, thetaddot, robotDisc, V0dot=[0,0,0,0,0,9.81], Ftip=[0,0,0,0,0,0], V0=[0,0,0,0,0,0]):
    ''' Inputs: initial joint variables
        default gravity, force at tip
        discription of robot as a dictionary of M, G, and screw axis S wrt base frame
        Output: joint torques at that instant of time
    '''

    Vidot = [V0dot]
    Vi = [V0]
    Mi = robotDisc['Mi']
    Mii = robotDisc['Mii']
    Si = robotDisc['Si']
    Gi = robotDisc['Gi']
    Ai = []
    Tii = []
    n = len(theta)
    for i in range(n):

        Ai.append(np.dot(Adjoint(TransInv(Mi[i])),Si[i])) 
        
        Tii.append(np.dot(Mii[i],MatrixExp6(np.dot(Ai[-1],theta[i]))))
        
        Vi.append(np.dot(Adjoint(TransInv(Tii[-1])),Vi[-1]) + np.dot(Ai[-1],thetadot[i]))
        
        # note: because of the append to Vi and Ai, the index below changes to i
        # instead of i-1

        Vidot.append(np.dot(Adjoint(TransInv(Tii[-1])),Vidot[-1]) +\
            np.dot(LieBracket(Vi[-1], Ai[-1]), thetadot[i]) + np.dot(Ai[-1], thetaddot[i]))

    taui = []
    # hard code the Ti, i+1 e-e frame
    w = MatrixExp3([-np.pi/2,0,0])
    p = [0,0.0823,0]
    Tii.append(RpToTrans(w,p))
    Fi = Ftip # set the initial force for the backwards iterations
    # backwards integration
    for i in reversed(range(n)):
        Fi = np.dot(np.transpose(Adjoint(TransInv(Tii[i+1]))), Fi) +\
            np.dot(Gi[i+1], Vidot[i+1]) - adT(Vi[i+1], np.dot(Gi[i+1],Vi[i+1]))
        taui.append(np.dot(Fi, Ai[i]))
    taui = taui[::-1]
    return taui

def InertiaMatrix(theta, robotDisc):
    thetadot = np.zeros(len(theta))
    thetaddot = np.zeros(len(theta))
    M_theta = []
    for i in range(len(theta)):
        M_thetai = np.zeros(len(theta))
        M_thetai[i] = 1
        M_theta.append(InverseDynamics(theta, thetadot, M_thetai,robotDisc, V0dot=[0,0,0,0,0,0]))

    return np.transpose(M_theta)

def CoriolisForces(theta,thetadot,robotDisc):
    c = InverseDynamics(theta, thetadot, np.zeros(len(theta)), robotDisc, V0dot=[0,0,0,0,0,0]) 
    return c

def GravityForces(theta, g, robotDisc):
    thetadot = np.zeros(len(theta))
    thetaddot = np.zeros(len(theta))
    g_theta = InverseDynamics(theta, thetadot, thetaddot, robotDisc, V0dot=[0,0,0,0,0,g])
    return g_theta

def EndEffectorForces(theta, Ftip, robotDisc):
    thetadot = np.zeros(len(theta))
    thetaddot = np.zeros(len(theta))
    eef = InverseDynamics(theta, thetadot, thetaddot, robotDisc, Ftip=Ftip, V0dot=[0,0,0,0,0,0])
    return eef

def ForwardDynamics(theta, thetadot, tau, robotDisc, g=9.81, Ftip=[0,0,0,0,0,0]):

    b = np.asarray(tau) - np.asarray(CoriolisForces(theta, thetadot, robotDisc)) -\
            np.asarray(GravityForces(theta, g, robotDisc)) -\
            np.asarray(EndEffectorForces(theta, Ftip, robotDisc))
    A = InertiaMatrix(theta, robotDisc)

    return np.dot(np.linalg.pinv(A),b)

def EulerStep(theta, thetadot, thetaddot, dt):

    thetan = np.asarray(theta) + np.dot(thetadot,dt)
    thetadotn = np.asarray(thetadot) + np.dot(thetaddot, dt)
    return (thetan, thetadotn)

def InverseDynamicsTrajectory(trajectory, robotDisc, Ftip=[0,0,0,0,0,0]):
    '''
    Input: trajectory should be a dictionary with s, ds, dds, dt, and final time T
    Output: a list of torque forces as a function of time
    '''
    tau = []
    for i in range(trajectory['N']):
        tau.append(InverseDynamics(trajectory['s'][i], trajectory['ds'][i], trajectory['dds'][i], robotDisc, Ftip=Ftip))

    return tau

def ForwardDynamicsTrajectory(tau, theta0, thetadot0, dt,  robotDisc):
    theta = []
    # theta.append(theta0)
    for i in range(len(tau)):
        thetaddot = ForwardDynamics(theta0, thetadot0, tau[i], robotDisc)
        theta0, thetadot0 =  EulerStep(theta0, thetadot0, thetaddot, dt)
        theta.append(theta0)
    return theta




