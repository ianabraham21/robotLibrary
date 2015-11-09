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
    print Tsd
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
        #Jb = np.dot(Adjoint(TransInv(Tsb)),Js)
        #print "Tsb="
        #print Tsb
        #print "Tsb_inv="
        #print np.asarray(TransInv(Tsb))
        #print "Adj= "
        #print np.asarray(Adjoint(TransInv(Tsb)))
        #print "Js= "
        #print Js
        #print "Jb="
        #print Jb
        #theta0 = theta0 + np.dot( np.linalg.pinv(Jb), Vb )
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