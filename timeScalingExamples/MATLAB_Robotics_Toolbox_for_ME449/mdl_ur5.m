%MDL_UR5 Create model of Barrett WAM manipulator
%
% MDL_UR5 is a script that creates the workspace variable ur5 which
% describes the kinematic and dynamic characteristics of a 6-DOF
% Universal Robots UR5 manipulator using standard DH conventions. 
%
%
% Notes::
% - SI units are used.
%
% Reference::
% - "http://rsewiki.elektro.dtu.dk/index.php/UR5#Denavit-Hartenberg_parameters",
%   Last accessed Oct. 2015.
%
% See also SerialRevolute, mdl_puma560akb, mdl_stanford, 6DOF.

% MODEL: Universal, UR5, standard_DH

clear L

%##########################################################
% By default, all joints are revolute joints.  The model uses the standard
% DH convention.  See RTB documentation for more details.

%            theta    d         a        alpha
L(1) = Link([ 0       0.089159  0        pi/2]);
L(2) = Link([ 0       0        -0.425    0]);
L(3) = Link([ 0       0        -0.39225  0]);
L(4) = Link([ 0       0.10915   0        pi/2]);
L(5) = Link([ 0       0.09465   0       -pi/2]);
L(6) = Link([ 0       0.0823    0        0]);

%##########################################################

ur5 = SerialLink(L, 'name', 'UR5');

%##########################################################

clear L