import numpy as np
import math
import scipy.io

def rot_quaternion(rot):
	'''
	tranform rotation to quaternion using https://blog.csdn.net/lql0716/article/details/72597719
	'''
	condition = rot[0,0]+rot[1,1]+rot[2,2]+1
	# print condition
	if (condition-0.0)>1e-4:
		q0 = 0.5*np.sqrt(condition)
		q1 = (rot[2,1]-rot[1,2])/(4.0*q0)
		q2 = (rot[0,2] - rot[2,0])/(4.0*q0)
		q3 = (rot[1,0] - rot[0,1])/(4.0*q0)
	elif max(rot[0,0],rot[1,1],rot[2,2]) == rot[0,0]:
		t = np.sqrt(1+rot[0,0]-rot[1,1]-rot[2,2])
		q0= (rot[2,1]-rot[1,2])/t
		q1= t/4
		q2= (rot[0,2] + rot[2,0])/t
		q3= (rot[0,1] + rot[1,0])/t
	elif max(rot[0,0],rot[1,1],rot[2,2]) == rot[1,1]:
		t = np.sqrt(1-rot[0,0]+rot[1,1]-rot[2,2])
		q0 = (rot[0,2]-rot[2,0])/t
		q1 = (rot[0,1]+rot[1,0])/t
		q2 = t/4
		q3 = (rot[2,1]+rot[1,2])/t
	elif max(rot[0,0],rot[1,1],rot[2,2]) == rot[2,2]:
		t = np.sqrt(1-rot[0,0]-rot[1,1]+rot[2,2])
		q0 = (rot[1,0]-rot[0,1])/t
		q1 = (rot[0,2]+rot[2,0])/t
		q2 = (rot[1,2]-rot[2,1])/t
		q3 = t/4
	q = (q0,q1,q2,q3)
	q = np.array(q)
	return q

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """

    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def quaternion_Euler(rot):
	"""tranfer the rotation from quanternion to Euler"""

	q0, q1, q2, q3 = rot
	theta_x = math.atan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2))* 180 / np.pi
	theta_y = math.asin(2*(q0*q2-q1*q3)) * 180 / np.pi
	theta_z = math.atan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))* 180 / np.pi
	theta = (theta_x, theta_y, theta_z)
	return theta

q0 = 0.46150756
q1 = 0.76479465
q2 = 0.01802805
q3 = 0.02581866
q = [q0, q1, q2, q3]
euler = quaternion_Euler(q) 
euler = np.array(euler)


# R0 = [q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2]
# R1 = [2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1]
# R2 = [2*q1*q3-2*q0*q2, 2*q2*q3-2*q0*q1, q0**2-q1**2-q2**2+q3**2]
# R = np.vstack((R0,R1,R2))
# print R

gt_pose = scipy.io.loadmat('/data/YCB-dataset/YCB_Video_Dataset/data/0001/000001-meta.mat')['poses'].astype(np.float64)
for idx in range(gt_pose.shape[2]):
	gt_rot = gt_pose[0:3,0:3,idx]
	#print (gt_rot)
	gt_q_0 = rot_quaternion(gt_rot)
	R = np.zeros((4,4),dtype = np.float64)
	R[0:3,0:3] = gt_rot
	R[3,3] = 1
	gt_q_1 = quaternion_from_matrix(R)
	print (gt_q_0)
	print (gt_q_1)
# gt_euler = quaternion_Euler(gt_q)
# gt_euler = np.array(gt_euler)

# print (euler)
# print (gt_euler)
# # Error_q = np.arccos((np.trace(q*np.linalg.inv(gt_q))-1)/2)
# # print ('Error of Rotation in Quaternion is: ',Error_q)
# Error_rot = np.arccos((np.trace(R*np.linalg.inv(gt_rot))-1)/2)
# print ('Error of Rotation in matrix is: ',Error_rot)
# Error_euler = np.arccos((np.trace(euler*np.linalg.inv(gt_euler))-1)/2)
# print ('Error of Rotation in Euler is: ',Error_euler)