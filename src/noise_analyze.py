import scipy.io
import numpy as np
poses = np.zeros((7,30),dtype=np.float32)
for i in range(8,39):
	mat = scipy.io.loadmat('/home/xiaoqiangyan/Documents/MyData/Detection/11:21_13_05_2018/{}.mat'.format(i))
	# mat = scipy.io.loadmat('~/Documents/MyData/11:22_13_05_2018/1.mat')
	poses[:,i-8] =mat['poses'].astype(np.float32)
print (np.mean(poses,1))
print (np.cov(poses))