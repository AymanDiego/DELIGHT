import numpy as np
import sys
import matplotlib.pyplot as plt

ifile = sys.argv[1]
# Replace 'your_file.npy' with the path to your file
data = np.load(ifile)

# Display the data
print(data)
print(data[:,0]+data[:,1]+data[:,2]+data[:,3])
print(data.shape)


# 0,1,2,3
# phonon, triplet, UV, IR,
# 4,5,6,7
# before penning quenching
# 8,9,... rest
# integers for cross checks


#print(np.geomspace(10, 1e6, 500))

fig,ax=plt.subplots()
plt.hist(data[:,0],histtype='step',bins=15,label='phonon channel')
plt.hist(data[:,1],histtype='step',bins=15,label='triplet channel')
plt.hist(data[:,2],histtype='step',bins=15,label='UV channel')
plt.hist(data[:,3],histtype='step',bins=15,label='IR channel')
plt.legend()
plt.savefig("/web/aratey/public_html/delight/test.png")

