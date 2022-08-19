import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# make custom colormap
# purple mountian_majesty
def purple_mountain_majesty(type = 'smooth',plot = False):
    my_cmap  = np.ones((256,4))
    c1 = matplotlib.colors.to_rgb("#7C5AA3")
    c2 = matplotlib.colors.to_rgb("#9577B7")
    c3 = matplotlib.colors.to_rgb("#A493BE")
    c4 = matplotlib.colors.to_rgb("#EACDB4")
    c5 = matplotlib.colors.to_rgb("#E4A780")
    c6 = matplotlib.colors.to_rgb("#E4734A")

    all_cs = np.vstack((c1,c2,c3,c4,c5,c6))
    my_cmap[0:43,0:3] = c1
    my_cmap[43:86,0:3] = c2
    my_cmap[86:129,0:3] = c3
    my_cmap[129:172,0:3] = c4
    my_cmap[172:215,0:3] = c5
    my_cmap[215:256,0:3] = c6

    new_cmap = ListedColormap(my_cmap)

    my_cmap_2 = np.ones((256,4))
    for j in range(0,all_cs.shape[0]-1):
        for i in range(0,3):
            if j < all_cs.shape[0]-2:
                my_cmap_2[j*(256//(all_cs.shape[0]-1)):(j+1)*(256//(all_cs.shape[0]-1)),i] = np.linspace(all_cs[j,i],all_cs[j+1,i],256//(all_cs.shape[0]-1))
            else:
                my_cmap_2[j*(256//(all_cs.shape[0]-1)):256,i] = np.linspace(all_cs[j,i],all_cs[j+1,i],256-j*(256//(all_cs.shape[0]-1)))

    new_cmap_2 = ListedColormap(my_cmap_2)

    if plot:
        plt.figure(1)
        plt.title("purple_mountain_majesty_sequential")
        plt.imshow(np.ones((100,100))*np.linspace(0,1,100),cmap = new_cmap)
        plt.figure(2)
        for i in range(0,all_cs.shape[0]):
            result = plt.hist(np.random.randn(200)+i*2,color = all_cs[i,:],histtype = 'stepfilled',label = "C"+str(i),zorder = all_cs.shape[0]*2-2*i,alpha = 0.9,bins=15)
            bin_centers = ((result[1]+np.roll(result[1],1))/2.)[1:len(result[1])]
            plt.plot(np.hstack((result[1],result[1][-1]+(result[1][-1]-result[1][-2]))),np.hstack((0,result[0],0)),ls='steps',color = 'k',zorder = all_cs.shape[0]*2-1-2*i)
        plt.legend()
        plt.title("Natural for histograms")
        plt.figure(3)
        plt.title("purple_mountain_majesty")
        plt.imshow(np.ones((100,100))*np.linspace(0,1,100),cmap = new_cmap_2)
        plt.show()

    if type == 'sequential':
        return new_cmap
    elif type ==list:
        return all_cs
    else:
        return new_cmap_2

