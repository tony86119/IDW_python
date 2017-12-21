# -*- coding: UTF-8 -*-  
import numpy as np
import matplotlib.pyplot as plt
import idw
import sys
from geopy.distance import vincenty


def main(): 
    #filee=open(sys.argv[1],'r')
    #print sys.argv[1]
    with open(sys.argv[1],'r') as f:
        arraysX=[map( float,line.split(' ')) for line in f]
        #arrays2 = [map(float, line.split('\n')) for line in f]
    X1=np.array(arraysX)
    
#找出Bounding box
    biggestLNG=-180
    smallestLNG=180
    biggestLAT=-90
    smallestLAT=90


    for i in range(X1.shape[0]):
        LNG=X1[i][0]
        
        if i == 0:
            biggestLNG=LNG
            smallestLNG=LNG
        elif LNG>biggestLNG:
            biggestLNG=LNG
        elif LNG<biggestLNG:
            if LNG<smallestLNG:
                smallestLNG=LNG
        LAT=X1[i][1]
        if i==0:
            biggestLAT=LAT
            smallestLAT=LAT
        elif LAT>biggestLAT:
            biggestLAT=LAT
        elif LAT<biggestLAT:
            if LAT<smallestLAT:
                smallestLAT=LAT

#求出經度距離與緯度距離
    leftdown=(smallestLNG-0.005,smallestLAT-0.005)
    lefttop=(smallestLNG-0.005,biggestLAT+0.005)
    righttop=(biggestLNG+0.005,biggestLAT+0.005)
    distanceLNG=vincenty(righttop, lefttop).kilometers
    distanceLAT=vincenty(leftdown, lefttop).kilometers
    #print(X1)
    #print (distanceLNG)
    #print (distanceLAT)
#利用cellsize算出網格數量 cellsize單位公里
    cellsize=float(sys.argv[3])
    cellCountLAT=round(distanceLAT/cellsize)
    cellCountLNG=round(distanceLNG/cellsize)
    f.close()
    #X1=np.array([[47.11285,7.222309],[47.085272,7.20377],[47.092285,7.156734],[47.13294,7.220936],[47.088311,7.128925],[47.124765,7.234669],[47.055107,7.07159]])
    with open(sys.argv[2],'r') as f:
        arraysZ=map( float,f.read().split(' ')) 
        #arrays2 = [map(float, line.split('\n')) for line in f]
    z1=np.array(arraysZ)
    #print(arraysZ)
    f.close()

    print("loading successfully")
#Z1就是Z值
#z1 = func(X1[:,0], X1[:,1])
    #z1=np.array([33,25,8,12,9,39,6])
    # 'train'
    idw_tree = idw.tree(X1, z1)

    # 'X軸的起點、終點與區間'
    spacing = np.linspace(smallestLNG-0.005, biggestLNG+0.005, cellCountLNG)
    # 'Y軸的起點、終點與區間'
    spacingY= np.linspace(smallestLAT-0.005,biggestLAT+0.005,cellCountLAT)
    X2 = np.meshgrid(spacing, spacingY)
    grid_shape = X2[0].shape
    X2 = np.reshape(X2, (2,-1)).T
    z2 =idw_tree(X2)
    result=z2.reshape(grid_shape)
    #print (z2.reshape(grid_shape))
    np.savetxt('result.txt', result,fmt='%.8f', delimiter=',')
    f=open('./metadata.txt','w')
    f.write("leftdown: "+str(leftdown)+" ,righttop: "+str(righttop)+" ,cellCountLNG: "+str(cellCountLNG)+" ,cellCountLAT: "+str(cellCountLAT))
    f.close()
    print ("saving successfully")

    # plot
    fig, (ax2, ax3) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,3))
    #ax1.contourf(spacing, spacingY, func(*np.meshgrid(spacing, spacingY)))
    #ax1.set_title('Ground truth')
    ax2.scatter(X1[:,0], X1[:,1], c=z1, linewidths=0)
    ax2.set_title('Samples')
    ax3.contourf(spacing, spacingY, z2.reshape(grid_shape))
    ax3.set_title('Reconstruction')
    plt.show()


if __name__ == "__main__":
    main()