import cv2
import numpy as np


A=cv2.imread('tomatoes.jpeg')#Read image
B=cv2.cvtColor(A,cv2.COLOR_BGR2RGB) #Investment channel
B=cv2.cvtColor(B,cv2.COLOR_RGB2HSV) #Image to hsv conversion

H=B[:,:,0]
S=B[:,:,1]
V=B[:,:,2]

maskrojo=((H>0) & (H<25)) |  ((H>165) & (H<180)) &  ((S>80) & (S<255))  &  ((V>80) & (V<255))#Boolean condition was created, for true or false
maskrojo=maskrojo*255  #Must be passed to a numeric value
maskrojo=maskrojo.astype(np.uint8) #8 bits format

maskverde=((H>30) & (H<45))&  ((S>10) & (S<255))  &  ((V>65) & (V<255))    
maskverde=maskverde*255  
maskverde=maskverde.astype(np.uint8) 

res =cv2.bitwise_and(A,A,mask=maskrojo) 
res2 =cv2.bitwise_and(A,A,mask=maskverde) 

#grayscale conversion
B=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)  
C=cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY) 


#Circle hough transform
circle=cv2.HoughCircles(B,cv2.HOUGH_GRADIENT,1,41,param1=50,param2=30,minRadius=2,maxRadius=40)
circle=np.uint16(np.around(circle))
for i in circle[0,:]:
    #draw circle
    cv2.circle(A,(i[0],i[1]),i[2],(0,255,0),2)
     #draw point
    cv2.circle(A,(i[0],i[1]),2,(0,0,255),3)

circle2=cv2.HoughCircles(C,cv2.HOUGH_GRADIENT,1,41,param1=50,param2=30,minRadius=9,maxRadius=40)
circle2=np.uint16(np.around(circle2))
for i in circle2[0,:]:
    cv2.circle(A,(i[0],i[1]),i[2],(0,0,0),2)
    cv2.circle(A,(i[0],i[1]),2,(0,0,255),3)


print('Number of red tomatoes:',len(circle[0,:]))
print('Number of green tomatoes:',len(circle2[0,:]))
print('Total number of tomatoes:',len(circle[0,:])+len(circle2[0,:]))

result= 255 * np.ones((210,150,3), dtype = np.uint8)

cv2.circle(result, (30,30), 15, (0,0,255), -1)
cv2.circle(result, (30,70), 15, (0,255,0), -1)
cv2.putText(result,str(len(circle[0,:])),(65,40), 1, 2,(0,0,0),2)
cv2.putText(result,str(len(circle2[0,:])),(65,80), 1, 2,(0,0,0),2)
totalCnts = len(circle[0,:]) + len(circle[0,:])
cv2.putText(result,'Total: '+str(totalCnts),(0,120), 1, 2,(0,0,0),2)
cv2.imshow('t',result)

cv2.imshow('Tomato_detection',A) #Print
cv2.imwrite("Savephoto.png",A) #Save
cv2.imshow('Tomato red mask',res) #Print
cv2.imshow('Tomato green mask',res2) #Print
cv2.waitKey(0)
cv2.destroyAllWindows()

