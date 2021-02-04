import cv2
#opencv documentation
#https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale

Face_img = "face.jpg" #Declaring the img to the one located in your python file
Face_Classifier = "Face.xml" #Face classifier xml file-this xml will check our image and check it against the data and if it passed then it will be classed as a face

#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 

FaceCheck = cv2.imread(Face_img)#open cv will take the image and read the pixel data which we can then use later imread = image read
Grayscale = cv2.cvtColor(FaceCheck,cv2.COLOR_BGR2GRAY)#The reason to convert the file to gray scale is because it uses less data and when scanning the image it can be
#alot easier to detect

CheckFace = cv2.CascadeClassifier(Face_Classifier)#the xml that I am using is in a cascade format which is why its a cascade classifier
#this is just used now to check the image for any faces

FaceDetector = CheckFace.detectMultiScale(Grayscale)#this will apply the face classifier to the image and then will give out  coordinates for each face found
#if i were to print this then it would give the coordinates of the face

for(X, Y,W, H) in FaceDetector:#This will take the coordinates given in face tracker and then from that it will draw a rectangle to  the face position
    cv2.rectangle(FaceCheck, (X, Y), (X+W, Y+H), (255, 0, 0), 2) #This is drawing the rectangle from the given coordinates


cv2.imshow('Face detection',FaceCheck)#opens the file that we just made and calls the window "Face detection"
cv2.waitKey()#stops the window from autoclosing-click a key to close
