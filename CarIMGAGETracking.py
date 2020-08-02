from cv2 import cv2


#our Image
img_file = 'carImage2.webp'

#pre-trained car classifer
classifier_file = 'cardetector.xml'

#create opencv image
img = cv2.imread(img_file)

#conver to grayscale 
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Detect car classsifer
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectanle around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)


#display image with car spoted
cv2.imshow('My Car Detector', img)

#don't autoclose (wait here in the code and listen for a key pree)
cv2.waitKey()






print("Code Completed")