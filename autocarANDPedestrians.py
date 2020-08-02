from cv2 import cv2


#our Video
#video = cv2.VideoCapture('TeslaDashcamClears.mp4')
video = cv2.VideoCapture('PedestriansCompilation.mp4')

#pre-trained car classifer
car_tracker = cv2.CascadeClassifier('cardetector.xml')
pedestrian_tracker = cv2.CascadeClassifier('Pedestrians.xml')

#run forever until cars stops..
while True:
    #read the current frame
    (read_successful, frame) = video.read()

    #safe codng..
    if read_successful:
        #convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #Detect car and pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    #draw rectanle around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
    #draw rectanle around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    

    

    #display image with car spoted
    cv2.imshow('My Car Detector', frame)

    #don't autoclose (wait here in the code and listen for a key pree)
    key = cv2.waitKey(1)


    #stop if Q is pressed
    if key==81 or key==113:
        break

#release the VideoCapture object
video.release()
 

print("Code Completed")