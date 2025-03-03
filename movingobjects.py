#In task two, i used centroid to track the person. centroid is the center of the detected object surrounded by rectangle. 
#i tracked the person by taking the difference of 2 nearest centroid and put the same id to the nearest detected centroid.
#In task two for choosing the 3 nearest person from the video, i took the measurement of the detected rectangle and selected 3 larger height
#from the rectangle. because, if the person is near to the camera he will be bigger then the other detected person


import cv2
import sys
import numpy as np 

#function to track object and put ID
def track_object(rects,maxDisappeared = 50, maxDistance=20):
    objects= {}

    def make_id(centroid):
        nonlocal nextObjectID
        objectID = nextObjectID
        objects[objectID] = (centroid, 0)
        nextObjectID +=1
        return objectID
    
    def delete_id(objectID):
        del objects[objectID]


    nextObjectID = 0

#loop existing object ID and upsate their properties
    for objectID in list(objects.keys()):
        objects[objectID] = (objects[objectID][0], objects[objectID][1]+1)
        if objects[objectID][1] > maxDisappeared:
            delete_id(objectID)

    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    #calculate and store centroids of detected object

    for i, (startX, startY, endX,endY) in enumerate(rects):
        centroidX = int((startX +endX)/ 2.0)
        centroidY = int((startY+endY)/ 2.0)
        inputCentroids[i] = (centroidX,centroidY)

    if not objects:
        #if there is no object, creat new object ID for all input centroids
        for i in range(0, len(inputCentroids)):
            make_id(inputCentroids[i])
    else:
        objectIDs = list(objects.keys())
        objectCentroids = [objects[obj][0] for obj in objectIDs] 

        def calculate_distance(a,b):
            return np.linalg.norm(np.array(a) - np.array(b))
        #calculate distance between existing object and input centroids
        
        D = np.zeros((len(objectCentroids), len(inputCentroids)))
        for i, objectCentroid in enumerate(objectCentroids):
            for j, inputCentroid in enumerate(inputCentroids):
                D[i,j] = calculate_distance(objectCentroid, inputCentroid)

        #find the match between object and input centroids
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[row]
        usedRows = set()
        usedCols = set()
        #update object properties based on matching
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if D[row, col] > maxDistance:
                continue

            objectID = objectIDs[row]
            objects[objectID] = (inputCentroids[col], 0)
            usedRows.add(row)
            usedCols.add(col)
        #hand unusede rows for disappeard objects
        unusedRows = set(range(0, D.shape[0])).difference(usedRows) 
        for row in unusedRows:
            objectID = objectIDs[row]
            objects[objectID] = (objects[objectID][0], objects[objectID][1]+1)

            if objects[objectID][1]>maxDisappeared:
               delete_id(objectID)
        #handle unused columns for new object
        for col in set(range(0,D.shape[1])).difference(usedCols):
            make_id(inputCentroids[col])

    return objects    

#function to find the closest person to the camera 
def find_closest_person(detected_persons):
    detected_persons.sort(key = lambda person: person[3] - person[1], reverse = True)
    closest_person = detected_persons[0]
    return closest_person





#function to process video frame
def process_frames(video_path, config_file, model_file):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error")
        return

  
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    

   
    model = cv2.dnn.readNet(model=model_file, config=config_file, framework='TensorFlow')
    
    bbox_colors = (0, 255, 0)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id= frame.copy()
        frame_box = frame.copy()
        frame_close_person = frame.copy()
        
        (h, w) = frame.shape[:2]
#creat blob from the current frame 
        blob = cv2.dnn.blobFromImage(image=frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)

        model.setInput(blob)
        detections = model.forward()

        detected_persons = []
        person_dict = {}

        for i in np.arange(0, detections.shape[2]):
            
            confidence = detections[0,0,i,2]

            

            if confidence > 0.4:
                idx = int(detections[0,0,i,0])
            
                
                if class_names[idx] == "person":
                    bounding_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    detected_persons.append((startX,startY,endX,endY))
                    label = "{}:".format(class_names[idx], confidence * 100) 
                    y = startY - 15 if startY - 15 > 15 else startY + 15 
        #track object and assign ids            
        objects = track_object(detected_persons)
        if detected_persons:  # Check if there are detected persons
            closest_person = find_closest_person(detected_persons)
            (startX, startY, endX, endY) = closest_person
            cv2.rectangle(frame_close_person, (startX, startY), (endX, endY), (255, 0, 0), 6)
        #draw rectangles, labels and circle for the tracked objects
        for objectID, (centroid, _) in objects.items():
            (startX,startY, endX, endY) = detected_persons[objectID]
            text = f"ID{objectID}"
            cv2.rectangle(frame_box, (startX, startY), (endX, endY), bbox_colors, 2)
            cv2.putText(frame_box, "person", (startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,1,bbox_colors,2)
 

            cv2.rectangle(frame_id, (startX, startY), (endX, endY), bbox_colors, 2) 
            cv2.putText(frame_id, text, (startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,1,bbox_colors,2)
            cv2.circle(frame_id,(centroid[0], centroid[1]), 4 , bbox_colors, -1)

        #display the frames 
        stack1 = np.hstack((frame,frame_box))
        stack2 = np.hstack((frame_id,frame_close_person))
        stack3 = np.vstack((stack1,stack2))
        cv2.imshow('Task two', stack3)

        key = cv2.waitKey(1)
        if key != -1:
            break

    cap.release()
    cv2.destroyAllWindows()


#function for define object in task one 
def object_for_task1(contour):
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.6:
        return "person"
    elif aspect_ratio >= 0.6:
        return "car"
    else:
        return "other"
#function for task one    
def taskone(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_num = 1

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(480,360))
        fg_mask = bg_subtractor.apply(frame)
        fg_mask_without_filter = fg_mask.copy()


        #applying morphological operation to remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,(5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,(15,15))
        #finding contour from the fg_mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:
            if cv2.contourArea(contour)> 100:
                object_class = object_for_task1(contour)
                x,y,w,h = cv2.boundingRect(contour)
                objects.append((x,y,w,h,object_class))
        #getting background img
        original_frame = frame
        background_frame = bg_subtractor.getBackgroundImage() 
        #converting 2d binary mask to 3d
        fg_mask_frame2 = cv2.cvtColor(fg_mask_without_filter, cv2.COLOR_BAYER_BG2GRAY)
        height,width = fg_mask_frame2.shape
        img2 = np.zeros((height,width,3),dtype=np.uint8)
        img2[:,:,0] = fg_mask_frame2
        img2[:,:,1] = fg_mask_frame2
        img2[:,:,2] = fg_mask_frame2


        object_frame = np.zeros_like(frame)
        #returing original color of the detected object contour
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            mask = np.zeros_like(frame)
            cv2.drawContours(mask, [contour],0,(255,255,255), thickness=cv2.FILLED)
            object_region = cv2.bitwise_and(frame, mask)
            object_frame = cv2.add(object_frame,object_region)
        #showing image by stacking
        stack1 = np.hstack((original_frame,background_frame))
        stack2 = np.hstack((img2,object_frame))
        stack3 = np.vstack((stack1,stack2))
        cv2.imshow("Task one", stack3)
       #showing counted object in terminal
        obj_count = {"person": 0, "car":0, "other": 0}
        for(x,y,w,h,obj_class) in objects:
            obj_count[obj_class] +=1

        obj_count_str= [f"{count} {obj}" for obj, count in obj_count.items()]
        print(f"Frame {frame_num:04d}: {len(objects)} objects({','.join(obj_count_str)})")
        frame_num += 1

        key = cv2.waitKey(1)
        if key != -1:
            break   
    cap.release()
    cv2.destroyAllWindows()

 









#function for task two
def tasktwo(video_path):
    process_frames(video_path, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 'frozen_inference_graph.pb')
       


if len(sys.argv) == 3 and sys.argv[1] == "-d":
    tasktwo(sys.argv[2])
elif len(sys.argv)== 3 and sys.argv[1] == "-b":
     taskone(sys.argv[2])
  
