import cv2

def faceBox(faceNet,frame):
    #on lui affecte la longeur et largeur du frame
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    #creer un blob a partir dune image
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    #on lui affecte le blob pour nous rend detection(detecter le visage)
    detection=faceNet.forward()
    bboxs=[]
    #pour préciser les detection (il prend que les trucs qu'il est sur qu'il sont des visage)
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        #preciser les cotes du cadre
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            #creer le rectancle avec les dimensions précis
            #0,255,5 : color et 1 : somk
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs

#des Models de detection des visages
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

#des Models de detection d'ages
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

#des Models de detection des genres
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


#Load : read(les variables definit en haut)
faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#préciser les ages
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
#preciser les types
genderList = ['Male', 'Female']

#on lui associe la webCam
video=cv2.VideoCapture(0)

padding=20

while True:
    # lire la video
    ret,frame=video.read()
    #detecter le frame dans la video
    frame,bboxs=faceBox(faceNet,frame)
    #boucle sur tous les frames q'il trouve
    for bbox in bboxs:
        # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #appliquer les dimensions de l img
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        #on lui affecte le blob
        genderNet.setInput(blob)
        #commence a prédire le genre du blob
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        #commence a prédire l'age
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        # initialisant le label qu'on va afficher par le genre et l age
        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1)
        # on lui affecte le style du texte et sa position
        # (bbox[0], bbox[1]-10) : la position du text : fo9
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        #Affiche l img
    cv2.imshow("Age-Gender",frame)
    #attend le clique , s'il est q : il quitte
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()