import os, cv2
import requests
from resnet50_face_sfew_dag import resnet50_face_sfew_dag
from PIL import Image
import matplotlib.pyplot as plt
import mtcnn
from torchvision import transforms
#https://www.robots.ox.ac.uk/~albanie/pytorch-models.html


def downloadModel():
    if not os.path.isfile('resnet50_face_sfew_dag.pth'):
        weights_path = 'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_face_sfew_dag.pth'
        r = requests.get(weights_path, allow_redirects=True)

        open('resnet50_face_sfew_dag.pth', 'wb').write(r.content)

def loadModel():
    model = resnet50_face_sfew_dag('resnet50_face_sfew_dag.pth')
    return model

def recognizeEmotion(model):
    pixels = cv2.cvtColor(get_new_img_webcam(), cv2.COLOR_BGR2RGB)
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(pixels)
    print(faces)
    x, y, width, height = faces[0]['box']
    face = pixels[y:y + height, x:x + width]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(pixels)
    axs[1].imshow(face)
    plt.show()
    face = Image.fromarray(face, 'RGB')
    face = transforms.Resize((224, 224))(face)
    face = transforms.ToTensor()(face).unsqueeze(1)

    # https://stackoverflow.com/questions/56789038/runtimeerror-given-groups-1-weight-of-size-64-3-3-3-expected-input4-50
    face = face.permute(1, 0, 2, 3)
    output = model(face)
    pred = output.argmax(dim=1, keepdim=True)
    class_ = getClassById(pred)
    return class_

def start():
    downloadModel()
    model = loadModel()
    class_ = recognizeEmotion(model)
    print(class_)


def getClassById(classId):
    if classId == 0:
        return "Angry"
    if classId == 1:
        return "Disgust"
    if classId == 2:
        return "Fear"
    if classId == 3:
        return "Happy"
    if classId == 4:
        return "Neutral"
    if classId == 5:
        return "Sad"
    if classId == 6:
        return "Surprise"


def get_new_img_webcam():
    vid = cv2.VideoCapture(0)
    ret = []
    while (True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            ret = frame
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    return ret


if __name__ == '__main__':
    start()
