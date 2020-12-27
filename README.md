## Emotion Recognition

### Goal
- Take a photo with a webcam and the program recognizes your emotion.

### Instructions

I used a [pre-trained model](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_face_sfew_dag.py) with its [weights](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_face_sfew_dag.pth).
In main.py you will find the code that downloads the model weights.

````
def downloadModel():
    if not os.path.isfile('resnet50_face_sfew_dag.pth'):
        weights_path = 'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_face_sfew_dag.pth'
        r = requests.get(weights_path, allow_redirects=True)
        open('resnet50_face_sfew_dag.pth', 'wb').write(r.content)
````

After that, the model is instantiated with the weights.

````
def loadModel():
    model = resnet50_face_sfew_dag('resnet50_face_sfew_dag.pth')
    return model
````

Having the model, the method recognizeEmotion will take a picture using your webcam.
Press "w" to take the picture.

Then, the picture will be modified, focusing on the face.

Finally, the method recognizeEmotion will print your emotion.

### References

- [Learning Grimaces by Watching TV](https://www.robots.ox.ac.uk/~albanie/publications/albanie16learning.pdf)
- [Samuel Albanie Pytorch Models](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html)
