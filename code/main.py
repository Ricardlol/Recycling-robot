import classModels
import orientationDetector
import cv2
import uuid
import socket

tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('172.31.0.1', 11001)
#server_address = ('localhost', 11002)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)
print("Success Connect")
data = sock.recv(16)

cap = cv2.VideoCapture(0)

leido, frame = cap.read()
cv2.imwrite("./background.png", frame)

names_class = ['blue', 'green', 'organic', 'yellow']
model = classModels.modelsDenseNet121("modelUse")


takeFoto = False
stopProgram = False

if data == b'I':
    while not(stopProgram):
        while not(takeFoto):
            data = sock.recv(16)
            print(data)
            if data == b'R':
                takeFoto = True
        leido, frame = cap.read()

        if leido:
            nameFoto = './img/' +str(uuid.uuid4()) + ".png" # uuid4 regresa un objeto, no una cadena. Por eso lo convertimos
            print("Foto tomada correctamente con el nombre {}".format(nameFoto))
            cv2.imwrite(nameFoto, frame)

            #print("Obtain orientation")
            #angle = orientationDetector.getOrientation(nameFoto)

            print("Classify image")
            resultClass = model.classifyImage(names_class, nameFoto)
            print(resultClass)
            print('sending {!r}'.format(resultClass))
            sock.sendall(resultClass)

        else:
            print("Error al acceder a la cámara")
            stopProgram = True
        takeFoto = False

cap.release()
sock.close()




'''
secondModel.myImages(names_class, "./dataset/myImages/yellow3.JPG")
secondModel.myImages(names_class, "./dataset/myImages/yellow4.JPG")
'''
