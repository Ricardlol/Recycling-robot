import classModels
import cv2
import uuid
import socket

cap = cv2.VideoCapture(1)

names_class = ['blue', 'green', 'organic', 'yellow']
model = classModels.modelsDenseNet121("modelUse")

tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 11002)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

takeFoto = False
stopProgram = False

while not(stopProgram):

    if takeFoto:
        leido, frame = cap.read()

        if leido:
            nameFoto = "../img/"+str(uuid.uuid4()) + ".png" # uuid4 regresa un objeto, no una cadena. Por eso lo convertimos
            cv2.imwrite(nameFoto, frame)
            resultClass = model.classifyImage(names_class, nameFoto)

            print("Foto tomada correctamente con el nombre {}".format(nameFoto))

            message = bytes(resultClass)
            print('sending {!r}'.format(message))
            sock.sendall(message)

        else:
            print("Error al acceder a la cámara")
            stopProgram = True
cap.release()
sock.close()




'''
secondModel.myImages(names_class, "./dataset/myImages/yellow3.JPG")
secondModel.myImages(names_class, "./dataset/myImages/yellow4.JPG")
'''
