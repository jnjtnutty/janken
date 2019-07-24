import cv2
import numpy as np
from keras.models import load_model

model = load_model('fucking_model2.h5')
cap = cv2.VideoCapture(0)
x1 = 10
y1 = 100
w1 = 200
h1 = 200

x2 = 400
y2 = 100
w2 = 200
h2 = 200

dic = {'[0]': 'hammer', '[1]': 'paper', '[2]': 'scissor'}

while True:
    try:
        _, frame = cap.read()
        # bound box fo detect
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinmask = cv2.inRange(hsv, lower, upper)
        skinmask = cv2.GaussianBlur(skinmask, (3, 3), 0)

        # mix threshold+skinmask
        skin = cv2.bitwise_and(thresh, thresh, mask=skinmask)
        bound2 = skin[y2:y2 + h2, x2:x2 + w2]
        bound1 = skin[y1:y1 + h1, x1:x1 + w1]

        contours2, _ = cv2.findContours(bound2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours2) != 0:
            max_area2 = 0
            for i in range(len(contours2)):
                cnt2 = contours2[i]
                area2 = cv2.contourArea(cnt2)
                # print(area)
                if area2 > max_area2:
                    max_area2 = area2
                    ci2 = i
            cnt2 = contours2[ci2]
            hull_2 = cv2.convexHull(cnt2)
        else:
            pass

        contours1, _ = cv2.findContours(bound1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours1)) != 0:
            max_area1 = 0
            for i in range(len(contours1)):
                cnt1 = contours1[i]
                area1 = cv2.contourArea(cnt1)
                # print(area)
                if (area1 > max_area1):
                    max_area1 = area1
                    ci1 = i
            cnt1 = contours1[ci1]
            # print(max_area1)
            hull_1 = cv2.convexHull(cnt1)
        else:
            pass

        drawing = np.zeros(frame.shape, np.uint8)

        drawing2 = drawing[y2:y2 + h2, x2:x2 + w2]
        cv2.drawContours(drawing2, [cnt2], 0, (255, 0, 0), 7)
        cv2.drawContours(drawing2, [hull_2], 0, (0, 0, 255), 7)

        drawing1 = drawing[y1:y1 + h1, x1:x1 + w1]
        cv2.drawContours(drawing1, [cnt1], 0, (255, 0, 0), 7)
        cv2.drawContours(drawing1, [hull_1], 0, (0, 0, 255), 7)

        player1 = drawing1.copy()
        player1 = cv2.resize(player1, (80, 80))
        pic_player1 = cv2.copyMakeBorder(player1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        player2 = drawing2.copy()
        player2 = cv2.resize(player2, (80, 80))
        pic_player2 = cv2.copyMakeBorder(player2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.imshow('player2', pic_player2)
        cv2.imshow('player1', pic_player1)

        pic_player1 = np.reshape(pic_player1, (1, 100, 100, 3))
        # print(pic_player1.shape)
        pic_player2 = np.reshape(pic_player2, (1, 100, 100, 3))

        ans_player1 = model.predict_classes(pic_player1)
        ans_player2 = model.predict_classes(pic_player2)
        ans1 = dic[str(ans_player1)]
        ans2 = dic[str(ans_player2)]

        if ans1 == 'scissor':
            cv2.putText(frame, ans1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                        lineType=cv2.LINE_AA)
        else:
            if max_area1 > 12000:
                cv2.putText(frame, ans1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                            lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame, 'rock', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                            lineType=cv2.LINE_AA)

        # print('ans'+str(ans_player1))
        cv2.putText(frame, ans2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                    lineType=cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.imshow('skin',skin)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite('drawcontour1.jpg',pic_player1)
            break

    except:
        pass
cap.release()
cv2.destroyAllWindows()
