import cv2 as cv
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('C:\\Users\\shobh\\PycharmProjects\\ROCK_PAPER_SCISSOR\\my_model1.h5')
model.load_weights('C:\\Users\\shobh\\PycharmProjects\\ROCK_PAPER_SCISSOR\\my_model_weights1.h5')

list_images = [cv.imread(
    'C:\\Users\\shobh\\PycharmProjects\\ROCK_PAPER_SCISSOR\\dataset-rps\\rps-test-set\\rock\\testrock01-12.png'),
    cv.imread(
        'C:\\Users\\shobh\\PycharmProjects\\ROCK_PAPER_SCISSOR\\dataset-rps\\rps-test-set\\paper\\testpaper01-12.png'),
    cv.imread(
        'C:\\Users\\shobh\\PycharmProjects\\ROCK_PAPER_SCISSOR\\dataset-rps\\rps-test-set\\scissors\\testscissors01-12.png')]


def rescle_img(img):
    x = np.zeros((600, 600, 3))
    for i in range(300):
        for j in range(300):
            x[i * 2, j * 2, :] = x[i * 2 + 1, j * 2, :] = x[i * 2, j * 2 + 1, :] = x[i * 2 + 1][j * 2 + 1] = img[i,j,:]

    return np.array(x, np.uint8)


rescle_img(list_images[0])
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, test_img = cap.read()
    if not ret:
        continue

    test_img = test_img[90:390, 170:470]
    img = rescle_img(test_img)

    cv.imshow('player', img)

    test_img = np.expand_dims(test_img, axis=0)

    pred = model.predict(test_img)
    max_index = np.argmax(pred[0])
    gesture = ('paper', 'rock', 'scissors')

    if max_index == 0:
        cv.imshow('AI', rescle_img(list_images[2]))
    elif max_index == 1:
        cv.imshow('AI', rescle_img(list_images[1]))
    else:
        cv.imshow('AI', rescle_img(list_images[0]))

    predicted_gesture = gesture[max_index]
    print(predicted_gesture)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


def rescle_img(img):
    x = []
    for i in range(300):
        for j in range(300):
            x[i * 2][j * 2] = img[i][j]
            x[i * 2 + 1][j * 2] = img[i][j]
            x[i * 2][j * 2 + 1] = img[i][j]
            x[i * 2 + 1][j * 2 + 1] = img[i][j]

    cv.imshow('assa', np.array(x))
