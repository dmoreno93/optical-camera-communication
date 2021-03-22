import multiprocessing as mp
import threading
import MultispectralCamera
import Decoder
import time
import random
import string
import cv2
import numpy as np
from numpy_ringbuffer import RingBuffer
import tifffile as tiff
import serial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def clean_pipe(pipe):
    while pipe.poll():
        pipe.recv()


# First of all we create the pipe
parent_pipe, child_pipe = mp.Pipe()

# And then we start the process
process = mp.Process(target=MultispectralCamera.camera_loop, args=(parent_pipe,))
process.start()

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)  # open serial port
time.sleep(2)  # wait for the Arduino bootloader

array = RingBuffer(capacity=300, dtype=np.ndarray)  # create an array containing the band values of the selected pixel

child_pipe.send("START")

# We select the pixel of interest
image = child_pipe.recv()  # receive a frame

child_pipe.send("STOP")
clean_pipe(child_pipe)

plt.imshow(image[:, :, 8], 'coolwarm')  # show the frame to choose the pixel
coords = plt.ginput()  # select coordinates
plt.close()


def save_image(img, path='/home/mithrandir/Documentos/Images/', filename='img', count=0):
    cv2.imwrite(path + filename + '_' + '%d' % count + '.tiff', img)


def hypercube(img, shape_x, shape_y, size_x_macro_pixel, size_y_macro_pixel):
    sorted_array = [3, 2, 1, 5, 6, 7, 4, 8, 0]
    base_x = np.linspace(0, shape_x - 1, shape_x, dtype=int)
    base_y = np.linspace(0, shape_y - 1, shape_y, dtype=int)
    cube = np.zeros((shape_y, shape_x, size_x_macro_pixel * size_y_macro_pixel), dtype=np.uint16)
    for x in range(0, size_x_macro_pixel):
        for y in range(0, size_y_macro_pixel):
            # cube[:, :, sorted_array[x*size_x_macro_pixel + y]] = img[x::size_x_macro_pixel, y::size_y_macro_pixel]
            tM = img[:, base_x * size_x_macro_pixel + x]
            cube[:, :, sorted_array[y * size_y_macro_pixel + x]] = tM[base_y * size_y_macro_pixel + y, :]
    return cube


def read_ms_image(path, filename):
    """ Read an raw image and split it into different MS bands"""
    img = tiff.TiffReader(path + filename + '.tiff')  # read TIFF file
    img_array = img.asarray()  # turn into an array
    cube = hypercube(np.array(img_array), 426, 339, 3, 3)  # split raw image into 9 bands
    return cube


def crop_image(img, pixel_x, pixel_y, offset_x=20, offset_y=20):
    """ Crop the MS image"""
    image_cropped = np.squeeze(
        img[(pixel_x - offset_x):(pixel_x + offset_x), (pixel_y - offset_y):(pixel_y + offset_y), :])  # cropping image
    return image_cropped


def get_spectral_signature(ms_image, pixel_x=None, pixel_y=None):
    spectral_response = np.zeros((9, 1))  # preallocation
    if (pixel_x is None) or (pixel_y is None):  # get the spectral response from the pixel mean of each band
        for i in range(9):
            spectral_response[i] = np.mean(np.mean(ms_image[:, :, i]))
    else:  # get the spectral response from the specified pixels
        for i in range(9):
            spectral_response[i] = ms_image[pixel_x, pixel_y, i]
    return spectral_response


def channel_matrix(sp_1, sp_2):
    h = np.zeros((9, 2))  # preallocation
    for i in range(9):
        h[i, 0] = sp_1[i]
        h[i, 1] = sp_2[i]
    return h


#def write_read(data):
#    while True:
#        arduino.write(bytes(data, 'utf-8'))
#        print("character written")
#        # time.sleep(0.05)
#        ack = arduino.readline()
#        while ack == b'':
#            ack = arduino.readline()
#            print("waiting for ack")

def write_read():
    file = open("/home/mithrandir/Documentos/sample.txt", "a")
    while True:
        c = random.choice(string.ascii_letters + string.digits)
        file.write(c + "\n")
        arduino.write(bytes(c, 'utf-8'))
        print("character written")
        # time.sleep(0.05)
        ack = arduino.readline()
        while ack == b'':
            ack = arduino.readline()
            print("waiting for ack")


def generate_chars_to_send(amount):
    # return [0, 0]
    return [random.randint(0, 255) for i in range(amount)]
    #characters = []
    #for i in range(10):
        #characters.append(random.choice(string.ascii_letters + string.digits))
    #    i += 1
    #return characters

#def generate_char():
#    c = random.choice(string.ascii_letters + string.digits)
#    file = open("/home/mithrandir/Documentos/sample.txt", "a")
#    file.write(c + "\n")
#    file.close()
#    return c


def store_pixel():
    while True:
        frame = child_pipe.recv()
        array.append(frame[int(coords[0][1]), int(coords[0][0]), :])
        # print(len(array))


def arduino_send(character):
    time.sleep(0.5)
    arduino.write(int.to_bytes(character, length=1, byteorder='big', signed=False))

# t1 = threading.Thread(target=store_pixel)
# t2 = threading.Thread(target=write_read, args=generate_char())
# t2 = threading.Thread(target=write_read)
# t1.start()
# t2.start()
# t1.join()
# t2.join()


try:
    buffer_length = 300
    chars_to_send = generate_chars_to_send(500)
    np.save("captures/list.npy", chars_to_send)

    for index, current_char in enumerate(chars_to_send):
        child_pipe.send("START")
        print("START SENT")
        t = threading.Thread(target=arduino_send, args=(current_char,))
        t.start()
        print("ARDUINO STARTED")
        frame_counter = 0
        while frame_counter < buffer_length:
            frame = child_pipe.recv()
            array.append(frame[int(coords[0][1]), int(coords[0][0]), :])
            frame_counter += 1
        t.join()
        print("Bit stream captured")
        child_pipe.send("STOP")
        clean_pipe(child_pipe)
        np.save("captures/" + str(index) + ".npy", array)
    print("Capture completed")

    child_pipe.send('FINISH')

    for band in range(9):
        Decoder.execute(band)

except KeyboardInterrupt:
    child_pipe.send('ABORT')



###### LOAD DATA ######
# save np.load
#np_load_old = np.load
## modify the default parameters of np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
## call load_data with allow_pickle implicitly set to true
#data = np.load('/home/mithrandir/Documentos/1.npy')
#char_tx = np.load('/home/mithrandir/Documentos/list.npy')
## restore np.load for future normal usage
#np.load = np_load_old
#
#print(data.shape)
#print(data[0][:])
#print(char_tx)
#plt.plot(range(9), data[45][:])
#plt.show()
#######################

# np.savetxt('/home/mithrandir/Documentos/Images/frames.npy', array) # save arrays


############ DISPLAY IMAGE ############
# while True:
    # frame = child_pipe.recv()
    # print(np.max(frame.flatten()))
    # cv2.imshow('Display', frame[100:, 100:]) # to show the raw frame
    # cube_display = np.array(frame, dtype=np.uint8) # to show the MS frame
    # cv2.imshow('Display', cube_display[:, :, 8])
    # cv2.waitKey(1)
#######################################

############## CHECK FPS ##############
# print("Saving images")
# for x in range(250):
#    save_image(frame, '/home/mithrandir/Documentos/Images/', 'test', x)
#    time.sleep(0.02)
# print("All images saved")
#######################################

# Here's an example of usage
#image = read_ms_image('/home/mithrandir/Documentos/Images/', 'img_0')  # read image
#plt.imshow(image[:, :, 3], 'coolwarm')  # show image
#coords = plt.ginput()  # select coordinates
#print(coords[0][0])
#print(coords[0][1])
#plt.close()
#sp = get_spectral_signature(image, int(coords[0][0]), int(coords[0][0]))
#plt.plot(range(9), sp[:])
#plt.show()

