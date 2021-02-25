import multiprocessing
import MultispectralCamera
import time
import cv2
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import serial

# First of all we create the pipe
parent_pipe, child_pipe = multiprocessing.Pipe()

# And then we start the process
process = multiprocessing.Process(target=MultispectralCamera.camera_loop, args=(parent_pipe,))
process.start()

#arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)


def save_image(image, path='/home/mithrandir/Documentos/optical-camera-communication-master/Python/MultispectralCapture/Images/',
               filename='img', count=0):
    cv2.imwrite(path + filename + '_' + '%d' % count + '.tiff', image)


def Hypercube(image, shapeX, shapeY, tailleXMacropixel, tailleYMacropixel):
    sortedArray = [3, 2, 1, 5, 6, 7, 4, 8, 0]
    baseX = np.linspace(0, shapeX-1, shapeX, dtype=int)
    baseY = np.linspace(0, shapeY-1, shapeY, dtype=int)
    cube = np.zeros((shapeY, shapeX, tailleXMacropixel * tailleYMacropixel), dtype=np.uint16)
    for x in range(0, tailleXMacropixel):
        for y in range(0, tailleYMacropixel):
            # cube[:, :, sortedArray[x*tailleXMacropixel + y]] = image[x::tailleXMacropixel, y::tailleYMacropixel]
            tM = image[:, baseX*tailleXMacropixel + x]
            cube[:, :, sortedArray[y*tailleYMacropixel + x]] = tM[baseY*tailleYMacropixel+y, :]
    return cube


def read_ms_image(path, filename):
    """ Read an raw image and split it into different MS bands"""
    img = tiff.TiffReader(path + filename + '.tiff') # read TIFF file
    img_array = img.asarray() # turn into an array
    cube = Hypercube(np.array(img_array), 426, 339, 3, 3) # split raw image into 9 bands
    return cube


def crop_image(img, pixel_x, pixel_y, offset_x=20, offset_y=20):
    """ Crop the MS image"""
    image_cropped = np.squeeze(img[(pixel_x - offset_x):(pixel_x + offset_x), (pixel_y - offset_y):(pixel_y + offset_y), :]) # cropping image
    return image_cropped


def get_spectral_signature(ms_image, pixel_x=None, pixel_y=None):
    spectral_response = np.zeros((9, 1)) # preallocation
    if (pixel_x is None) or (pixel_y is None): # get the spectral response from the pixel mean of each band
        for i in range(9):
            spectral_response[i] = np.mean(np.mean(ms_image[:, :, i]))
    else: # get the spectral response from the specified pixels
        for i in range(9):
            spectral_response[i] = ms_image[pixel_x, pixel_y, i]
    return spectral_response


def channel_matrix(sp_1, sp_2):
    h = np.zeros((9, 2)) # preallocation
    for i in range(9):
        h[i, 0] = sp_1[i]
        h[i, 1] = sp_2[i]
    return h


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data_read = arduino.readline()
    return data_read


while True:
    data = child_pipe.recv()

    ############ DISPLAY IMAGE ############
    # print(np.max(data.flatten()))
    cv2.imshow('Display', data[100:, 100:])
    cv2.waitKey(1)
    #######################################

    ############ SAVE IMAGES ############
    #for k in range():
        #save_image(data, count=k)

#data = child_pipe.recv()
#save_image(data)


#image = read_ms_image('/home/mithrandir/Documentos/optical-camera-communication-master/Python/MultispectralCapture/Images/', 'img_0') # read image
#plt.imshow(image[:,:,3], 'coolwarm') # show image
#plt.show()
#coords = plt.ginput() # select coordinates
#print(coords[0][0])
#print(coords[0][1])
#plt.close()