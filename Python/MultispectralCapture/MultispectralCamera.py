from pyueye import ueye
import numpy as np
import threading
import time


# Loop to send images very niceeeely!
def camera_loop(pipe):
    # We create the Camera object
    ms = MultispectralCamera()
    # We set the camera parameters
    ms.set_shutter_mode('gs')
    ms.set_pixel_clock(71)
    ms.set_fps(50)
    ms.set_exposure_time(6)
    ms.set_hw_gain(100)
    print("REEEADY")

    # We go live
    ms.go_live()
    ms.piped_acquisition(pipe)


def hypercube(img, shape_x, shape_y, size_x_macro_pixel, size_y_macro_pixel):
    """
    Split the image into spectral bands.
    :param img: raw multispectral image
    :param shape_x: width of the multispectral image
    :param shape_y: height of the multispectral image
    :param size_x_macro_pixel: width of the macro-pixel
    :param size_y_macro_pixel: height of the macro-pixel
    :return: multispectral image divided into 9 bands
    """
    img = np.squeeze(img)  # remove the axis of length one
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


class MultispectralCamera(object):

    # # # #  # # # #
    #  CONSTRUCTOR #
    # # # #  # # # #

    def __init__(self):
        # self.cam = ueye.HIDS(0 | ueye.IS_ALLOW_STARTER_FW_UPLOAD) # Upload new starter firmware during initialization
        self.cam = ueye.HIDS(0)
        self.cam_info = ueye.CAMINFO()
        self.sensor_info = ueye.SENSORINFO()
        self.image_memory = ueye.c_mem_p()
        self.memory_id = ueye.int()
        self.rect_aoi = ueye.IS_RECT()
        self.bits_per_pixel = ueye.INT(24)
        self.bytes_per_pixel = int(self.bits_per_pixel/8)
        self.pitch = ueye.INT()
        self.color_mode = ueye.INT()
        self.width = 0
        self.height = 0
        self.status = 'IDLE'
        self.data = []
        # self.lock = Lock()
        self.pipe = 0
        self.offset_x = 2
        self.offset_y = 2

        self.__init_camera()
        self.__get_camera_info()
        self.__get_sensor_info()
        self.__default()
        self.__display_mode()
        self.__get_color_mode()
        self.__get_dimensions()
        self.__memory_allocation()
        self.__set_color_mode()
        # self.__set_events()
        self.__mandatory_shit()

        print("Color mode {}".format(self.color_mode))
        print("Bits per pixel {}".format(self.bits_per_pixel))
        print("Bytes per pixel {}".format(self.bytes_per_pixel))

    # # # # #  # # # # #
    #  PRIVATE METHODS #
    # # # # #  # # # # #

    # Mandatory shit ######################
    def __mandatory_shit(self):
        self.__set_gamma(100)
        self.__disable_hot_pixel_correction(0)
        self.__disable_hdr()
        self.__disable_gain_boost()
        self.__device_feature(1)
        self.__set_black_level(0)
        # self.__disable_white_balance()  # doesn't work
        self.__disable_binning()
        # self.__disable_color_correction() # doesn't work

    def __set_gamma(self, value=100):
        gamma = ueye.INT(int(value))
        if not ueye.is_Gamma(self.cam, ueye.IS_GAMMA_CMD_SET, gamma, ueye.sizeof(gamma)) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_GAMMA_CMD_SET failed")

    def __disable_hot_pixel_correction(self, value=0):
        hot_pixel = ueye.c_void_p(int(value))
        if not ueye.is_HotPixel(self.cam, ueye.IS_HOTPIXEL_DISABLE_CORRECTION, hot_pixel,
                                ueye.sizeof(hot_pixel)) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_HOTPIXEL_DISABLE_CORRECTION failed")
        if not ueye.is_HotPixel(self.cam, ueye.IS_HOTPIXEL_DISABLE_SENSOR_CORRECTION, hot_pixel,
                                ueye.sizeof(hot_pixel)) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_HOTPIXEL_DISABLE_SENSOR_CORRECTION failed")

    def __disable_hdr(self):
        if not ueye.is_EnableHdr(self.cam, ueye.IS_DISABLE_HDR) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_DISABLE_HDR failed")

    def __disable_gain_boost(self):
        if not ueye.is_SetGainBoost(self.cam, ueye.IS_SET_GAINBOOST_OFF) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_SET_GAINBOOST_OFF failed")

    def __device_feature(self, value=1):
        param = ueye.INT(int(value))
        if not ueye.is_DeviceFeature(self.cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LOG_MODE, param,
                                     ueye.sizeof(param)) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_DEVICE_FEATURE_CMD_SET_LOG_MODE failed")

    def __set_black_level(self, level):
        bl = ueye.UINT(int(level))
        if not ueye.is_Blacklevel(self.cam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, bl,
                                  ueye.ctypes.sizeof(bl)) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_BLACKLEVEL_CMD_SET_OFFSET failed")
        return bl.value

    def __disable_binning(self):
        if not ueye.is_SetBinning(self.cam, ueye.IS_BINNING_DISABLE) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_BINNING_DISABLE failed")

    def __disable_white_balance(self):
        if not ueye.is_SetWhiteBalance(self.cam, ueye.IS_SET_WB_DISABLE) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_SET_WB_DISABLE failed")

    def __disable_color_correction(self):
        if not ueye.is_SetColorCorrection(self.cam, ueye.IS_CCOR_DISABLE) == ueye.IS_SUCCESS:
            raise RuntimeError("IS_CCOR_DISABLE failed")
    #################################################

    # Camera Initialization
    def __init_camera(self):
        if not ueye.is_InitCamera(self.cam, None) == ueye.IS_SUCCESS:
            raise RuntimeError("Camera not initialized")
#            # Check if GigE uEye needs a new starter firmware
#            if ueye.is_InitCamera(self.cam, None) == ueye.IS_STARTER_FW_UPLOAD_NEEDED:
#                # Upload new starter firmware during initialization
#                self.cam = self.cam | ueye.IS_ALLOW_STARTER_FW_UPLOAD
#                if not ueye.is_InitCamera(self.cam, None) == ueye.IS_SUCCESS:
#                    raise RuntimeError("Camera not initialized")

    # Sensor Info structure
    def __get_sensor_info(self):
        if not ueye.is_GetSensorInfo(self.cam, self.sensor_info) == ueye.IS_SUCCESS:
            raise RuntimeError("Sensor Info not fetched")

        print("Sensor info acquired")

    # Camera info structure
    def __get_camera_info(self):
        if not ueye.is_GetCameraInfo(self.cam, self.cam_info) == ueye.IS_SUCCESS:
            raise RuntimeError("Camera Info not fetched")

        print("Camera info acquired")

    # Reset all parameters of the camera to the standard values
    def __default(self):
        if not ueye.is_ResetToDefault(self.cam) == ueye.IS_SUCCESS:
            raise RuntimeError("Could not reset to default")

    # Display Mode
    def __display_mode(self):
        if not ueye.is_SetDisplayMode(self.cam, ueye.IS_SET_DM_DIB) == ueye.IS_SUCCESS:
            raise RuntimeError("Display mode error")

    # Sensor dimensions
    def __get_dimensions(self):
        if not ueye.is_AOI(self.cam,
                           ueye.IS_AOI_IMAGE_GET_AOI,
                           self.rect_aoi,
                           ueye.sizeof(self.rect_aoi)) == ueye.IS_SUCCESS:
            raise RuntimeError("Dimensions not fetched")

        self.width = self.rect_aoi.s32Width
        self.height = self.rect_aoi.s32Height

        print("Sensor dimensions acquired")

    # Image memory allocation and set up
    def __memory_allocation(self):
        if not ueye.is_AllocImageMem(self.cam,
                                     self.width,
                                     self.height,
                                     self.bits_per_pixel,
                                     self.image_memory,
                                     self.memory_id) == ueye.IS_SUCCESS:
            raise RuntimeError("Memory not allocated")

        if not ueye.is_SetImageMem(self.cam, self.image_memory, self.memory_id) == ueye.IS_SUCCESS:
            raise RuntimeError("Memory not set")

        print("Memory allocated")

    # Color mode
    def __get_color_mode(self):
        # Set the right color mode
        if int.from_bytes(self.sensor_info.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            ueye.is_GetColorDepth(self.cam, self.bits_per_pixel, self.color_mode)
            self.bytes_per_pixel = int(self.bits_per_pixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tcolor_mode: \t\t", self.color_mode)
            print("\tnBitsPerPixel: \t\t", self.bits_per_pixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sensor_info.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.color_mode = ueye.IS_CM_BGRA8_PACKED
            self.bits_per_pixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.bits_per_pixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tcolor_mode: \t\t", self.color_mode)
            print("\tnBitsPerPixel: \t\t", self.bits_per_pixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sensor_info.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.color_mode = ueye.IS_CM_MONO10
            self.bits_per_pixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.bits_per_pixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tcolor_mode: \t\t", self.color_mode)
            print("\tnBitsPerPixel: \t\t", self.bits_per_pixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            self.color_mode = ueye.IS_CM_MONO8
            self.bits_per_pixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.bits_per_pixel / 8)

    def __set_color_mode(self):
        ueye.is_SetColorMode(self.cam, self.color_mode)

    # Set Events
    def __set_events(self):
        if not ueye.is_EnableEvent(self.cam, ueye.IS_SET_EVENT_FRAME) == ueye.IS_SUCCESS:
            raise RuntimeError("Event not set")

        print("Events enabled")

    # Disable events
    def __disable_events(self):
        if not ueye.is_DisableEvent(self.cam, ueye.IS_SET_EVENT_FRAME) == ueye.IS_SUCCESS:
            raise RuntimeError("Event not disabled")

        print("Events disabled")

    # Disables the "cam" camera handle and releases the data structures and memory areas taken up by the uEye camera
    def __exit(self):
        if not ueye.is_ExitCamera(self.cam) == ueye.IS_SUCCESS:
            raise RuntimeError("Camera is still attached")

        print("Camera detached")

    # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
    def __free_memory(self):
        if not ueye.is_FreeImageMem(self.cam, self.image_memory, self.memory_id) == ueye.IS_SUCCESS:
            raise RuntimeError("Image memory was not released")

        print("Image memory released")

    # Pick up spectral signature of a given pixel
    def __pick_up_signature(self, data, x, y):
        x_index = int(x/3)*3
        y_index = int(y/3)*3

        base_x_pos = self.offset_x + x_index
        base_y_pos = self.offset_y + y_index

        return data[base_x_pos:base_x_pos+3, base_y_pos:base_y_pos+3].flatten()

    # # # # # # # # # #
    #  PUBLIC METHODS #
    # # # # # # # # # #

    # ---------- PIXEL CLOCK --------- #
    def set_pixel_clock(self, pixel_clock):
        """
        Set the pixel clock in MHz.
        :param pixel_clock: pixel clock to be set
        :return: actual pixel clock
        """
        pc = ueye.UINT(int(pixel_clock))
        if not ueye.is_PixelClock(self.cam, ueye.IS_PIXELCLOCK_CMD_SET, pc, ueye.ctypes.sizeof(pc)) == ueye.IS_SUCCESS:
            raise RuntimeError("Pixel Clock not set")

        print("Pixel clock set to", pc.value, "MHz")

        return pc.value

    # ---------- FPS --------- #
    def set_fps(self, fps):
        """
        Set the frame rate in frames per second.
        :param fps: frames per second to be set
        :return: actual frame rate
        """
        new_fps = ueye.ctypes.c_double()
        if not ueye.is_SetFrameRate(self.cam, float(fps), new_fps) == ueye.IS_SUCCESS:
            raise RuntimeError("Frame Rate not set")

        if new_fps.value != fps:
            print("Warning actual fps is", new_fps.value)
        else:
            print("Frame rate set to %8.3f" % fps)

        return new_fps.value

    # ---------- SHUTTER MODE --------- #
    def set_shutter_mode(self, shutter_mode):
        """
        Set shutter mode to global shutter (gs), rolling shutter (rs) or rolling shutter with global start (rsgs).
        :param shutter_mode: shutter mode to be set
        :return: None
        """
        if shutter_mode == 'gs':
            s_mode = ueye.ctypes.c_uint32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL)
            # s_mode = ueye.UINT((ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL))
            print('Global shutter is set')
        elif shutter_mode == 'rs':
            s_mode = ueye.ctypes.c_uint32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING)
            # s_mode = ueye.UINT((ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING))
            print('Rolling shutter is set')
        elif shutter_mode == 'rsgs':
            s_mode = ueye.ctypes.c_uint32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START)
            # s_mode = ueye.UINT((ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START))
            print('Rolling shutter with global start is set')
        else:
            raise RuntimeError("set_shutter_mode ERROR: 'shuttermode' must be 'gs', 'rs' or 'rsgs'")
        if not ueye.is_DeviceFeature(self.cam, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE, s_mode,
                                     ueye.ctypes.sizeof(s_mode)) == ueye.IS_SUCCESS:
            raise RuntimeError("Shutter mode not set")

    # ---------- EXPOSURE TIME --------- #
    def set_exposure_time(self, exposure_ms):
        """
        Set exposure time in milliseconds. If 0 is passed, the exposure time is set to the maximum value of 1/frameRate.
        :param exposure_ms: exposure time to be set in ms
        :return: None
        """

        exposure_ms_double = ueye.ctypes.c_double(exposure_ms)
        if not ueye.is_Exposure(self.cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_ms_double,
                                ueye.ctypes.sizeof(exposure_ms_double)) == ueye.IS_SUCCESS:
            raise RuntimeError("Exposure time not set")
        actual = exposure_ms_double.value
        print("Exposure time is set to", actual, "ms")
        if actual != exposure_ms:
            print("Warning: actual value of exposure time is", actual, "ms")

    # ---------- GAIN --------- #
    def set_hw_gain(self, gain):
        """
        Set the master amplification. It can be tuned between 0% and 100%.
        :param gain: amplification to be set
        :return: None
        """
        gn = ueye.UINT(int(gain))
        # nRet = ueye.is_SetHWGainFactor(self.cam, ueye.IS_SET_MASTER_GAIN_FACTOR, gn)
        # if not ueye.is_SetHardwareGain(self.cam, gn, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER,
        #                                ueye.IS_IGNORE_PARAMETER) == ueye.IS_SUCCESS:
        if not ueye.is_SetHardwareGain(self.cam, gn, ueye.UINT(int(0)), ueye.INT(int(0)),
                                       ueye.INT(int(0))) == ueye.IS_SUCCESS:
            raise RuntimeError("Gain not set")
        print("Master gain set to", gn.value)

    # ----------- GO LIVE ---------- #
    def go_live(self):
        if not ueye.is_CaptureVideo(self.cam, ueye.IS_DONT_WAIT) == ueye.IS_SUCCESS:
            raise RuntimeError("Capture mode failed")

        if not ueye.is_InquireImageMem(self.cam,
                                       self.image_memory,
                                       self.memory_id,
                                       self.width,
                                       self.height,
                                       self.bits_per_pixel,
                                       self.pitch) == ueye.IS_SUCCESS:
            raise RuntimeError("Memory inquiry failed")

        print("Pitch is {}".format(self.pitch))
        print("Camera in live mode")

    # ----------- GO LIVE ---------- #
    def control_pipe(self):
        inner_status = 'OK'
        while inner_status == 'OK':
            command = self.pipe.recv()
            print(command)
            if command == "START":
                self.__set_events()
                self.status = 'RUN'
            elif command == "STOP":
                self.__disable_events()
                self.status = 'IDLE'
            else:
                self.__disable_events()
                self.__free_memory()
                self.__exit()
                self.status = 'GO_TO_HELL'
                inner_status = 'KO'

    # ---------- PIPED ACQUISITION ---------- #
    def piped_acquisition(self, pipe):
        self.status = 'IDLE'
        self.pipe = pipe
        t = threading.Thread(target=self.control_pipe)
        t.start()

        print("Acquisition started!")

        while True:
            if self.status == 'RUN':
                if ueye.is_WaitEvent(self.cam, ueye.IS_SET_EVENT_FRAME, 5000) == ueye.IS_SUCCESS:
                    data = ueye.get_data(self.image_memory, self.width, self.height, self.bits_per_pixel, self.pitch,
                                         False)
                    frame = np.reshape(data, (self.height.value, self.width.value, self.bytes_per_pixel))
                    # self.pipe.send(frame)  # send raw image
                    # raw_frame = frame[:, :, 1] * 256 + frame[:, :, 0]  # Image raw in 10bits
                    # self.pipe.send(raw_frame)
                    cube = hypercube(frame, 426, 339, 3, 3)
                    self.pipe.send(cube)  # send MS image
            elif self.status == 'IDLE':
                time.sleep(0.25)
            else:
                break
