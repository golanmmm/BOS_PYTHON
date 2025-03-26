import ctypes
import numpy as np
import cv2
from pyueye import ueye

def main():
    hCam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hCam, None)
    if ret != ueye.IS_SUCCESS:
        print(f"Camera initialization failed with error code: {ret}")
        return

    pcImageMemory = None
    MemID = ueye.int()

    try:
        # Set color mode to 8-bit monochrome
        ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)
        if ret != ueye.IS_SUCCESS:
            print(f"SetColorMode failed with error code: {ret}")
            return

        # Set region of interest (AOI)
        sensor_width = 752  # UI-1220ME-M-GL max width
        sensor_height = 480  # UI-1220ME-M-GL max height
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(sensor_width)
        rect_aoi.s32Height = ueye.int(sensor_height)
        ret = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ctypes.sizeof(rect_aoi))
        if ret != ueye.IS_SUCCESS:
            print(f"AOI setup failed with error code: {ret}")
            return

        # Allocate image memory
        pcImageMemory = ueye.c_mem_p()
        ret = ueye.is_AllocImageMem(hCam, sensor_width, sensor_height, 8, pcImageMemory, MemID)
        if ret != ueye.IS_SUCCESS:
            print(f"AllocImageMem failed with error code: {ret}")
            return

        # Set active image memory
        ret = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if ret != ueye.IS_SUCCESS:
            print(f"SetImageMem failed with error code: {ret}")
            return

        # Enable automatic exposure
        enable = ueye.double(1)  # 1 to enable, 0 to disable
        ret = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable, None)
        if ret != ueye.IS_SUCCESS:
            print(f"Auto exposure enable failed with error code: {ret}")
            return

        # Enable automatic gain
        ret = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, enable, None)
        if ret != ueye.IS_SUCCESS:
            print(f"Auto gain enable failed with error code: {ret}")
            return

        # Start live video capture
        ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
        if ret != ueye.IS_SUCCESS:
            print(f"CaptureVideo failed with error code: {ret}")
            return

        # Continuous capture loop
        while True:
            # Retrieve image data
            array = np.zeros((sensor_height, sensor_width), dtype=np.uint8)
            ueye.is_CopyImageMem(hCam, pcImageMemory, MemID, array.ctypes.data)

            # Display the image
            cv2.imshow("uEye Camera", array)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup resources
        if pcImageMemory:
            ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
        ueye.is_ExitCamera(hCam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
