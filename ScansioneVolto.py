###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
# Autore: Crivellaro Diego                                                    #
#                                                                             #
#                                                                             #
# Data:   2016                                                                #
#                                                                             #
#                                                                             #
###############################################################################


### Imports ###################################################################

from picamera.array import PiRGBArray
from picamera import PiCamera
from functools import partial

import multiprocessing as mp
import cv2
import os
import time


### Setup #####################################################################

os.putenv( 'SDL_FBDEV', '/dev/fb0' )

resX = 640
resY = 480

cx = resX / 2
cy = resY / 2

os.system( "echo 0=150 > /dev/servomotori" )
os.system( "echo 1=150 > /dev/servomotori" )

xdeg = 150
ydeg = 150


# Setup the camera
camera = PiCamera()
camera.resolution = ( resX, resY )
camera.framerate = 60

# Use this as our output
rawCapture = PiRGBArray( camera, size=( resX, resY ) )

# The face cascade file to be used
face_cascade = cv2.CascadeClassifier('/home/ai/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')

t_start = time.time()
fps = 0


### Helper Functions ##########################################################

def get_faces( img ):

    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale( gray )

    return faces, img

def draw_frame( img, faces ):

    global xdeg
    global ydeg
    global fps
    global time_t

    # Draw a rectangle around every face
    for ( x, y, w, h ) in faces:

        cv2.rectangle( img, ( x, y ),( x + w, y + h ), ( 200, 255, 0 ), 2 )
        cv2.putText(img, "Rilevamento_" + str( len( faces ) ), ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )

        tx = x + w/2
        ty = y + h/2

        if   ( cx - tx > 15 and xdeg <= 190 ):
            xdeg += 1
            os.system( "echo 0=" + str( xdeg ) + " > /dev/servomotori" )
        elif ( cx - tx < -15 and xdeg >= 110 ):
            xdeg -= 1
            os.system( "echo 0=" + str( xdeg ) + " > /dev/servomotori" )

        if   ( cy - ty > 15 and ydeg >= 110 ):
            ydeg -= 1
            os.system( "echo 1=" + str( ydeg ) + " > /dev/servomotori" )
        elif ( cy - ty < -15 and ydeg <= 190 ):
            ydeg += 1
            os.system( "echo 1=" + str( ydeg ) + " > /dev/servomotori" )

    # Calculate and show the FPS
    fps = fps + 1
    sfps = fps / (time.time() - t_start)
    cv2.putText(img, "Fotogrammi : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )

    cv2.imshow( "Frame", img )
    cv2.waitKey( 1 )


### Main ######################################################################

if __name__ == '__main__':

    pool = mp.Pool( processes=4 )
    fcount = 0

    camera.capture( rawCapture, format="bgr" )

    r1 = pool.apply_async( get_faces, [ rawCapture.array ] )
    r2 = pool.apply_async( get_faces, [ rawCapture.array ] )
    r3 = pool.apply_async( get_faces, [ rawCapture.array ] )
    r4 = pool.apply_async( get_faces, [ rawCapture.array ] )

    f1, i1 = r1.get()
    f2, i2 = r2.get()
    f3, i3 = r3.get()
    f4, i4 = r4.get()

    rawCapture.truncate( 0 )

    for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
        image = frame.array

        if   fcount == 1:
            r1 = pool.apply_async( get_faces, [ image ] )
            f2, i2 = r2.get()
            draw_frame( i2, f2 )

        elif fcount == 2:
            r2 = pool.apply_async( get_faces, [ image ] )
            f3, i3 = r3.get()
            draw_frame( i3, f3 )

        elif fcount == 3:
            r3 = pool.apply_async( get_faces, [ image ] )
            f4, i4 = r4.get()
            draw_frame( i4, f4 )

        elif fcount == 4:
            r4 = pool.apply_async( get_faces, [ image ] )
            f1, i1 = r1.get()
            draw_frame( i1, f1 )

            fcount = 0

        fcount += 1

        rawCapture.truncate( 0 )
