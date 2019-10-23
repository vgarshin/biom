import os
import sys
import cv2
from time import gmtime, strftime, sleep

def video_to_frames(ip_cam, shots_path, delay=1, max_counts=10):
    cap = cv2.VideoCapture(ip_cam)
    count = 0
    while cap.isOpened():
        success, img = cap.read()
        if success:
            file_name = '{}{}_{}.png'.format(shots_path, strftime('%Y%m%d_%H%M%S', gmtime()), count)
            #cv2.imshow('frame', img)
            cv2.imwrite(file_name, img)
            count += 1
        else:
            break    
        sleep(delay)
        if count >= max_counts:
            break
    cap.release()
def main():
	#python startrec.py rtsp://admin:SWSCFX@192.168.10.159 shots 1 10
    ip_cam = sys.argv[1]
    shots_path = './{}/'.format(sys.argv[2]) #shots
    delay = float(sys.argv[3])
    max_counts = int(sys.argv[4])
    video_to_frames(ip_cam, shots_path, delay, max_counts)

if __name__ == '__main__':
    main()

