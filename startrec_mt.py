import os
import sys
import cv2
import numpy as np
from time import gmtime, strftime, sleep
import socket
import time
import uuid
from xml.etree import ElementTree
from multiprocessing import Process
from threading import Thread, Event
from queue import Queue

def retrieve_camera_IP(serial, timeout=0):
    src_ip = '0.0.0.0'
    src_port = 37020
    dst_ip = '239.255.255.250'
    dst_port = src_port
    uid = str(uuid.uuid1()).upper()
    inquiry = (''
        + '<?xml version="1.0" encoding="utf-8"?>'
        + '<Probe>'
        +     '<Uuid>{}</Uuid>'
        +     '<Types>inquiry</Types>'
        + '</Probe>').format(uid).encode('utf-8')
    prev = 0
    start = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((src_ip, src_port)) 
        sock.setblocking(0)
        data = None
        while True:
            now = time.time()
            if timeout > 0 and now - start > timeout:
                return None
            if now - prev >= 1.0:
                sock.sendto(inquiry, (dst_ip, dst_port))
                prev = now
            try:
                data, addr = sock.recvfrom(1024)
                xml = data.decode("utf-8")
                tree = ElementTree.ElementTree(ElementTree.fromstring(xml))
                root = tree.getroot()
                if root.tag == 'ProbeMatch' and root.find('Uuid').text == uid and root.find('DeviceSN').text.endswith(serial):
                    return root.find('IPv4Address').text
            except BaseException as e:
                pass
class VideoStreamProcessor():
    def __init__(self, ip_cam):
        self.capture = cv2.VideoCapture(ip_cam)
        self.thread = Thread(target=self.update, args=())
        self.frame = None
        self.status = None
        self.thread.daemon = True
        self.thread.start()
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            sleep(.01)
    def save_frame(self, shots_path, count):
        if isinstance(self.frame, np.ndarray):
            file_name = '{}{}_{}.png'.format(shots_path, strftime('%Y%m%d_%H%M%S', gmtime()), count)
            cv2.imwrite(file_name, self.frame)
    def stop(self):
        self.capture.release()
        self.thread.join()
def main(): 
    #---python startrec.py shots .25 100---
    #ip = retrieve_camera_IP('D17274300', timeout=10)
    #print('found camera IP: ', ip)
    #ip_cam = 'rtsp://admin:SWSCFX@{}'.format(ip)
    ip = '192.168.3.8' 
    print('got camera IP: ', ip)
    ip_cam = 'rtsp://admin:biom2019@{}'.format(ip)
    shots_path = './{}/'.format(sys.argv[1]) #shots
    delay = float(sys.argv[2]) #.25
    max_count = int(sys.argv[3]) #100
    video_stream = VideoStreamProcessor(ip_cam)
    count = 0
    while True:
        try:
            video_stream.save_frame(shots_path, count)
            count += 1
            sleep(delay)
            if count > max_count:
                video_stream.stop()
                print('video stream stopped')
                break
        except BaseException as e:
            print(e)
    
if __name__ == '__main__':
    main()