import os
import cv2
import sys
import dlib
import pickle
import shutil
import json
import numpy as np
import pandas as pd
import skimage.transform as tr
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
from keras.models import load_model
from cnn import get_model, load_weights
print('available devices: ', [x.name for x in K.tensorflow_backend.device_lib.list_local_devices()])

IMG_SIZE = 96
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
DATA_PATH = './data/'
DLIB_PREDICTOR_PATH = '{}shape_predictor_68_face_landmarks.dat'.format(DATA_PATH)
FACE_TEMPLATE_PATH = '{}face_template.npy'.format(DATA_PATH)
DLIB_DETECTOR = dlib.get_frontal_face_detector()
DLIB_PREDICTOR = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
FACE_TEMPLATE = np.load(FACE_TEMPLATE_PATH)
print('global variables initialized...')
model = load_model('{}model.h5'.format(DATA_PATH), custom_objects={'tf': tf})
print('model loaded...')

def get_landmarks(img, face_rect):
    points = DLIB_PREDICTOR(img, face_rect)
    return np.array(list(map(lambda p: [p.x, p.y], points.parts())))
def align_face(img, face_rect, *, dim=96, border=0, mask=INNER_EYES_AND_BOTTOM_LIP):
    landmarks = get_landmarks(img, face_rect)
    proper_landmarks = border + dim * FACE_TEMPLATE[mask]
    A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
    B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
    T = np.linalg.solve(A, B).T
    wrapped = tr.warp(img,
                      tr.AffineTransform(T).inverse,
                      output_shape=(dim + 2 * border, dim + 2 * border),
                      order=3,
                      mode='constant',
                      cval=0,
                      clip=True,
                      preserve_range=True)
    return wrapped
def get_faces_rects(img, upscale_factor=1):
    try:
        face_rects = list(DLIB_DETECTOR(img, upscale_factor))
    except:
        face_rects = []
    return face_rects
def get_face_xywh(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.right() - x
    h = face_rect.bottom() - y
    return (x, y, w, h)
def image_to_embedding(image, model):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255., decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding
def find_identity(img, database, model, level):
    min_dist = 100.
    identity = None
    img_embedding = image_to_embedding(img, model)
    for (label, data) in database.items():
        dist = np.linalg.norm(data[4] - img_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = data[0]
    if min_dist < level:
        return identity, min_dist
    else:
        return None, min_dist
def process_shots(path_shots, start_time, database, model, logs_path, path_shots_prcd, level=.7):
    img_files = os.listdir(path_shots)
    for img_file in img_files:
        print('image processing: ', img_file)
        results = []
        file_name = '{}{}'.format(path_shots, img_file)
        img = cv2.imread(file_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        font_scale = max(img.shape[0], img.shape[0]) / 1000
        line_scale = round(3 * font_scale)
        shift_x = 5
        shift_y = int(shift_x * font_scale)
        faces = get_faces_rects(img, 1)
        for face_rect in faces:
            img_face = align_face(img, face_rect)
            (x, y, w, h) = get_face_xywh(face_rect)
            identity, min_dist = find_identity(img_face, database, model, level=level)
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            if identity:
                result = {'identity': str(identity), 
                          'min dist': str(min_dist), 
                          'level': str(level),
                          'rectangle': ' '.join(str(x) for x in [x1, y1, x2, y2])}
                print(str(identity), str(min_dist))
            else:
                result = {'identity': 'Unknown', 
                          'min dist': 'None', 
                          'level': str(level),
                          'rectangle': ' '.join(str(x) for x in [x1, y1, x2, y2])}
                print('Unknown')
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, line_scale)
            cv2.putText(img, result['identity'], 
                        (x1 + shift_x, y1 - shift_y), 
                        font, font_scale, color, line_scale)
            results.append(result)
        cv2.imwrite(file_name, img)
        log_file_path = '{}{}.txt'.format(logs_path, img_file[:img_file.find('.')])
        with open(log_file_path, 'w') as file:
            json.dump(results, file)
        shutil.move('{}{}'.format(path_shots, img_file), '{}{}'.format(path_shots_prcd, img_file))
def translit(text):
    symbols = ('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ',
               'abvgdeejzijklmnoprstufhccss_yieuaABVGDEEJZIJKLMNOPRSTUFHCCSS_YIEUA ')
    tr = {ord(a):ord(b) for a, b in zip(*symbols)}
    return text.translate(tr)
def get_database(database, model, path):
    label = 900000
    files = os.listdir(path)
    for file in tqdm(files):
        label += 1 
        name = os.path.splitext(os.path.basename(file))[0]
        department, subject = 'Updated', 'Updated'
        path_file = '{}{}'.format(path, file)
        upd_img = cv2.imread(path_file, 1)
        faces = get_faces_rects(upd_img, 1)
        for face_rect in faces:
            img_face = align_face(upd_img, face_rect)
            img_face = img_face.astype(int)
            database[label] = (name, department, subject, path_file, image_to_embedding(img_face, model))
    return database
def main():
    #python process.py nodbcreate photos shots shotsprcd logs starttime .7
    database_create = True if sys.argv[1] == 'dbcreate' else False #dbcreate
    photos_path = './{}/'.format(sys.argv[2]) #photos
    shots_path = './{}/'.format(sys.argv[3]) #shots
    shots_prcd_path = './{}/'.format(sys.argv[4]) #shots
    logs_path = './{}/'.format(sys.argv[5]) #logs
    start_time = sys.argv[6]
    level = float(sys.argv[7])
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(shots_prcd_path):
        os.makedirs(shots_prcd_path)
    if database_create:
        database = {}
        database = get_database(database, model, photos_path)
        with open('{}database.pkl'.format(DATA_PATH), 'wb') as file:
            pickle.dump(database, file)
        print('database created...')
    with open('{}database.pkl'.format(DATA_PATH), 'rb') as file:
        database = pickle.load(file)
    print('database ready...')    
    process_shots(shots_path, start_time, database, model, logs_path, shots_prcd_path, level)

if __name__ == '__main__':
    main()