import tensorflow as tf
import numpy as np
import facenet
import os
import align
from align import detect_face
import cv2
# import psycopg2
import sys
from keras.backend.tensorflow_backend import set_session

# some constants from facenet
minsize = 20
threshold = [0.6, 0.8, 0.9]
factor = 0.709
margin = 0
input_image_size = 160

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess = tf.Session(config=config)
set_session(sess)

pnet, rnet, onet = detect_face.create_mtcnn(sess, './align')

facenet.load_model("./20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.95:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces


def getEmbedding(resized):
    """
    Получаем эмбеддинги 

    Args:
        resized (str): Путь в os к изображению человека
    Returns:
        img (bytes): Переменная соедержащая изображение
    """
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def readImage(path_to_jpg):
    fin = None
    try:
        fin = open(path_to_jpg, "rb")
        img = fin.read()
        return img
    except IOError as e:
        print(f'Error {e.args[0]}, {e.args[1]}')
        sys.exit(1)
    finally:
        if fin:
            fin.close()


def compare_with_all_faces(embedding_new, all_embeddings, known_face_names):
    distances = []
    for embedding in all_embeddings:
        dist = np.sqrt(np.sum(np.square(np.subtract(embedding_new, embedding))))
        distances.append(dist)
        if min(distances) < limit:
            index_of_min_dist = np.argmin(distances)
            name = known_face_names[index_of_min_dist]
            # в этой ветке значит что есть СОВПАДЕНИЕ!
        else:
            name = f"person_{len(known_face_names)}"
    return name


limit = 1.10    # set yourself to meet your requirement
font = cv2.FONT_HERSHEY_DUPLEX # setting font for putting text names

"""
Участок кода по формированию БД из отпарсенных пользователей Vk and Fb.
Струкутруа БД:
id [PK Serial]    id vk or id fb    image (bytea)    embeddings (text)
"""


"""
Место для формирования имён и эмбеддингов из существующих изображений из БД PostgresQL
"""
# with postgresConnection:

#     cur = postgresConnection.cursor()
#     cur.execute("SELECT * FROM kindergarden_alpha")

#     rows = cur.fetchall()

#     for row in rows:
#         known_face_names.append(row[1])
#         embeddings_from_db.append(row[3])
#         buf = [i.replace("{{", "[").replace("}}", "]") for i in embeddings_from_db]
#         known_face_embeddings = [eval(i) for i in buf]

"""
Конец участка кода по формированию данных из БД PostgresQL
"""

def compare_img_with_db(image):
    image = cv2.imread(image)
    face = getFace(image)

    face_names = []
    face_locations = []
    face_embeddings = []

    for person in face:
        # Добавляем в face_locations точки прямоугольника лица
        face_locations.append((person['rect'][0], person['rect'][1], person['rect'][2], person['rect'][3]))
        # Добавляем в face_embeddings эмбеддинг этого лица
        face_embeddings.append(person['embedding'])

        # Определяем имя уже известное или person_X
        name = compare_with_all_faces(person['embedding'], known_face_embeddings, known_face_names)
        # Добавляем в список face_names новое имя
        face_names.append(name)

        if name not in known_face_names:
            # Добавляем новое лицо и его имя в списки известных имён и эмбеддингов
            known_face_names.append(name)
            known_face_embeddings.append(person['embedding'])

            # Конвертируем массив эмбеддинга в нужный формат для PostgreSQL
            array_of_embedding = list(np.array(person['embedding']))
            pg_format_embedding = np.array(array_of_embedding).tolist()

            # Смотрим что за участок изображения, где было распознано лицо человека
            cropped = frame[person['rect'][1]:person['rect'][3], person['rect'][0]:person['rect'][2]]
            # Создаём новое изобржение этого лица в паке images
            cv2.imwrite(f"images/{name}.jpg", cropped)






