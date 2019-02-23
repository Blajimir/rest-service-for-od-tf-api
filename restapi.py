from flask import Flask, jsonify, request
import os
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

sys.path.append("/pydir/models/research")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# from object_detection.utils import visualization_utils as vis_util

LIBRARY_FREEZE_MODEL = '/pydir/models/research/object_detection/freeze-models'
CURRENT_FREEZE_MODEL_DIR = os.path.join(LIBRARY_FREEZE_MODEL, 'current')
MODELS_LIST_FILE = os.path.join(LIBRARY_FREEZE_MODEL, 'freeze-model-list.json')
model_list = []
DEFAULT_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28'
#'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
CURRENT_MODEL = None
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
LABELS_DIR = '/pydir/models/research/object_detection/data'
LABELS_NAME = 'mscoco_label_map.pbtxt'
PATH_TO_FROZEN_GRAPH = os.path.join(CURRENT_FREEZE_MODEL_DIR, FROZEN_GRAPH_NAME)

ALLOWED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']

category_index = None
detection_graph = None

app = Flask(__name__)


def create_freeze_model_list():
    global model_list
    model_list = []
    for f in os.listdir(LIBRARY_FREEZE_MODEL):
        fpath = os.path.join(LIBRARY_FREEZE_MODEL, f)
        if os.path.isfile(fpath):
            model_list.append({"name": f.split(".")[0], "path": fpath})


def unpack_model(mname=None):
    global model_list
    global PATH_TO_FROZEN_GRAPH
    global CURRENT_MODEL
    global CURRENT_FREEZE_MODEL_DIR
    global FROZEN_GRAPH_NAME
    global LABELS_DIR
    if os.path.exists(PATH_TO_FROZEN_GRAPH):
        os.remove(PATH_TO_FROZEN_GRAPH)
    name = mname if mname else CURRENT_MODEL
    path = None
    for item in model_list:
        if item['name'] == name:
            path = item['path']
            break
    tar_file = tarfile.open(path)
    for file in tar_file.getmembers():
        file.name = os.path.basename(file.name)
        if FROZEN_GRAPH_NAME == file.name:
            tar_file.extract(file, CURRENT_FREEZE_MODEL_DIR)
        elif allowed_file(file.name, ['pbtxt']):
            tar_file.extract(file, LABELS_DIR)


def load_graph():
    # Load a (frozen) Tensorflow model into memory.
    global detection_graph
    global category_index
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    # Loading label map
    category_index = label_map_util.create_category_index_from_labelmap(os.path.join(LABELS_DIR, LABELS_NAME),
                                                                        use_display_name=True)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict


def is_name_exist(name):
    result = False
    for model in model_list:
        if model['name'] == name:
            result = True
            break
    return result


def init(app):
    create_freeze_model_list()
    global CURRENT_MODEL
    global DEFAULT_MODEL
    global CURRENT_FREEZE_MODEL_DIR
    global PATH_TO_FROZEN_GRAPH
    CURRENT_MODEL = DEFAULT_MODEL
    if not os.path.exists(CURRENT_FREEZE_MODEL_DIR):
        os.makedirs(CURRENT_FREEZE_MODEL_DIR)
    if not os.path.exists(PATH_TO_FROZEN_GRAPH):
        unpack_model()
    load_graph()


def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extension


def load_image_file_into_numpy_array(fimage):
    import io
    image = Image.open(io.BytesIO(fimage.read()))
    (im_width, im_height) = image.size
    arr = np.array(image.getdata())
    if arr.shape[1] == 4:
        image.load()  # required for png.split()
        bgrnd = Image.new("RGB", image.size, (255, 255, 255))
        bgrnd.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        arr = np.array(bgrnd.getdata())
    return arr.reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def prepare_npfloat32(fl):
    return round(np.float32(fl).item(), 8)


def prepare_output_dict(out_dict):
    global category_index
    result = {"Predictions": []}
    if out_dict and out_dict['num_detections'] > 0:
        for i in range(out_dict['num_detections']):
            class_name = category_index[out_dict['detection_classes'][i]]['name']
            region = {"Left": prepare_npfloat32(out_dict['detection_boxes'][i][0]),
                      "Top": prepare_npfloat32(out_dict['detection_boxes'][i][1]),
                      "Width": prepare_npfloat32(out_dict['detection_boxes'][i][2] - out_dict['detection_boxes'][i][0]),
                      "Height": prepare_npfloat32(out_dict['detection_boxes'][i][3] - out_dict['detection_boxes'][i][1])}
            result["Predictions"].append({"Probability": prepare_npfloat32(out_dict['detection_scores'][i]),
                                          "TagName": class_name,
                                          "Region": region})
    return result


init(app)


@app.route("/")
def health():
    return jsonify({"health": "up"})


@app.route("/api/model", methods=['GET'])
def get_model():
    global CURRENT_MODEL
    return jsonify({"modelName": CURRENT_MODEL})


@app.route("/api/model/list", methods=['GET'])
def get_model_list():
    return jsonify(model_list)


@app.route("/api/model", methods=['POST'])
def set_model():
    global CURRENT_MODEL
    model_name = request.form['modelName']
    result = {"status": "ok"}
    if not model_name:
        result = {"status": "error", "comment": "You don't get model name"}
    elif not is_name_exist(model_name):
        result = {"status": "error", "comment": "model name is not exist"}
    else:
        CURRENT_MODEL = model_name
        load_graph()
    return jsonify(result)


@app.route("/api/model", methods=['PUT'])
def upload_model():
    # TODO: подумать стоит ли сделать возможность перезаписи файлов
    result = {"status": ":("}
    if 'file' not in request.files:
        result = {"status": "error", "comment": "Image file not found"}
    file = request.files['file']
    if not allowed_file(file.filename, ['tar.gz', 'zip']):
        result = {"status": "error", "comment": "File with this extension doesn't support"}
    elif file:
        fn = os.path.basename(file.filename)
        if not os.path.exists(os.path.join(LIBRARY_FREEZE_MODEL, fn)):
            file.save(os.path.join(LIBRARY_FREEZE_MODEL, fn))
            create_freeze_model_list()
            result = {"status": "ok"}
        else:
            result = {"status": "error", "comment": "file already exists"}
    return jsonify(result)


@app.route("/api/predict", methods=['POST'])
def predict():
    result = {"status": ":("}
    global detection_graph
    if 'file' not in request.files:
        result = {"status": "error", "comment": "Image file not found"}
    file = request.files['file']
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        result = {"status": "error", "comment": "File with this extension doesn't support"}
    elif file:
        image_np = load_image_file_into_numpy_array(file)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        result = prepare_output_dict(output_dict)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
