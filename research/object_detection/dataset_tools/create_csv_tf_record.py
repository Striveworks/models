"""
Adapted from
https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

Usage:
  # From tensorflow/models/
  # Create train data:
  python create_csv_tf_record.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python create_csv_tf_record.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
import os
import io
import pandas as pd
import tensorflow as tf
import argparse

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
from object_detection.utils.label_map_util import get_label_map_dict
from object_detection.core.standard_fields import TfExampleFields


def class_text_to_int(row_label, label_map_path):
    lm_dict = get_label_map_dict(label_map_path)
    if row_label in lm_dict.keys():
        return lm_dict[row_label]
    return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_path, relative_bboxes=False):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    if group.filename.endswith('png'):
        image_format = b'png'
    else:
        image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if relative_bboxes:
            scale_x, scale_y = 1, 1
        else:
            scale_x, scale_y = width, height
        xmins.append(row['xmin'] / scale_x)
        xmaxs.append(row['xmax'] / scale_x)
        ymins.append(row['ymin'] / scale_y)
        ymaxs.append(row['ymax'] / scale_y)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_path))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        TfExampleFields.height: dataset_util.int64_feature(height),
        TfExampleFields.width: dataset_util.int64_feature(width),
        TfExampleFields.filename: dataset_util.bytes_feature(filename),
        TfExampleFields.source_id: dataset_util.bytes_feature(filename),
        TfExampleFields.image_encoded: dataset_util.bytes_feature(encoded_img),
        TfExampleFields.image_format: dataset_util.bytes_feature(image_format),
        TfExampleFields.object_bbox_xmin: dataset_util.float_list_feature(xmins),
        TfExampleFields.object_bbox_xmax: dataset_util.float_list_feature(xmaxs),
        TfExampleFields.object_bbox_ymin: dataset_util.float_list_feature(ymins),
        TfExampleFields.object_bbox_ymax: dataset_util.float_list_feature(ymaxs),
        TfExampleFields.object_class_text: dataset_util.bytes_list_feature(classes_text),
        TfExampleFields.object_class_label: dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(args):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.images_path)
    examples = pd.read_csv(args.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, args.label_map_path,
                                       relative_bboxes=args.relative_bboxes)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), args.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_input', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--label_map_path', type=str)
    parser.add_argument('--images_path', type=str)
    parser.add_argument('--relative_bboxes', action='store_true')

    args = parser.parse_args()
    main(args)
