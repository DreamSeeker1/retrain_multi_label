# -*-coding:utf-8-*-

import os.path
import random
import sys
import tarfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

MAX_NUM_IMAGES = 2 ** 27 - 1

TRAINING = 'training'
TESTING = 'testing'
VALIDATION = 'validation'

IMAGE_DIR = "images"
IMAGE_LABEL_DIR = 'image_labels_dir'
ALL_LABELS_FILE = "labels.txt"
bottleneck_path = "bottlenecks"
summaries_dir = 'tmp/retrain_logs'
final_tensor_name = 'final_result'
output_graph = 'tmp/output_graph.pb'
model_dir = 'model_dir'

# Model 参数
data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
bottleneck_tensor_size = 2048
input_width = 299
input_height = 299
input_depth = 3
model_file_name = 'classify_image_graph_def.pb'

# 训练参数
train_batch_size = 100
validation_batch_size = 100
test_batch_size = 100
learning_rate = 0.01
eval_step_interval = 10
how_many_training_steps = 2000

CACHED_GROUND_TRUTH_VECTORS = {}
sess, image_lists, class_count, labels = None, None, 0, []


def prepare_file_system():
    # 设置tensorboard写summaries的目录
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)


def maybe_download_and_extract():
    #如果模型不存在，将会下载模型
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 回调函数，用于查看下载进度
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        # urllib模块提供的 urlretrieve()函数。该函数直接将远程数据下载到本地。
        filepath, _ = urlretrieve(data_url, filepath, _progress)

        # 返回文件的系统状态信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# 创建图
def create_model_graph():
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_file_name)

        # 读取训练好的Inception-v3模型
        # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # 加载读取的Inception-v3模型，并返回数据：输入所对应的张量和计算瓶颈层结果所对应的张量。
            bottleneck_tensor, jpeg_data_tensor = (tf.import_graph_def(
                graph_def,
                return_elements=['pool_3/_reshape:0', 'DecodeJpeg/contents:0']
            ))
    return graph, bottleneck_tensor, jpeg_data_tensor


# 将数据集分成训练集、测试集和验证集，测试集验证集各占总数据集的10%
def create_image_lists():
    testing_percentage = validation_percentage = 0.1
    file_list = os.listdir(IMAGE_DIR)

    testing_count = int(len(file_list) * testing_percentage)
    validation_count = int(len(file_list) * validation_percentage)
    return {
        TESTING: file_list[:testing_count],
        VALIDATION: file_list[testing_count:(testing_count + validation_count)],
        TRAINING: file_list[(testing_count + validation_count):]
    }


# 这个函数通过类别名称、所属数据集和图片编号来获取文件地址。
def get_path_by_folder(folder, index, category):
    if category not in image_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = image_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.', category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    full_path = os.path.join(folder, base_name)
    return full_path


# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 判读一下工作路径下是否有模型目录，没有就创建
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的结点取值。
# 可以将这个过程类似的理解为一种特征提取。
def add_final_training_ops(bottleneck_tensor):
    with tf.name_scope('input'):

        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')  # 可以不feed ,缺省使用input
        #正确标签
        ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

    # 定义一层全连接层来解决新的图片分类问题。
    # 训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，
    # 所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    with tf.name_scope('final_training_ops'):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases

    final_tensor = tf.nn.sigmoid(logits, name=final_tensor_name)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_input, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


# 函数作用是计算正确率
def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(result_tensor), ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step


# 这个函数作用是获取一张图片经过Inception-v3模型处理之后的特征向量，并将特征向量存入文件。
def create_bottleneck_file(bottleneck_f_path, index, category, jpeg_data_tensor, bottleneck_tensor):
    tf.logging.info('Creating bottleneck at ' + bottleneck_f_path)
    image_path = get_path_by_folder(IMAGE_DIR, index, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(image_data, jpeg_data_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_f_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)
    return bottleneck_values


# 这个函数会先试图寻找已经计算且保存下来的特征向量，如果找不到调用create_bottleneck_file函数生成文件
def get_or_create_bottleneck(index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径。
    ensure_dir_exists(bottleneck_path)
    bottleneck_f_path = get_path_by_folder(bottleneck_path, index, category) + '.txt'
    if not os.path.exists(bottleneck_f_path):
        return create_bottleneck_file(bottleneck_f_path, index, category, jpeg_data_tensor, bottleneck_tensor)
    # 从文件中获取图片相应的特征向量。
    with open(bottleneck_f_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values


# 生成图片的特征向量
# category 是图片类别（TRAINING, TESTING, VALIDATIO）
# index 是图片在对应类别的数组下标
def cache_bottlenecks(jpeg_data_tensor, bottleneck_tensor):
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_path)
    for category in [TRAINING, TESTING, VALIDATION]:
        category_list = image_lists[category]
        for index, _ in enumerate(category_list):
            get_or_create_bottleneck(index, category, jpeg_data_tensor, bottleneck_tensor)

            how_many_bottlenecks += 1
            if how_many_bottlenecks % 10 == 0:
                print(str(how_many_bottlenecks) + ' bottleneck files created.')


# 获得正确的标签
def get_ground_truth(labels_file):
    if labels_file in CACHED_GROUND_TRUTH_VECTORS:
        ground_truth = CACHED_GROUND_TRUTH_VECTORS[labels_file]
    else:
        with open(labels_file) as f:
            true_labels = f.read().splitlines()
        ground_truth = np.zeros(class_count, dtype=np.float32)

        for index, label in enumerate(labels):
            if label in true_labels:
                ground_truth[index] = 1.0

        CACHED_GROUND_TRUTH_VECTORS[labels_file] = ground_truth

    return ground_truth  # [1,0,0,1,0]


# 这个函数随机获取数据集的howmany个图片(通常为一个batch)的特征值和真实标签
def get_random_cached_bottlenecks(how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []

    # 随机恢复howmany个样本的特征向量值
    for unused_i in range(how_many):
        image_index = random.randrange(MAX_NUM_IMAGES + 1)
        bottleneck = get_or_create_bottleneck(image_index, category, jpeg_data_tensor, bottleneck_tensor)
        labels_file = get_path_by_folder(IMAGE_LABEL_DIR, image_index, category) + '.txt'
        ground_truth = get_ground_truth(labels_file)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


# 保存计算图到文件
def save_graph_to_file(graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def main(_):
    global sess, image_lists, labels, class_count
    prepare_file_system()
    maybe_download_and_extract()  # 下载inception-2015-12-05.tgz 并解压
    graph, bottleneck_tensor, jpeg_data_tensor = create_model_graph()   # 把瓶颈层张量，解码后的图片张量，以及inception的图给返回

    image_lists = create_image_lists()     # 创建所有图片的列表，并把图片分成训练集测试集和验证集

    with open(ALL_LABELS_FILE) as f:
        labels = f.read().splitlines()     # 所有的标签
    class_count = len(labels)              # 标签总数

    with tf.Session(graph=graph) as session:  # 启动会话
        sess = session
        # 生成所有图的特征向量
        cache_bottlenecks(jpeg_data_tensor, bottleneck_tensor)

        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(
            bottleneck_tensor)

        evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程
        for i in range(how_many_training_steps):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                train_batch_size, TRAINING, jpeg_data_tensor, bottleneck_tensor)
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            if (i % eval_step_interval) == 0 or i + 1 == how_many_training_steps:
                # 获得正确率和loss
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],
                                                               feed_dict={bottleneck_input: train_bottlenecks,
                                                                          ground_truth_input: train_ground_truth})
                print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))

                # 验证
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    validation_batch_size, VALIDATION, jpeg_data_tensor, bottleneck_tensor)

                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                print('%s: Step %d: Validation accuracy = %.1f  (N=%d)' %
                      (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))

        # 测试
        test_bottlenecks, test_ground_truth = (
            get_random_cached_bottlenecks(test_batch_size, TESTING, jpeg_data_tensor, bottleneck_tensor))
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        save_graph_to_file(graph, output_graph)


if __name__ == '__main__':
    tf.app.run(main=main)
