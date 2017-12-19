# -*-coding:utf-8-*-

import os.path
import random
import sys
import tarfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

TRAINING = 'training'
TESTING = 'testing'
VALIDATION = 'validation'


class Retrain(object):

    IMAGE_DIR = "images"
    IMAGE_LABEL_DIR = 'image_labels_dir'
    ALL_LABELS_FILE = "labels.txt"
    bottleneck_path = "bottlenecks"
    intermediate_store_frequency = 0
    summaries_dir = 'tmp/retrain_logs'
    intermediate_output_graphs_dir = 'tmp/intermediate_graph/'
    final_tensor_name = 'final_result'
    output_graph = 'tmp/output_graph.pb'
    flip_left_right = False

    CACHED_GROUND_TRUTH_VECTORS = {}
    image_lists = None
    model = None

    def __init__(self, train_batch_size=100, random_crop=0, random_scale=0,
                 validation_batch_size=100, test_batch_size=100,
                 output_labels='tmp/output_labels.txt', model_dir='model_dir', learning_rate=0.01,
                 eval_step_interval=10, how_many_training_steps=1000, random_brightness=0):
        self.sess = None
        self.train_batch_size = train_batch_size
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.test_batch_size = test_batch_size
        self.validation_batch_size = validation_batch_size
        self.output_labels = output_labels
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.eval_step_interval = eval_step_interval
        self.how_many_training_steps = how_many_training_steps
        self.random_brightness = random_brightness

    def create_image_lists(self):
        testing_percentage = validation_percentage = 0.1

        if not gfile.Exists(self.IMAGE_DIR):
            print("Image directory not found.")
            exit(1)

        file_list = os.listdir(self.IMAGE_DIR)  # 输出的是图片文件的文件名
        if not (20 < len(file_list) < MAX_NUM_IMAGES_PER_CLASS):
            tf.logging.warning('WARNING: 图片数量异常.')
            print('WARNING: 图片数量异常.')
            exit(1)

        testing_count = int(len(file_list) * testing_percentage)
        validation_count = int(len(file_list) * validation_percentage)
        self.image_lists = {
            TESTING: file_list[:testing_count],
            VALIDATION: file_list[testing_count:(testing_count + validation_count)],
            TRAINING: file_list[(testing_count + validation_count):],
        }

    def get_image_labels_path(self, index, category):
        if category not in self.image_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = self.image_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.', category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        full_path = os.path.join(self.IMAGE_LABEL_DIR, base_name)
        full_path += '.txt'
        return full_path

        # 获取单张图片的地址
    def get_image_path(self, index, category):
        return self.get_path_by_folder(self.IMAGE_DIR, index, category)

    def get_path_by_folder(self, folder, index, category):
        if category not in self.image_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = self.image_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.', category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        full_path = os.path.join(folder, base_name)
        return full_path

    def get_bottleneck_path(self, index, category):
        return self.get_path_by_folder(self.bottleneck_path, index, category) + '.txt'

    def create_model_graph(self):  # 读取训练好的Inception-v3模型来创建graph
        with tf.Graph().as_default() as graph:
            model_path = os.path.join(self.model_dir, self.model['model_file_name'])
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()  # 把默认图付给graph_def
                graph_def.ParseFromString(f.read())  # 把inception模型图付给默认图
                bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(  # import_graph_def将graph_def的图导入到Python中
                    graph_def,
                    name='',
                    return_elements=[
                        self.model['bottleneck_tensor_name'],
                        self.model['resized_input_tensor_name'],
                    ]))
        return graph, bottleneck_tensor, resized_input_tensor


    def run_bottleneck_on_image(self, image_data, image_data_tensor,
                                decoded_image_tensor, resized_input_tensor,
                                bottleneck_tensor):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = self.sess.run(decoded_image_tensor,
                                        {image_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = self.sess.run(bottleneck_tensor,
                                     {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def maybe_download_and_extract(self):
        dest_directory = self.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.model['data_url'].split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                # sys.stdout.write 跟print差不多，可以理解成print是它的一个封装
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename,
                                  float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urlretrieve(self.model['data_url'], filepath, _progress)
            # _progress 是个回调函数，根据这个函数显示当前下载进度
            # urllib的urlretrieve函数是直接将远程数据下载到本地。参数一：远程路径，参数二：本地路径，参数三：回调函数

            statinfo = os.stat(filepath) # stat 系统调用时用来返回相关文件的系统状态信息的。
            tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                            'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory) #解压文件

    @staticmethod
    def ensure_dir_exists(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def create_bottleneck_file(self, bottleneck_f_path, index,
                               category, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor):
        """Create a single bottleneck file."""
        tf.logging.info('Creating bottleneck at ' + bottleneck_f_path)
        image_path = self.get_image_path(index, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        try:
            bottleneck_values = self.run_bottleneck_on_image(
                image_data, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor)
        except Exception as e:
            raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                         str(e)))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_f_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    def get_or_create_bottleneck(self, index,
                                 category, jpeg_data_tensor,
                                 decoded_image_tensor, resized_input_tensor,
                                 bottleneck_tensor):
        self.ensure_dir_exists(self.bottleneck_path)
        bottleneck_path = self.get_bottleneck_path(index, category)
        if not os.path.exists(bottleneck_path):
            self.create_bottleneck_file(bottleneck_path, index,
                                        category, jpeg_data_tensor,
                                        decoded_image_tensor, resized_input_tensor,
                                        bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        try:
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        except ValueError:
            tf.logging.warning('Invalid float found, recreating bottleneck')

            self.create_bottleneck_file(bottleneck_path, index, category, jpeg_data_tensor,
                                        decoded_image_tensor, resized_input_tensor,
                                        bottleneck_tensor)
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            # Allow exceptions to propagate here, since they shouldn't happen after a
            # fresh creation
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values

    def cache_bottlenecks(self, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
        how_many_bottlenecks = 0
        self.ensure_dir_exists(self.bottleneck_path)
        for category in [TRAINING, TESTING, VALIDATION]:
            category_list = self.image_lists[category]
            for index, unused_base_name in enumerate(category_list):
                self.get_or_create_bottleneck(
                    index, category,
                    jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')

    def get_ground_truth(self, labels_file, labels, class_count):
        if labels_file in self.CACHED_GROUND_TRUTH_VECTORS.keys():
            ground_truth = self.CACHED_GROUND_TRUTH_VECTORS[labels_file]
        else:
            with open(labels_file) as f:
                true_labels = f.read().splitlines()
            ground_truth = np.zeros(class_count, dtype=np.float32)

            for index, label in enumerate(labels):
                if label in true_labels:
                    ground_truth[index] = 1.0

            self.CACHED_GROUND_TRUTH_VECTORS[labels_file] = ground_truth

        return ground_truth

    def get_random_cached_bottlenecks(self, how_many, category,
                                      jpeg_data_tensor,
                                      decoded_image_tensor, resized_input_tensor,
                                      bottleneck_tensor, labels):
        # class_count = len(image_lists.keys())
        class_count = len(labels)
        bottlenecks = []
        ground_truths = []
        filenames = []

        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = self.get_image_path(image_index, category)
            bottleneck = self.get_or_create_bottleneck(
                image_index, category,
                jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor)
            labels_file = self.get_image_labels_path(image_index, category)

            # ground_truth = np.zeros(class_count, dtype=np.float32)
            # ground_truth[label_index] = 1.0
            ground_truth = self.get_ground_truth(labels_file, labels, class_count)
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)

        return bottlenecks, ground_truths, filenames

    def get_random_distorted_bottlenecks(self, how_many, category, input_jpeg_tensor,
            distorted_image, resized_input_tensor, bottleneck_tensor, labels):
        class_count = len(labels)
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_index, category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            # Note that we materialize the distorted_image_data as a numpy array before
            # sending running inference on the image. This involves 2 memory copies and
            # might be optimized in other implementations.
            distorted_image_data = self.sess.run(distorted_image,
                                            {input_jpeg_tensor: jpeg_data})
            bottleneck_values = self.sess.run(bottleneck_tensor,
                                         {resized_input_tensor: distorted_image_data})
            bottleneck_values = np.squeeze(bottleneck_values)
            labels_file = self.get_image_labels_path(image_index, category)
            ground_truth = self.get_ground_truth(labels_file, labels, class_count)

            bottlenecks.append(bottleneck_values)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths

    def should_distort_images(self):
        return self.flip_left_right or (self.random_crop != 0) or (self.random_scale != 0)\
               or (self.random_brightness != 0)

    def add_input_distortions(self):
        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=self.model['input_depth']) #将图片变成整形张量
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32) #将uint8变成浮点型
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0) # 0
        margin_scale = 1.0 + (self.random_crop / 100.0)
        resize_scale = 1.0 + (self.random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                               minval=1.0,
                                               maxval=resize_scale)
        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_width = tf.multiply(scale_value, self.model['input_width'])
        precrop_height = tf.multiply(scale_value, self.model['input_height'])
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                    precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(
            precropped_image_3d, [self.model['input_height'], self.model['input_width'], self.model['input_depth']])
        if self.flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (self.random_brightness / 100.0)
        brightness_max = 1.0 + (self.random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                             minval=brightness_min,
                                             maxval=brightness_max)
        brightened_image = tf.multiply(flipped_image, brightness_value)
        offset_image = tf.subtract(brightened_image, self.model['input_mean'])
        mul_image = tf.multiply(offset_image, 1.0 / self.model['input_std'])
        distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
        return jpeg_data, distort_result

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def add_final_training_ops(self, class_count, bottleneck_tensor):
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[None, self.model['bottleneck_tensor_size']],
                name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(tf.float32,
                                                [None, class_count],
                                                name='GroundTruthInput')

        # Organizing the following ops as `final_training_ops` so they're easier
        # to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [self.model['bottleneck_tensor_size'], class_count], stddev=0.001)

                layer_weights = tf.Variable(initial_value, name='final_weights')

                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)
                # 这里用的sigmoid激活函数 为什么多标签用sigmoid
        final_tensor = tf.nn.sigmoid(logits, name=self.final_tensor_name)
        tf.summary.histogram('activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor)

    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.round(result_tensor), ground_truth_tensor)
                # 我们是多个标签 所以不能用argmax返回等于1的index
                # prediction = tf.argmax(result_tensor, 1)
                # correct_prediction = tf.equal(
                #     prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step

    def save_graph_to_file(self, graph, graph_file_name):
        output_graph_def = graph_util.convert_variables_to_constants(
            self.sess, graph.as_graph_def(), [self.final_tensor_name])
        with gfile.FastGFile(graph_file_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        return

    def prepare_file_system(self):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)
        if self.intermediate_store_frequency > 0:
            self.ensure_dir_exists(self.intermediate_output_graphs_dir)
        return

    def create_model_info(self):
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

        self.model = {
            'data_url': data_url,
            'bottleneck_tensor_name': bottleneck_tensor_name,
            'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width,
            'input_height': input_height,
            'input_depth': input_depth,
            'resized_input_tensor_name': resized_input_tensor_name,
            'model_file_name': model_file_name,
            'input_mean': input_mean,
            'input_std': input_std,
        }

    def add_jpeg_decoding(self):
        jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=self.model['input_depth'])
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([self.model['input_height'], self.model['input_width']])  # [[239],[239]]
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)  # 变成张量
        resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)  # 以双线性插值的形式将decoded_image_4d变成resize_shape_as_int
        offset_image = tf.subtract(resized_image, self.model['input_mean'])
        mul_image = tf.multiply(offset_image, 1.0 / self.model['input_std'])  # ？（x-均值)/标准差  像素归一化
        return jpeg_data, mul_image

    def run(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        # Prepare necessary directories that can be used during training
        self.prepare_file_system()

        # Gather information about the model we'll be using.
        self.create_model_info()

        # Set up the pre-trained grapha.
        self.maybe_download_and_extract()  # 下载inception-2015-12-05.tgz 并解压
        graph, bottleneck_tensor, resized_image_tensor = self.create_model_graph()  #把瓶颈层张量以及inception的图给获取了

        # Look at the folder structure, and create lists of all the images.
        self.create_image_lists()   # 把图片分成训练集测试集和验证集
        with open(self.ALL_LABELS_FILE) as f:
            labels = f.read().splitlines()
        class_count = len(labels)   # 我的总标签数

        do_distort_images = self.should_distort_images()  # 是否要进行数据增强操作

        with tf.Session(graph=graph) as sess:
            # Set up the image decoding sub-graph.
            self.sess = sess
            jpeg_data_tensor, decoded_image_tensor = self.add_jpeg_decoding()

            if do_distort_images:
                # We will be applying distortions, so setup the operations we'll need.
                (distorted_jpeg_data_tensor,
                 distorted_image_tensor) = self.add_input_distortions()
            else:
                # We'll make sure we've calculated the 'bottleneck' image summaries and
                # cached them on disk.
                self.cache_bottlenecks(jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor)

            # Add the new layer that we'll be training.
            (train_step, cross_entropy, bottleneck_input, ground_truth_input,
             final_tensor) = self.add_final_training_ops(class_count, bottleneck_tensor)

            # Create the operations we need to evaluate the accuracy of our new layer.
            evaluation_step = self.add_evaluation_step(final_tensor, ground_truth_input)

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summaries_dir + '/train',
                                                 sess.graph)

            validation_writer = tf.summary.FileWriter(
                self.summaries_dir + '/validation')

            # Set up all our weights to their initial default values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Run the training for as many cycles as requested on the command line.
            for i in range(self.how_many_training_steps):
                # Get a batch of input bottleneck values, either calculated fresh every
                # time with distortions applied, or from the cache stored on disk.
                if do_distort_images:
                    (train_bottlenecks,
                     train_ground_truth) = self.get_random_distorted_bottlenecks(
                        self.train_batch_size, TRAINING, distorted_jpeg_data_tensor,
                        distorted_image_tensor, resized_image_tensor, bottleneck_tensor, labels)
                else:
                    (train_bottlenecks, train_ground_truth, _) = self.get_random_cached_bottlenecks(
                        self.train_batch_size, TRAINING, jpeg_data_tensor, decoded_image_tensor,
                        resized_image_tensor, bottleneck_tensor, labels)
                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                is_last_step = (i + 1 == self.how_many_training_steps)
                if (i % self.eval_step_interval) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                    (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                    (datetime.now(), i, cross_entropy_value))
                    validation_bottlenecks, validation_ground_truth, _ = (
                        self.get_random_cached_bottlenecks(
                            self.validation_batch_size, VALIDATION,
                            jpeg_data_tensor,
                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,labels))
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={bottleneck_input: validation_bottlenecks,
                                   ground_truth_input: validation_ground_truth})
                    validation_writer.add_summary(validation_summary, i)
                    tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                    (datetime.now(), i, validation_accuracy * 100,
                                     len(validation_bottlenecks)))

                # Store intermediget_random_cached_bottlenecksate results
                intermediate_frequency = self.intermediate_store_frequency

                if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                        and i > 0):
                    intermediate_file_name = (self.intermediate_output_graphs_dir +
                                              'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to : ' +
                                    intermediate_file_name)
                    self.save_graph_to_file(graph, intermediate_file_name)

            # We've completed all our training, so run a final test evaluation on
            # some new images we haven't used before.

            test_bottlenecks, test_ground_truth, _ = (
                self.get_random_cached_bottlenecks(
                    self.test_batch_size, TESTING,
                    jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, labels))
            test_accuracy = sess.run(
                evaluation_step,
                feed_dict={bottleneck_input: test_bottlenecks,
                           ground_truth_input: test_ground_truth})
            tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                            (test_accuracy * 100, len(test_bottlenecks)))

            self.save_graph_to_file(graph, self.output_graph)



if __name__ == '__main__':

    train = Retrain()
    train.run()
