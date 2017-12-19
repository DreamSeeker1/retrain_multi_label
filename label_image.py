import tensorflow as tf
import sys

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return rstrip() 去除最后的字符 默认为空格
label_lines = [line.rstrip() for line in tf.gfile.GFile("labels.txt").readlines() if line.rstrip()]

# Unpersists graph from file
with tf.gfile.FastGFile("tmp/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    # jpeg_data_tensor = (tf.import_graph_def(
    #     graph_def,
    #     return_elements=['DecodeJpeg/contents:0']
    # ))
print ('aaa')
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    sigmoid_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(sigmoid_tensor, {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    

    filename = "results.txt"
    with open(filename, 'a+') as f:
        f.write('\n**%s**\n' % (image_path))
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            f.write('%s (score = %.5f)\n' % (human_string, score))
    
    
