#!/usr/bin/env python

import argparse
import errno
import math
import os
import struct
import sys
import tarfile
import urllib

import numpy
import tensorflow

#
# Constants
#
URL_INCEPTION3 = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

#
# Arguments
#
arg_parser = argparse.ArgumentParser(description='Convert Inception v3 batch-normalized weights into weights and biases for MPSCNNConvolution.')

arg_parser.add_argument('--inception3-url', default=URL_INCEPTION3, help='URL to Inception v3 model [%(default)s]')
arg_parser.add_argument('--input-dir',      default='./input',      help='Directory to download model [%(default)s]')
arg_parser.add_argument('--output-dir',     default='./output',     help='Directory to generate weights and biases [%(default)s]')
arg_parser.add_argument('--dat-dir',        default='./dat',        help='Directory of MetalImageRecognition .dat files [%(default)s]')

#
# ===== UTILITY
# ----- OS
#
def dir_create(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if errno.EEXIST == e.errno:
            return # success
        raise

#
# ----- INPUT
#
def url_download_extract(download_dir, url):
    # check
    filename = url.split('/')[-1]
    filepath = os.path.join(download_dir, filename)
    if os.path.exists(filepath):
        return # no-op

    # download
    def _reporthook(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                            filename,
                            100.0 * float(count*block_size) / float(total_size),
                            ))
        sys.stdout.flush()
    urllib.urlretrieve(
        url,
        filepath,
        _reporthook,
        )
    print ''

    fileinfo = os.stat(filepath)
    print 'Downloaded %s %s bytes.' % (
            filename,
            fileinfo.st_size,
            )

    # extract
    tarfile.open(filepath, 'r:gz').extractall(download_dir)

#
# ----- OUTPUT
#
def graph_create(graphpath):
    with tensorflow.python.platform.gfile.FastGFile(graphpath, 'r') as graphfile:
        graphdef = tensorflow.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        tensorflow.import_graph_def(graphdef, name='')

def dat_readformat(data_len):
    return dat_writeformat(data_len/struct.calcsize('<f'))

def dat_writeformat(data_count):
    return '<' + str(data_count) + 'f'

def conv_write(output_dir, dat_dir, sess, name):
    # read
    beta    = sess.graph.get_tensor_by_name(name + '/batchnorm/beta:0').eval()
    gamma   = sess.graph.get_tensor_by_name(name + '/batchnorm/gamma:0').eval()
    mean    = sess.graph.get_tensor_by_name(name + '/batchnorm/moving_mean:0').eval()
    var     = sess.graph.get_tensor_by_name(name + '/batchnorm/moving_variance:0').eval()
    weights = sess.graph.get_tensor_by_name(name + '/conv2d_params:0').eval()

    # calculate
    weight_modifiers = gamma / numpy.sqrt(var+0.001) # BN transform scale

    weights = weights * weight_modifiers
    biases  = beta - (weight_modifiers * mean)

    # write
    name_output = name.replace('/', '_')

    # - weights
    weights_output = numpy.zeros(reduce(lambda x,y: x*y, weights.shape), numpy.float32)

    output_i = 0
    for l in xrange(weights.shape[3]):
        for i in xrange(weights.shape[0]):
            for j in xrange(weights.shape[1]):
                for k in xrange(weights.shape[2]):
                    weights_output[output_i] = weights[i][j][k][l]
                    output_i += 1

    weights_filename = 'weights_%s.dat' % (name_output,)
    weights_filepath = os.path.join(output_dir, weights_filename)
    with open(weights_filepath, 'wb') as f:
        f.write(struct.pack(dat_writeformat(len(weights_output)), *weights_output))

    # - biases
    biases_filename = 'bias_%s.dat' % (name_output,)
    biases_filepath = os.path.join(output_dir, biases_filename)
    with open(biases_filepath, 'wb') as f:
        f.write(struct.pack(dat_writeformat(len(biases)), *biases))

    # check
    weights_dat_filepath = os.path.join(dat_dir, weights_filename)
    biases_dat_filepath  = os.path.join(dat_dir, biases_filename)
    if not os.path.exists(weights_dat_filepath) or \
       not os.path.exists(biases_dat_filepath):
        print '%-40s' % (name_output,)
        return

    weights_maxdelta = '?'
    with open(weights_dat_filepath, 'rb') as f:
        weights_dat = numpy.fromstring(f.read(), dtype='<f4')
        weights_maxdelta = max(map(abs, weights_output - weights_dat))

    biases_maxdelta = '?'
    with open(biases_dat_filepath) as f:
        biases_dat = numpy.fromstring(f.read(), dtype='<f4')
        biases_maxdelta = max(map(abs, biases - biases_dat))

    print '%-40s [max delta: w=%-8f b=%-8f]' % (name_output, weights_maxdelta, biases_maxdelta,)

def softmax_write(output_dir, dat_dir, sess):
    name = 'softmax'

    # read
    weights = sess.graph.get_tensor_by_name('softmax/weights:0').eval()
    biases  = sess.graph.get_tensor_by_name('softmax/biases:0' ).eval()

    # write
    # - weights
    weights_output = numpy.zeros(reduce(lambda x,y: x*y, weights.shape), numpy.float32)

    output_i = 0
    for l in xrange(weights.shape[1]):
        for k in xrange(weights.shape[0]):
            weights_output[output_i] = weights[k][l]
            output_i += 1

    weights_filename = 'weights_%s.dat' % (name,)
    weights_filepath = os.path.join(output_dir, weights_filename)
    with open(weights_filepath, 'wb') as f:
        f.write(struct.pack(dat_writeformat(len(weights_output)), *weights_output))

    # - biases
    biases_filename = 'bias_%s.dat' % (name,)
    biases_filepath = os.path.join(output_dir, biases_filename)
    with open(biases_filepath, 'wb') as f:
        f.write(struct.pack(dat_writeformat(len(biases)), *biases))

    # check
    weights_dat_filepath = os.path.join(dat_dir, weights_filename)
    biases_dat_filepath  = os.path.join(dat_dir, biases_filename)
    if not os.path.exists(weights_dat_filepath) or \
       not os.path.exists(biases_dat_filepath):
        print '%-40s' % (name,)
        return

    weights_maxdelta = '?'
    with open(weights_dat_filepath, 'rb') as f:
        weights_dat = numpy.fromstring(f.read(), dtype='<f4')
        weights_maxdelta = max(map(abs, weights_output - weights_dat))

    biases_maxdelta = '?'
    with open(biases_dat_filepath) as f:
        biases_dat = numpy.fromstring(f.read(), dtype='<f4')
        biases_maxdelta = max(map(abs, biases - biases_dat))

    print '%-40s [max delta: w=%-8f b=%-8f]' % (name, weights_maxdelta, biases_maxdelta,)

#
# ===== MAIN
#
def main():
    # ===== ARGUMENTS
    args =  arg_parser.parse_args()

    inception3_url = args.inception3_url
    input_dir      = args.input_dir
    output_dir     = args.output_dir
    dat_dir        = args.dat_dir

    # ===== INPUT
    dir_create(input_dir)
    url_download_extract(input_dir, inception3_url)

    # ===== OUTPUT
    dir_create(output_dir)

    # ----- LOAD
    graph_create(os.path.join(input_dir, 'classify_image_graph_def.pb'))

    with tensorflow.Session() as sess:
        # filters
        conv_write(output_dir, dat_dir, sess, 'conv')
        conv_write(output_dir, dat_dir, sess, 'conv_1')
        conv_write(output_dir, dat_dir, sess, 'conv_2')
        # pool
        conv_write(output_dir, dat_dir, sess, 'conv_3')
        conv_write(output_dir, dat_dir, sess, 'conv_4')
        # pool_1

        # inceptions with 1x1, 3x3, 5x5 convolutions
        conv_write(output_dir, dat_dir, sess, 'mixed/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed/tower_1/conv_2')
        # mixed/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower_1/conv_2')
        # mixed_1/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_1/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_2/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower_1/conv_2')
        # mixed_2/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_2/tower_2/conv')

        # inceptions with 1x1, 3x3(in sequence) convolutions
        conv_write(output_dir, dat_dir, sess, 'mixed_3/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_3/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_3/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_3/tower/conv_2')
        # mixed_3/pool

        # inceptions with 1x1, 7x1, 1x7 convolutions
        conv_write(output_dir, dat_dir, sess, 'mixed_4/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_1/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_1/conv_3')
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_1/conv_4')
        # mixed_4/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_4/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_5/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_1/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_1/conv_3')
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_1/conv_4')
        # mixed_5/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_5/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_6/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_1/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_1/conv_3')
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_1/conv_4')
        # mixed_6/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_6/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_7/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_1/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_1/conv_3')
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_1/conv_4')
        # mixed_7/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_7/tower_2/conv')

        # inceptions with 1x1, 3x3, 1x7, 7x1 filters
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower_1/conv_2')
        conv_write(output_dir, dat_dir, sess, 'mixed_8/tower_1/conv_3')
        # mixed_8/pool

        conv_write(output_dir, dat_dir, sess, 'mixed_9/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower/mixed/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower/mixed/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower_1/mixed/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower_1/mixed/conv_1')
        # mixed_9/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_9/tower_2/conv')

        conv_write(output_dir, dat_dir, sess, 'mixed_10/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower/mixed/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower/mixed/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower_1/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower_1/conv_1')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower_1/mixed/conv')
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower_1/mixed/conv_1')
        # mixed_10/tower_2/pool
        conv_write(output_dir, dat_dir, sess, 'mixed_10/tower_2/conv')

        # pool_3
        softmax_write(output_dir, dat_dir, sess)

if '__main__' == __name__:
    main()
