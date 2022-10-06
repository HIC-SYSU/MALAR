import tensorflow as tf
import math

is_training = tf.placeholder_with_default(True, shape=[], name='is_training')


def discriminator(image, options, reuse=False, name='discriminator'):
    def first_layer(input_, out_channels, ks=4, s=2, name='first_layer'):
        with tf.variable_scope(name):
            return lrelu(conv2d(input_, out_channels, ks=ks, s=s))

    def conv_layer(input_, out_channels, ks=4, s=2, name='conv_layer'):
        with tf.variable_scope(name):
            return lrelu(_norm(conv2d(input_, out_channels, ks=ks, s=s)))

    def last_layer(input_, out_channels, ks=4, s=1, name='last_layer'):
        with tf.variable_scope(name):
            return conv2d(input_, out_channels, ks=ks, s=s, use_bias=True)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        l1 = first_layer(image, options.df_dim, name='disc_layer1')
        l2 = conv_layer(l1, options.df_dim * 2, name='disc_layer2')
        l3 = conv_layer(l2, options.df_dim * 4, name='disc_layer3')
        l4 = conv_layer(l3, options.df_dim * 8, s=1, name='disc_layer4')
        l5 = last_layer(l4, 1, name='disc_layer5')
        return l5


def generator(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def res_block(input_, base_channels, modules_per_block, name='res_block'):
            with tf.variable_scope(name + '_module_1'):  # first module
                rm1 = conv_layer(input_, base_channels, ks=3, conv_type='res_block', name='_m1')
                rm2 = conv_layer(rm1, base_channels, ks=3, conv_type='res_block', with_relu=False, name='_m2')
                l = conv_layer(input_, base_channels, ks=1, with_relu=False, name='_concat')
                l = l + rm2
                l = tf.nn.relu(l)
            for module in range(1, modules_per_block):  # non-first module
                with tf.variable_scope(name + '_module_%d' % (module + 1)):
                    rm1 = conv_layer(l, base_channels, ks=3, conv_type='res_block', name='_m1')
                    rm2 = conv_layer(rm1, base_channels, ks=3, conv_type='res_block', with_relu=False, name='_m2')
                    l = l + rm2
                    l = tf.nn.relu(l)
            return l

        skip_connection = list()
        l = conv_layer(image, options.gf_dim, ks=7, s=2, conv_type='conv_7x7', name='convlayer1')
        skip_connection.append(l)
        l = tf.layers.average_pooling2d(l, 3, 2, padding='same')

        with tf.variable_scope('block_1'):
            l = res_block(l, options.gf_dim, 3)
            skip_connection.append(l)
            l = tf.layers.average_pooling2d(l, 3, 2, padding='same')

        with tf.variable_scope('block_2'):
            l = res_block(l, options.gf_dim * 2, 4)
            with tf.variable_scope('attention'):
                l = mhsa(l)
            skip_connection.append(l)
            l = tf.layers.average_pooling2d(l, 3, 2, padding='same')

        with tf.variable_scope('block_3'):
            l = res_block(l, options.gf_dim * 4, 6)
            with tf.variable_scope('attention'):
                l = mhsa(l)
            skip_connection.append(l)
            l = tf.layers.average_pooling2d(l, 3, 2, padding='same')

        with tf.variable_scope('block_4'):
            l = res_block(l, options.gf_dim * 8, 3)
            with tf.variable_scope('attention'):
                l = mhsa(l)

        skip_connection = skip_connection[::-1]

        with tf.variable_scope('block_3-up'):
            l = deconv_layer(l, options.gf_dim * 4, name='deconv_3')
            l = tf.concat([l, skip_connection[0]], -1, name='concat_3')
            l = conv_layer(l, options.gf_dim * 4, name='_l31')
        with tf.variable_scope('block_2-up'):
            l = deconv_layer(l, options.gf_dim * 2, name='deconv_2')
            l = tf.concat([l, skip_connection[1]], -1, name='concat_2')
            l = conv_layer(l, options.gf_dim * 2, name='_l21')
        with tf.variable_scope('block_1-up'):
            l = deconv_layer(l, options.gf_dim, name='deconv_1')
            l = tf.concat([l, skip_connection[2]], -1, name='concat_1')
            l = conv_layer(l, options.gf_dim, name='_l11')

        with tf.variable_scope('output'):
            l = deconv_layer(l, options.gf_dim, name='deconv_1')
            l = tf.concat([l, skip_connection[3]], -1, name='concat')
            l = conv_layer(l, options.gf_dim, name='convlayer2')
            l = deconv_layer(l, options.gf_dim, name='deconv_2')
            l = tf.nn.tanh(conv2d(l, 1, ks=7, s=1, conv_type='conv_7x7', name='output'))

        return l


def mhsa(A):  # multi-head self-attention
    N = A.get_shape().as_list()[0]
    H = A.get_shape().as_list()[1]
    W = A.get_shape().as_list()[2]
    c = A.get_shape().as_list()[3]
    c_ = c // 8   # for dimensionality reduction

    # 1.linear transformation
    B = tf.layers.dense(A, c_, use_bias=True)  # keys
    C = tf.layers.dense(A, c_, use_bias=True)  # query
    D = tf.layers.dense(A, c_, use_bias=True)  # values
    # split and concat: (N, H, W, c_) -> (h*N, H, W, c_/h), h=8
    B_ = tf.concat(tf.split(B, 8, axis=-1), axis=0)
    C_ = tf.concat(tf.split(C, 8, axis=-1), axis=0)
    D_ = tf.concat(tf.split(D, 8, axis=-1), axis=0)

    # 2.Scaled Dot Product Attention
    # a.calculate S
    B_reshape = tf.reshape(B_, [-1, H * W, int(c_/8)])
    C_reshape = tf.reshape(C_, [-1, H * W, int(c_/8)])
    B_reshape_transpose = tf.transpose(B_reshape, perm=[0, 2, 1])  # (h*N, c_/8, H*W)
    S = tf.nn.softmax(tf.matmul(C_reshape, B_reshape_transpose) / math.sqrt(int(c_/8)))  # (h*N, H*W, H*W)

    # b.calculate E
    D_reshape = tf.reshape(D_, [-1, H * W, int(c_/8)])  # (h*N, H*W, c_/8)
    S_transpose = tf.transpose(S, perm=[0, 2, 1])  # (h*N, H*W, H*W)
    D_S_mul = tf.matmul(S_transpose, D_reshape)  # (h*N, H*W, c_/8)
    D_S_mul_reshape = tf.reshape(D_S_mul, [-1, H, W, int(c_/8)])  # (h*N, H, W, c_/8)

    # 3.Restore shape
    mhsa = tf.concat(tf.split(D_S_mul_reshape, 8, axis=0), axis=-1)  # (N, H, W, c_)
    mhsa = tf.layers.dense(mhsa, c, use_bias=True)  # c_ -> c

    # 4.residual connection
    with tf.variable_scope('pam'):
        alpha = tf.get_variable("alpha", [1], trainable=True, initializer=tf.constant_initializer(0.0))
    E = alpha * mhsa + A

    # 5.normalization
    E = _layer_norm(E)

    return E


def print_shape(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def _norm(input_, norm='instance'):
    if norm == 'instance':
        return _instance_norm(input_)
    elif norm == 'batch':
        return _batch_norm(input_, is_training)
    else:
        return input_


def _batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(input_, decay=0.9, scale=True, updates_collections=None, is_training=is_training)


def _instance_norm(input_, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input_.get_shape().as_list()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        return scale * normalized + offset


def _layer_norm(input_, epsilon=1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = input_.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(input_, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (input_ - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def conv2d(batch_input, out_channels, ks=3, s=1, use_bias=False, conv_type='conv_3x3', name="conv2d"):
    with tf.variable_scope(name):
        if conv_type == 'res_block':
            pad_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            l = tf.layers.conv2d(pad_input, out_channels, kernel_size=ks, strides=s, padding="VALID", use_bias=use_bias, kernel_initializer=tf.random_normal_initializer(0, 0.02))
        elif conv_type == 'conv_7x7':
            pad_input = tf.pad(batch_input, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
            l = tf.layers.conv2d(pad_input, out_channels, kernel_size=ks, strides=s, padding="VALID", use_bias=use_bias, kernel_initializer=tf.random_normal_initializer(0, 0.02))
        else:  # conv_3x3, conv_4x4
            l = tf.layers.conv2d(batch_input, out_channels, kernel_size=ks, strides=s, padding="SAME", use_bias=use_bias, kernel_initializer=tf.random_normal_initializer(0, 0.02))
        print_shape(l)
        return l


def conv_layer(batch_input, out_channels, ks=3, s=1, conv_type='conv_3x3', norm_type='instance', with_relu=True, name='conv_layer'):
    with tf.variable_scope(name):
        l = conv2d(batch_input, out_channels, ks=ks, s=s, conv_type=conv_type)
        l = _norm(l, norm_type)
        if with_relu:
            l = tf.nn.relu(l)
        return l


def deconv2d(batch_input, out_channels, ks=3, s=2, use_bias=False, name="deconv2d"):
    with tf.variable_scope(name):
        l = tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=ks, strides=s, padding="SAME", use_bias=use_bias, kernel_initializer=tf.random_normal_initializer(0, 0.02))
        print_shape(l)
        return l


def deconv_layer(batch_input, out_channels, ks=3, s=2, norm_type='instance', name='deconv_layer'):
    with tf.variable_scope(name):
        l = deconv2d(batch_input, out_channels, ks=ks, s=s)
        l = _norm(l, norm_type)
        l = tf.nn.relu(l)
        return l


def vgg_net(y_pred):
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')  # weights='vgg_16.ckpt'
    vgg.trainable = False
    for layer in vgg.layers:
        layer.trainable = False

    model_relu5_3 = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output)
    model_relu5_3.trainable = False

    y_pred = (y_pred + 1) / 2 * 255.0
    y_pred = tf.concat([y_pred, y_pred, y_pred], -1)
    y_pred = tf.keras.applications.vgg16.preprocess_input(y_pred)

    pred_relu5_3 = model_relu5_3(y_pred)

    return pred_relu5_3


def intra_feature(feature1):
    # 1.global average pooling
    gvp1 = tf.reduce_mean(feature1, [1, 2])
    # 2.transpose
    rs1_transpose = tf.transpose(gvp1, perm=[1, 0])
    # 3.gram matrix
    gm1 = tf.matmul(gvp1, rs1_transpose)
    # 4.l2 norm
    gm1_norm = tf.norm(gm1, ord=2, axis=1)
    # 5.reshape gm1_norm same to gm1
    norm1 = tf.reshape(gm1_norm, [-1, 1])
    # 6.relation matrix
    rm1 = gm1 / norm1

    return rm1


def least_square(A, B):
    return tf.reduce_mean((A - B) ** 2)


def reconstruction_loss(A, F_GA, lambda_):
    return lambda_ * (tf.reduce_mean(tf.abs(A - F_GA)))
