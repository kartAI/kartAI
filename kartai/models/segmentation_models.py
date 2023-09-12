import numbers
import tensorflow.keras.utils
from tensorflow.keras import layers, Sequential
from tensorflow import keras
import tensorflow as tf


def findNextPowerOf2(n):
    k = 1
    while k < n:
        k = k << 1
    return k


def conv_bn_act(x, features, name, kernel=1, stride=1, activation=None):
    """Basic building block of Convolution - Batch normalization - Activation"""

    # 3x3 convolution layer without bias, as we have learned gamma - beta parameters in the BN
    x = layers.Conv2D(features, kernel, stride,
                      padding="same",
                      name=name + "_conv",
                      use_bias=False,
                      data_format="channels_last")(x)

    # Batch normalization, with learned bias
    x = layers.BatchNormalization(name=name + "_batchnorm")(x)

    # Activation
    if activation:
        x = layers.Activation(activation, name=name + "_activation")(x)

    return x


def simple_conv_block(x, features, block_name, activation):
    """Create conv2D - BN - act - conv2D - BN - act block.

    A simple convolution block with two convolution layers and batch normalization.
    """

    # Two 3x3 convolutions
    x = conv_bn_act(x, features, f"{block_name}_c1", 3, 1, activation)
    x = conv_bn_act(x, features, f"{block_name}_c2", 3, 1, activation)

    return x


def conv_res_block(input, features, block_name, activation):
    """Create conv2D - BN - act - conv2D - BN - add(input) - act block.

    This is a residual block as used in ResNet, that may be more efficient in deep nets.
    It has two convolution layers with batch normalization
    """

    # Two 3x3 convolutions
    x = conv_bn_act(input, features, f"{block_name}_c1", 3, 1, activation)
    x = conv_bn_act(x, features, f"{block_name}_c2", 3, 1)

    # Add in residual connection from input
    x = layers.Add(name=f"{block_name}_add")([x, input])

    # Put activation last
    x = layers.Activation(activation, name=f"{block_name}_activation")(x)

    return x


def bottleneck_block(input, features, block_name, activation):
    """Create bottleneck block.

    This is a residual block with bottleneck. The number of features in the convolution layers
    are squeezed down to a fraction of the input for forcing the CNN to generalize,
    and expanded to the original number before adding the input. Similar block are used in YOLOv3 with good results.
    """
    squeeze = findNextPowerOf2(features // 4)
    block_depth = 3

    # 1x1 convolution layer for squeezing the number of parameters
    x = conv_bn_act(input, squeeze, f"{block_name}_squeeze", 1, 1, activation)

    for i in range(block_depth):
        # Bottlenecked 3x3 convolution
        x = conv_bn_act(
            x, squeeze, f"{block_name}_bottleneck{i + 1}", 3, 1, activation)

    # 1x1 convolution layer for expanding the number of parameters
    x = conv_bn_act(x, features, f"{block_name}_expand", 1, 1)

    # Add in residual connection from input
    x = layers.Add(name=f"{block_name}_add")([x, input])

    # Activation last
    x = layers.Activation(activation, name=f"{block_name}_activation")(x)

    return x


def CSPDense_block(input, features,  depth=2, shortcut=True, expansion=0.5, block_name="csp_block", activation="swish"):
    """Create Cross Stage Partial DenseNet (CSP) block

    The CSPDense block splits input into two data paths, both with a reduced number of channels.
    One path goes trough a dense block, the other bypassing and joining at the end.
    This approach is less resource demanding than a full dense block, but should give almost as good results.
    """
    features = features if features > 0 else input.shape[-1]
    hidden_features = int(features * expansion)

    # Projecting into the dense layers
    x = conv_bn_act(input, features,
                    f"{block_name}_in", 1, 1, activation)

    # Split into shortcut and internal processing
    short, x = tf.split(
        x, [features-hidden_features, hidden_features], axis=-1)

    out = [short, x]
    for i in range(depth):
        # Dense block part
        x = conv_bn_act(
            x, hidden_features, f"{block_name}_part{i + 1}_1", 3, 1, activation)
        x = conv_bn_act(
            x, hidden_features, f"{block_name}_part{i + 1}_2", 3, 1, activation)

        # Shortcut?
        if shortcut:
            x = x + out[-1]

        out.append(x)

    # Concatenating dense block part with other inputs
    out = tf.concat(out, axis=-1)

    # Reproject into number of channesl
    out = conv_bn_act(out, features,
                      f"{block_name}_out", 1, 1, activation)

    return out


def CSPDense_block_dep(input, features, block_name, activation):
    """Create Cross Stage Partial DenseNet (CSP) block

    The CSPDense block splits input into two data paths, both with a reduced number of channels.
    One path goes trough a dense block, the other bypassing and joining at the end.
    This approach is less resource demanding than a full dense block, but should give almost as good results.
    """
    part_fts = features // 2
    bypass_fts = features - part_fts
    block_depth = 3

    # Projecting into the bypass of the  the dense layers
    bypass = layers.Conv2D(bypass_fts, (1, 1), padding="same",
                           name=f"{block_name}_trans_bypass_conv",
                           use_bias=False,
                           data_format="channels_last")(input)

    # Projecting into the dense layers
    x = conv_bn_act(input, part_fts,
                    f"{block_name}_trans_part_in", 1, 1, activation)

    for i in range(block_depth):
        # Dense block part
        block_x = conv_bn_act(
            x, part_fts, f"{block_name}_part{i + 1}", 3, 1, activation)

        # Concatenating dense block part with other inputs
        x = layers.Concatenate(
            name=block_name + f"_part{i+1}_concat")([x, block_x])

    # Squeezing dense output
    x = layers.Conv2D(part_fts, (1, 1), padding="same",
                      name=block_name + "_conv_trans_part_out",
                      use_bias=False,
                      data_format="channels_last")(x)

    # concatenating dense output with bypass
    x = layers.Concatenate(
        name=block_name + "_part_bypass_concat")([bypass, x])

    # Normalize and activation
    x = layers.BatchNormalization(name=block_name + "_norm_out")(x)
    x = layers.Activation(activation, name=block_name + "_activation_out")(x)

    return x


def SPP_bottom_block(x, features, activation, kernels=(1, 5, 9, 13), layer_subname=''):
    """Spatial Pyramid Pooling block for wide scope analysis at the lowest resolution level.
    "Stolen" from YOLOv4
    """
    cnc = []
    for k in kernels:
        xk = conv_bn_act(x, findNextPowerOf2(features // k),
                         f"spp_k{k}{layer_subname}", k, 1, activation)
        cnc.append(xk)

    x = layers.Concatenate()(cnc)
    x = conv_bn_act(x, features, f"spp_out{layer_subname}", 1, 1, activation)

    return x


def pool_encoder_block(x, conv_block, features, block_name, activation):
    """Encoder blocks using max pooling"""
    x = conv_block(x, features, block_name, activation)
    p = layers.MaxPool2D((2, 2), name=f"{block_name}_down_pool")(x)
    return x, p


def strided_encoder_block(x, conv_block, features, block_name, activation):
    """Encoder blocks using strided convolution"""
    x = conv_block(x, features, block_name, activation)
    d = conv_bn_act(x, features * 2, f"{block_name}_down", 3, 2, activation)
    return x, d


def decoder_block_addskip(input, skip_features, conv_block, features, block_name, activation, dropout=0.0):
    # Upscale convolution
    x = layers.Conv2DTranspose(
        features, (3, 3), strides=2, name=block_name + "_conv_up", padding="same")(input)

    # Add in skip connection
    x = layers.Add(name=block_name+"_add_skip")([x, skip_features])

    # Dropout
    if dropout > 0:
        x = layers.Dropout(dropout, name=block_name+"_drop")(x)

    # Convolution block
    x = conv_block(x, features, block_name, activation)

    return x


def decoder_block_catskip(input, skip_features, conv_block, features, block_name, activation, dropout=0.0):
    # Upscale convolution
    x = layers.Conv2DTranspose(
        features, (3, 3), strides=2, name=block_name + "_conv_up", padding="same")(input)

    # Concatenate skip connection
    x = layers.Concatenate(name=block_name+"_cat_skip")([x, skip_features])

    # Reduce number of filters after concatenation
    x = conv_bn_act(x, features, block_name +
                    "_postconv", activation=activation)

    # Dropout
    if dropout > 0:
        x = layers.Dropout(dropout, name=block_name+"_drop")(x)

    # Convolution block
    x = conv_block(x, features, block_name, activation)

    return x


def get_model(img_size, num_classes, activation, conv_block, features, depth) -> Sequential:
    inputs = layers.Input(img_size)

    # An initial layer to get the number of channels right
    x = conv_bn_act(inputs, features, "initial_layer", 3, 1, activation)

    # Downsample
    skip = []
    for d in range(depth):
        s, x = strided_encoder_block(x, conv_block, features * (2**d), f'encoder{d+1}',
                                     activation)
        skip.append(s)

    # Final downsampled block
    x = SPP_bottom_block(x, features * (2**depth),
                         activation, (1, 5, 9, 13, 17))
    x = conv_block(x, features * (2**depth), "bottom", activation)

    # Upsample
    for d in reversed(range(depth)):
        x = decoder_block_addskip(x, skip[d], conv_block, features * (2**d), f'decoder{d+1}',
                                  activation)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 1, padding="same", activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="U-Net")
    return model


def get_cross_model(img_size, num_classes, activation, conv_block, features, depth, use_CPP=False) -> Sequential:
    """Cross Connected Convolutional Partitioning Segmentation Neural Network (CCCP - Net)"""
    inputs = layers.Input(img_size)

    # An initial layer to get the number of channels right
    inp = [conv_bn_act(inputs, features, "initial_layer", 3, 1, activation)]

    # Left half of model
    for d in range(1, depth+1):
        out = []
        for sub_d in range(d):
            inp_arr = []

            block_name = f"block_{d}_{sub_d+1}"
            # Downsample from layer at level above (unless we are at first level)
            if sub_d > 0:
                inp_arr.append(layers.Conv2D(
                    features * (2**sub_d), (3, 3), strides=2, name=f"{block_name}_down", padding="same")(inp[sub_d-1]))

            # Just take layer at same level
            if sub_d < len(inp):
                inp_arr.append(inp[sub_d])

            # Upsample from layer at level below
            if sub_d < len(inp) - 1:
                inp_arr.append(layers.Conv2DTranspose(
                    features * (2**sub_d), (3, 3), strides=2, name=f"{block_name}_up", padding="same")(inp[sub_d+1]))

            # Add inputs.
            # An alternative here could have been concatenating the up/level/down - layers
            # and squeezing the result into the correct number of features
            x = inp_arr[0]
            if len(inp_arr) > 1:
                x = layers.Add(name=f"{block_name}_inp_add")(inp_arr)
            out.append(conv_block(x, features *
                       (2**sub_d), block_name, activation))

        inp = out

    # Final downsampled block
    if use_CPP:
        inp[-1] = SPP_bottom_block(inp[-1], features *
                                   (2 ** depth), activation, (1, 5, 9, 13, 17))
        inp[-1] = conv_block(inp[-1], features *
                             (2 ** depth), "bottom", activation)

    # Right half of model, almost as above
    for d in reversed(range(1, depth)):
        out = []
        for sub_d in range(d):
            inp_arr = []

            block_name = f"block_{2 * depth - d}_{sub_d + 1}"
            # Downsample from layer at level above
            if sub_d > 0:
                inp_arr.append(layers.Conv2D(
                    features * (2 ** sub_d), (3, 3), strides=2, name=f"{block_name}_down", padding="same")(
                    inp[sub_d - 1]))

            # Just take layer at same level
            if sub_d < len(inp):
                inp_arr.append(inp[sub_d])

            # Upsample from layer at level below
            if sub_d < len(inp) - 1:
                inp_arr.append(layers.Conv2DTranspose(
                    features * (2 ** sub_d), (3, 3), strides=2, name=f"{block_name}_up", padding="same")(
                    inp[sub_d + 1]))

            out.append(conv_block(layers.Add(name=f"{block_name}_inp_add")(inp_arr), features * (2 ** sub_d), block_name,
                                  activation))

        inp = out

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, padding="same",
                            activation="sigmoid")(inp[0])

    # ...and create model
    model = keras.Model(inputs, outputs, name="U-Net")
    return model


def twin_model_main(inputs, layer_type, activation, conv_block, features, depth) -> Sequential:

    x1 = conv_bn_act(inputs[0], features, 'input_layer_orto' +
                     layer_type, 3, 1, activation)
    x2 = conv_bn_act(inputs[1], features, 'input_layer_height' +
                     layer_type, 3, 1, activation)

    # Downsample
    skip = []
    for d in range(depth):
        s1, x1 = strided_encoder_block(
            x1, conv_block, features * (2 ** d), f'encoder{d + 1}{layer_type}_orto', activation)
        s2, x2 = strided_encoder_block(
            x2, conv_block, features * (2 ** d), f'encoder{d + 1}{layer_type}_height', activation)
        skip.append(layers.concatenate([s1, s2]))

    # concatination
    x = layers.concatenate([x1, x2])

    # Final downsampled block
    x = SPP_bottom_block(x, features * (2 ** depth),
                         activation, (1, 5, 9, 13, 17), layer_type)
    x = conv_block(x, features * (2 ** depth),
                   f"bottom{layer_type}", activation)

    # Upsample
    for d in reversed(range(depth)):
        x = decoder_block_addskip(x, skip[d], conv_block, features * 2 * (2 ** d), f'decoder{d + 1}{layer_type}',
                                  activation)

    return x


def get_unet_twin_model(img_size, num_classes, activation, features, depth, height_img_size) -> Sequential:
    conv_block = simple_conv_block

    inputs_img = layers.Input(img_size)
    inputs_height_img = layers.Input(height_img_size)

    x = twin_model_main([inputs_img, inputs_height_img],
                        '_twin', activation, conv_block, features, depth)

    # This block might need some editing
    outputs = layers.Conv2D(
        num_classes, 1, padding='same', activation="sigmoid")(x)

    input_list = [inputs_img, inputs_height_img]
    model = keras.Model(inputs=input_list,
                        outputs=outputs, name="U-Net-twin")
    return model


def get_unet_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_model(img_size, num_classes, activation, simple_conv_block, features, depth)


def get_resnet_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_model(img_size, num_classes, activation, conv_res_block, features, depth)


def get_bottleneck_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_model(img_size, num_classes, activation, bottleneck_block, features, depth)


def get_csp_model(img_size, num_classes, activation, features, depth, expansion=0.3):
    """
    Create a CSP model, inspired by the yolo v8 backbone

    :param img_size: shape of input tensor
    :param num_classes: Number of output classes
    :param activation: Activation function: swish (aka SiLU) or mish recommended
    :param features: The number of hidden features.
        This can be a list (or tuple) of the same length as the depth list, or an integer.
        If 'features' is a list then it should have the same length as 'depth',
        and each element of the list is the number of hidden features in the corresponding csp block.
        A value of: [32, 64, 128] gives 32 hidden features in the full resolution csp block, 64 in the
        half-resolution block, and 128 in the quarter-resolution block.
        If 'features' is an integer it gives the number of hidden features in the full resolution block.
        The number of hidden features are then doubled for each downscaling. So, for a 'depth' of 4 and
        'features' of 16 this list is generated: [16, 32, 64, 128]
    :param depth: The depth of the CNN. This can be a list (or tuple) of positive integers or a positive integer.
        If 'depth' is a list then the length of the list describes the number of CSP blocks in the U-net structure,
        while each element describes the number of layers in each CSP block.
        A value of [3,6,9] gives 3 CSP layers in the full resolution block, 6 CSP layers in the half-resolution block
        and 9 in the quarter-resolution block.
        If 'depth' is an integer it describes the number of CSP blocks, and the number of layers in each CSP block
        follows this pattern: [3, 6, 9, 9, 9, ....]
    :param expansion: The fraction of the hidden parameters that are run through the inner convolution blocks
    :return: a CSP CNN model
    """

    if isinstance(depth, (list, tuple)):
        # If depth is a list or tuple, use as it is
        depth_list = depth
        # Let "depth" be length of depth list
        depth = len(depth_list)
    elif isinstance(depth, numbers.Integral):
        # If depth is an integer, generate depth_list of the given length
        depth_list = [min(9, 3 * (d + 1)) for d in range(depth)]
    else:
        raise ValueError(
            f"Argument 'depth' should be an integer or a list/tuple of integers but is: {depth}")

    if isinstance(features, (list, tuple)):
        # If features is a list or tuple, use as it is, but check that length is consistent with depth_list
        features_list = features
        if len(features_list) != depth:
            raise ValueError(
                f"depth and features doesn't have same length (depth: {depth}, features: {len(features_list)})")
    elif isinstance(features, numbers.Integral):
        # If features is an integer, generate features_list with doubling for each downscaling
        features_list = [features * 2 ** d for d in range(depth)]
    else:
        raise ValueError(
            f"Argument 'features' should be an integer or a list/tuple of integers but is: {features}")

    model_name = "csp_model"
    inputs = layers.Input(img_size)

    # An initial layer to get the number of channels right
    x = conv_bn_act(
        inputs, features_list[0], f"{model_name}_initial_layer", 3, 1, activation)

    # Downsample
    skip = []
    for layer_ix, (d, f) in enumerate(zip(depth_list, features_list)):
        if layer_ix >= 1:
            x = conv_bn_act(
                x, f, f'{model_name}_encoder_down_{layer_ix}', 3, 2, activation)
        x = CSPDense_block(x, f, d, True, expansion,
                           f'{model_name}_encoder_{layer_ix+1}', activation)
        skip.append(x)
        if layer_ix == depth - 1:
            # Final downsampled block
            x = SPP_bottom_block_fast(
                x, activation=activation, name=f"{model_name}_spp_fast")

    # Upsample
    for layer_ix, (d, f, s) in reversed(list(enumerate(zip(depth_list, features_list, skip)))):
        if layer_ix < depth - 1:
            x = layers.Conv2DTranspose(
                f, (3, 3), strides=2,
                name=f"{model_name}_decoder_conv_up_{layer_ix + 1}",
                use_bias=False, padding="same")(x)
            # Batch normalization, with learned bias
            x = layers.BatchNormalization(
                name=f"{model_name}_decoder_conv_up_{layer_ix + 1}_batchnorm")(x)

            # Activation
            if activation:
                x = layers.Activation(
                    activation, name=f"{model_name}_decoder_conv_up_{layer_ix + 1}_activation")(x)

        x = tf.concat([x, s], axis=-1)
        x = CSPDense_block_dep(x, f, d, False, expansion,
                               f'{model_name}_decoder_{layer_ix + 1}', activation)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 1, padding="same", activation="sigmoid" if num_classes == 1 else 'softmax')(x)

    model = keras.Model(inputs, outputs, name=model_name)
    return model


def SPP_bottom_block_fast(x, pool_size=5, name="spp_fast", activation="swish"):

    features = x.shape[-1]
    hidden_features = int(features // 2)

    x = conv_bn_act(x, hidden_features,
                    f"{name}_inp", 1, 1, activation=activation)
    pool_1 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(x)
    pool_2 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)

    out = tf.concat([x, pool_1, pool_2, pool_3], axis=-1)
    out = conv_bn_act(
        out, features, f"{name}_out", 1, 1, activation=activation)
    return out


def get_csp_model_deprecated(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_model(img_size, num_classes, activation, CSPDense_block_dep, features, depth)


def get_bottleneck_cross_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_cross_model(img_size, num_classes, activation, bottleneck_block, features, depth)


def get_bottleneck_cross_SPP_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_cross_model(img_size, num_classes, activation, bottleneck_block, features, depth, True)


def gets_csp_cross_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_cross_model(img_size, num_classes, activation, CSPDense_block_dep, features, depth)


def gets_csp_cross_SPP_model(img_size, num_classes, activation, features, depth) -> Sequential:
    return get_cross_model(img_size, num_classes, activation, CSPDense_block_dep, features, depth, True)


def mish(x):
    return tf.keras.layers.Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)))(x)


tensorflow.keras.utils.get_custom_objects().update(
    {'mish': keras.layers.Activation(mish)})

activations = {
    "relu": keras.activations.relu,
    "leaky_relu": tf.nn.leaky_relu,
    "elu": keras.activations.elu,
    "gelu": keras.activations.gelu,
    "selu": keras.activations.selu,
    "swish": keras.activations.swish,
    "mish": mish
}

loss_functions = [
    "binary_crossentropy",
    "focal_loss",
]

optimizers = {
    "SGD": keras.optimizers.SGD,
    "RMSprop": keras.optimizers.RMSprop,
    "nadam": keras.optimizers.Nadam,
}


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
