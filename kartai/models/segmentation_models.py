import tensorflow.keras.utils
from tensorflow.keras import layers
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


def CSPDense_block(input, features, block_name, activation):
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


def get_model(img_size, num_classes, activation, conv_block, features, depth):
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


def get_cross_model(img_size, num_classes, activation, conv_block, features, depth, use_CPP=False):
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


def twin_model_main(inputs, layer_type, activation, conv_block, features, depth):

    x1 = conv_bn_act(inputs[0], features, 'input_layer_orto' +
                    layer_type, 3, 1, activation)
    x2 = conv_bn_act(inputs[1], features, 'input_layer_height' +
                    layer_type, 3, 1, activation)

    # Downsample
    skip = []
    for d in range(depth):
        s1, x1 = strided_encoder_block(x1, conv_block, features * (2 ** d), f'encoder{d + 1}{layer_type}_orto', activation)
        s2, x2 = strided_encoder_block(x2, conv_block, features * (2 ** d), f'encoder{d + 1}{layer_type}_height', activation)
        skip.append(layers.concatenate([s1,s2]))

    # concatination
    x = layers.concatenate([x1, x2])

    # Final downsampled block
    x = SPP_bottom_block(x, features * (2 ** depth),activation, (1, 5, 9, 13, 17), layer_type)
    x = conv_block(x, features * (2 ** depth),f"bottom{layer_type}", activation)

    # Upsample
    for d in reversed(range(depth)):
        x = decoder_block_addskip(x, skip[d], conv_block, features * 2 * (2 ** d), f'decoder{d + 1}{layer_type}',
                                  activation)

    return x


def get_unet_twin_model(img_size, num_classes, activation, features, depth, height_img_size):
    conv_block = simple_conv_block

    inputs_img = layers.Input(img_size)
    inputs_height_img = layers.Input(height_img_size)

    x = twin_model_main([inputs_img, inputs_height_img],'_twin',activation, conv_block, features, depth)

    # This block might need some editing
    outputs = layers.Conv2D(
        num_classes, 1, padding='same', activation="sigmoid")(x)

    input_list = [inputs_img, inputs_height_img]
    model = keras.Model(inputs=input_list,
                        outputs=outputs, name="U-Net-twin")
    return model


def get_unet_model(img_size, num_classes, activation, features, depth):
    return get_model(img_size, num_classes, activation, simple_conv_block, features, depth)


def get_resnet_model(img_size, num_classes, activation, features, depth):
    return get_model(img_size, num_classes, activation, conv_res_block, features, depth)


def get_bottleneck_model(img_size, num_classes, activation, features, depth):
    return get_model(img_size, num_classes, activation, bottleneck_block, features, depth)


def get_csp_model(img_size, num_classes, activation, features, depth):
    return get_model(img_size, num_classes, activation, CSPDense_block, features, depth)


def get_bottleneck_cross_model(img_size, num_classes, activation, features, depth):
    return get_cross_model(img_size, num_classes, activation, bottleneck_block, features, depth)


def get_bottleneck_cross_SPP_model(img_size, num_classes, activation, features, depth):
    return get_cross_model(img_size, num_classes, activation, bottleneck_block, features, depth, True)


def gets_csp_cross_model(img_size, num_classes, activation, features, depth):
    return get_cross_model(img_size, num_classes, activation, CSPDense_block, features, depth)


def gets_csp_cross_SPP_model(img_size, num_classes, activation, features, depth):
    return get_cross_model(img_size, num_classes, activation, CSPDense_block, features, depth, True)


models = {
    "unet": get_unet_model,
    "resnet": get_resnet_model,
    "bottleneck": get_bottleneck_model,
    "bottleneck_cross": get_bottleneck_cross_model,
    "bottleneck_cross_SPP": get_bottleneck_cross_SPP_model,
    "CSP": get_csp_model,
    "CSP_cross": gets_csp_cross_model,
    "CSP_cross_SPP": gets_csp_cross_SPP_model,
    "unet-twin": get_unet_twin_model  # Used for combination of image and heigh data
}

models_supporting_stacked_input = {
    "unet",
    "resnet",
    "bottleneck",
    "bottleneck_cross",
    "bottleneck_cross_SPP",
    "CSP",
    "CSP_cross",
    "CSP_cross_SPP",
}
models_supporting_tupple_input = {
    "unet-twin",
}

# Mish Activation Function


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
