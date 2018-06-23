import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from dataset import Train_dataset
import math
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from utils import smooth_gan_labels, aggregate, subPixelConv3d
import nibabel as nib
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from keras.layers.convolutional import UpSampling3D
import argparse


def lrelu1(x):
    return tf.maximum(x, 0.25 * x)


def lrelu2(x):
    return tf.maximum(x, 0.3 * x)


def discriminator(input_disc, kernel, reuse, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    batch_size = 1
    div_patches = 4
    num_patches = 8
    img_width = 128
    img_height = 128
    img_depth = 92
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        input_disc.set_shape([int((batch_size * num_patches) / div_patches), img_width, img_height, img_depth, 1], )
        x = InputLayer(input_disc, name='in')
        x = Conv3dLayer(x, act=lrelu2, shape=[kernel, kernel, kernel, 1, 32], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1')
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 32, 32], strides=[1, 2, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv2')

        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv2', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 32, 64], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv3')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv3', act=lrelu2)
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 64], strides=[1, 2, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv4')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv4', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 128], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv5')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv5', act=lrelu2)
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 128, 128], strides=[1, 2, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv6')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv6', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 128, 256], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv7')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv7', act=lrelu2)
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 256, 256], strides=[1, 2, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv8')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv8', act=lrelu2)

        x = FlattenLayer(x, name='flatten')
        x = DenseLayer(x, n_units=1024, act=lrelu2, name='dense1')
        x = DenseLayer(x, n_units=1, name='dense2')

        logits = x.outputs
        x.outputs = tf.nn.sigmoid(x.outputs, name='output')

        return x, logits


def generator(input_gen, kernel, nb, upscaling_factor, reuse, feature_size, img_width, img_height, img_depth,
              subpixel_NN, nn, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)

    w_init_subpixel1 = np.random.normal(scale=0.02, size=[3, 3, 3, 64, feature_size])
    w_init_subpixel1 = zoom(w_init_subpixel1, [2, 2, 2, 1, 1], order=0)
    w_init_subpixel1_last = tf.constant_initializer(w_init_subpixel1)
    w_init_subpixel2 = np.random.normal(scale=0.02, size=[3, 3, 3, 64, 64])
    w_init_subpixel2 = zoom(w_init_subpixel2, [2, 2, 2, 1, 1], order=0)
    w_init_subpixel2_last = tf.constant_initializer(w_init_subpixel2)

    with tf.variable_scope("SRGAN_g", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x = InputLayer(input_gen, name='in')
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 1, feature_size], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1')
        x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='BN-conv1')
        inputRB = x
        inputadd = x

        # residual blocks
        for i in range(nb):
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, feature_size], strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-rb/%s' % i)
            x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='BN1-rb/%s' % i)
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, feature_size], strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv2-rb/%s' % i)
            x = BatchNormLayer(x, is_train=is_train, name='BN2-rb/%s' % i, )
            # short skip connection
            x = ElementwiseLayer([x, inputadd], tf.add, name='add-rb/%s' % i)
            inputadd = x

        # large skip connection
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, feature_size], strides=[1, 1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv2')
        x = BatchNormLayer(x, is_train=is_train, name='BN-conv2')
        x = ElementwiseLayer([x, inputRB], tf.add, name='add-conv2')

        # ____________SUBPIXEL-NN______________#

        if subpixel_NN:
            # upscaling block 1
            if upscaling_factor == 4:
                img_height_deconv = int(img_height / 2)
                img_width_deconv = int(img_width / 2)
                img_depth_deconv = int(img_depth / 2)
            else:
                img_height_deconv = img_height
                img_width_deconv = img_width
                img_depth_deconv = img_depth

            x = DeConv3dLayer(x, shape=[kernel * 2, kernel * 2, kernel * 2, 64, feature_size],
                              act=lrelu1, strides=[1, 2, 2, 2, 1],
                              output_shape=[tf.shape(input_gen)[0], img_height_deconv, img_width_deconv,
                                            img_depth_deconv, 64],
                              padding='SAME', W_init=w_init_subpixel1_last, name='conv1-ub-subpixelnn/1')

            # upscaling block 2
            if upscaling_factor == 4:
                x = DeConv3dLayer(x, shape=[kernel * 2, kernel * 2, kernel * 2, 64, 64],
                                  act=lrelu1, strides=[1, 2, 2, 2, 1], padding='SAME',
                                  output_shape=[tf.shape(input_gen)[0], img_height, img_width,
                                                img_depth, 64],
                                  W_init=w_init_subpixel2_last, name='conv1-ub-subpixelnn/2')

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 1], strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='convlast-subpixelnn')

        # ____________RC______________#

        elif nn:
            # upscaling block 1
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, 64], act=lrelu1,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-ub/1')
            x = UpSampling3D(name='UpSampling3D_1')(x.outputs)
            x = Conv3dLayer(InputLayer(x, name='in ub1 conv2'),
                            shape=[kernel, kernel, kernel, 64, 64],
                            act=lrelu1,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv2-ub/1')

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 64], act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv1-ub/2')
                x = UpSampling3D(name='UpSampling3D_1')(x.outputs)
                x = Conv3dLayer(InputLayer(x, name='in ub2 conv2'), shape=[kernel, kernel, kernel, 64,
                                                                           64], act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv2-ub/2')

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 64, 1], strides=[1, 1, 1, 1, 1],
                            act=tf.nn.tanh, padding='SAME', W_init=w_init, name='convlast')

        # ____________SUBPIXEL - BASELINE______________#

        else:

            if upscaling_factor == 4:
                steps_to_end = 2
            else:
                steps_to_end = 1

            # upscaling block 1
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, feature_size, 64], act=lrelu1,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-ub/1')
            arguments = {'img_width': img_width, 'img_height': img_height, 'img_depth': img_depth,
                         'stepsToEnd': steps_to_end,
                         'n_out_channel': int(64 / 8)}
            x = LambdaLayer(x, fn=subPixelConv3d, fn_args=arguments, name='SubPixel1')

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(x, shape=[kernel, kernel, kernel, int((64) / 8), 64], act=lrelu1,
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME', W_init=w_init, name='conv1-ub/2')
                arguments = {'img_width': img_width, 'img_height': img_height, 'img_depth': img_depth, 'stepsToEnd': 1,
                             'n_out_channel': int(64 / 8)}
                x = LambdaLayer(x, fn=subPixelConv3d, fn_args=arguments, name='SubPixel2')

            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, int(64 / 8), 1], strides=[1, 1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='convlast')

        return x


def train(upscaling_factor, residual_blocks, feature_size, path_prediction, checkpoint_dir, img_width, img_height,
          img_depth, subpixel_NN, nn, restore, batch_size=1, div_patches=4, epochs=10):
    traindataset = Train_dataset(batch_size)
    iterations_train = math.ceil((len(traindataset.subject_list) * 0.8) / batch_size)
    num_patches = traindataset.num_patches

    # ##========================== DEFINE MODEL ============================##
    t_input_gen = tf.placeholder('float32', [int((batch_size * num_patches) / div_patches), None,
                                             None, None, 1],
                                 name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [int((batch_size * num_patches) / div_patches),
                                                img_width, img_height, img_depth, 1],
                                    name='t_target_image')
    t_input_mask = tf.placeholder('float32', [int((batch_size * num_patches) / div_patches),
                                              img_width, img_height, img_depth, 1],
                                  name='t_image_input_mask')

    net_gen = generator(input_gen=t_input_gen, kernel=3, nb=residual_blocks, upscaling_factor=upscaling_factor,
                        img_height=img_height, img_width=img_width, img_depth=img_depth, subpixel_NN=subpixel_NN, nn=nn,
                        feature_size=feature_size, is_train=True, reuse=False)
    net_d, disc_out_real = discriminator(input_disc=t_target_image, kernel=3, is_train=True, reuse=False)
    _, disc_out_fake = discriminator(input_disc=net_gen.outputs, kernel=3, is_train=True, reuse=True)

    # test
    gen_test = generator(t_input_gen, kernel=3, nb=residual_blocks, upscaling_factor=upscaling_factor,
                         img_height=img_height, img_width=img_width, img_depth=img_depth, subpixel_NN=subpixel_NN,
                         nn=nn,
                         feature_size=feature_size, is_train=True, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###

    if np.random.uniform() > 0.1:
        # give correct classifications
        y_gan_real = tf.ones_like(disc_out_real)
        y_gan_fake = tf.zeros_like(disc_out_real)
    else:
        # give wrong classifications (noisy labels)
        y_gan_real = tf.zeros_like(disc_out_real)
        y_gan_fake = tf.ones_like(disc_out_real)

    d_loss_real = tf.reduce_mean(tf.square(disc_out_real - smooth_gan_labels(y_gan_real)),
                                 name='d_loss_real')
    d_loss_fake = tf.reduce_mean(tf.square(disc_out_fake - smooth_gan_labels(y_gan_fake)),
                                 name='d_loss_fake')
    d_loss = d_loss_real + d_loss_fake

    mse_loss = tf.reduce_sum(
        tf.square(net_gen.outputs - t_target_image), axis=[0, 1, 2, 3, 4], name='g_loss_mse')

    dx_real = t_target_image[:, 1:, :, :, :] - t_target_image[:, :-1, :, :, :]
    dy_real = t_target_image[:, :, 1:, :, :] - t_target_image[:, :, :-1, :, :]
    dz_real = t_target_image[:, :, :, 1:, :] - t_target_image[:, :, :, :-1, :]
    dx_fake = net_gen.outputs[:, 1:, :, :, :] - net_gen.outputs[:, :-1, :, :, :]
    dy_fake = net_gen.outputs[:, :, 1:, :, :] - net_gen.outputs[:, :, :-1, :, :]
    dz_fake = net_gen.outputs[:, :, :, 1:, :] - net_gen.outputs[:, :, :, :-1, :]

    gd_loss = tf.reduce_sum(tf.square(tf.abs(dx_real) - tf.abs(dx_fake))) + \
              tf.reduce_sum(tf.square(tf.abs(dy_real) - tf.abs(dy_fake))) + \
              tf.reduce_sum(tf.square(tf.abs(dz_real) - tf.abs(dz_fake)))

    g_gan_loss = 10e-2 * tf.reduce_mean(tf.square(disc_out_fake - smooth_gan_labels(tf.ones_like(disc_out_real))),
                                        name='g_loss_gan')

    g_loss = mse_loss + g_gan_loss + gd_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(1e-4, trainable=False)
    global_step = tf.Variable(0, trainable=False)
    decay_rate = 0.5
    decay_steps = 4920  # every 2 epochs (more or less)
    learning_rate = tf.train.inverse_time_decay(lr_v, global_step=global_step, decay_rate=decay_rate,
                                                decay_steps=decay_steps)

    # Optimizers
    g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

    session = tf.Session()
    tl.layers.initialize_global_variables(session)

    step = 0
    saver = tf.train.Saver()

    if restore is not None:
        saver.restore(session, tf.train.latest_checkpoint(restore))
        val_restore = 0 * epochs
    else:
        val_restore = 0

    array_psnr = []
    array_ssim = []

    for j in range(val_restore, epochs + val_restore):
        for i in range(0, iterations_train):
            # ====================== LOAD DATA =========================== #
            xt_total = traindataset.patches_true(i)
            xm_total = traindataset.mask(i)
            for k in range(0, div_patches):
                print('{}'.format(k))
                xt = xt_total[k * int((batch_size * num_patches) / div_patches):(int(
                    (batch_size * num_patches) / div_patches) * k) + int(
                    (batch_size * num_patches) / div_patches)]
                xm = xm_total[k * int((batch_size * num_patches) / div_patches):(int(
                    (batch_size * num_patches) / div_patches) * k) + int(
                    (batch_size * num_patches) / div_patches)]

                # NORMALIZING
                for t in range(0, xt.shape[0]):
                    normfactor = (np.amax(xt[t])) / 2
                    if normfactor != 0:
                        xt[t] = ((xt[t] - normfactor) / normfactor)

                x_generator = gaussian_filter(xt, sigma=1)
                x_generator = zoom(x_generator, [1, (1 / upscaling_factor), (1 / upscaling_factor),
                                                 (1 / upscaling_factor), 1], prefilter=False, order=0)
                xgenin = x_generator

                # ========================= train SRGAN ========================= #
                # update D
                errd, _ = session.run([d_loss, d_optim], {t_target_image: xt, t_input_gen: xgenin})
                # update G
                errg, errmse, errgan, errgd, _ = session.run([g_loss, mse_loss, g_gan_loss, gd_loss, g_optim],
                                                             {t_input_gen: xgenin, t_target_image: xt,
                                                              t_input_mask: xm})
                print(
                    "Epoch [%2d/%2d] [%4d/%4d] [%4d/%4d]: d_loss: %.8f g_loss: %.8f (mse: %.6f gdl: %.6f adv: %.6f)" % (
                        j, epochs + val_restore, i, iterations_train, k, div_patches - 1, errd, errg, errmse, errgd,
                        errgan))

                # ========================= evaluate & save model ========================= #

                if k == 1 and i % 20 == 0:
                    if j - val_restore == 0:
                        x_true_img = xt[0]
                        if normfactor != 0:
                            x_true_img = ((x_true_img + 1) * normfactor)  # denormalize
                        img_true = nib.Nifti1Image(x_true_img, np.eye(4))
                        img_true.to_filename(
                            os.path.join(path_prediction, str(j) + str(i) + 'true.nii.gz'))

                        x_gen_img = xgenin[0]
                        if normfactor != 0:
                            x_gen_img = ((x_gen_img + 1) * normfactor)  # denormalize
                        img_gen = nib.Nifti1Image(x_gen_img, np.eye(4))
                        img_gen.to_filename(
                            os.path.join(path_prediction, str(j) + str(i) + 'gen.nii.gz'))

                    x_pred = session.run(gen_test.outputs, {t_input_gen: xgenin})
                    x_pred_img = x_pred[0]
                    if normfactor != 0:
                        x_pred_img = ((x_pred_img + 1) * normfactor)  # denormalize
                    img_pred = nib.Nifti1Image(x_pred_img, np.eye(4))
                    img_pred.to_filename(
                        os.path.join(path_prediction, str(j) + str(i) + '.nii.gz'))

                    max_gen = np.amax(x_pred_img)
                    max_real = np.amax(x_true_img)
                    if max_gen > max_real:
                        val_max = max_gen
                    else:
                        val_max = max_real
                    min_gen = np.amin(x_pred_img)
                    min_real = np.amin(x_true_img)
                    if min_gen < min_real:
                        val_min = min_gen
                    else:
                        val_min = min_real
                    val_psnr = psnr(np.multiply(x_true_img, xm[0]), np.multiply(x_pred_img, xm[0]),
                                    dynamic_range=val_max - val_min)
                    val_ssim = ssim(np.multiply(x_true_img, xm[0]), np.multiply(x_pred_img, xm[0]),
                                    dynamic_range=val_max - val_min, multichannel=True)

        saver.save(sess=session, save_path=checkpoint_dir, global_step=step)
        print("Saved step: [%2d]" % step)
        step = step + 1


def evaluate(upsampling_factor, residual_blocks, feature_size, checkpoint_dir_restore, path_volumes, nn, subpixel_NN,
             img_height, img_width, img_depth):
    traindataset = Train_dataset(1)
    iterations = math.ceil(
        (len(traindataset.subject_list) * 0.2))
    print(len(traindataset.subject_list))
    print(iterations)
    totalpsnr = 0
    totalssim = 0
    array_psnr = np.empty(iterations)
    array_ssim = np.empty(iterations)
    batch_size = 1
    div_patches = 4
    num_patches = traindataset.num_patches

    # define model
    t_input_gen = tf.placeholder('float32', [1, None, None, None, 1],
                                 name='t_image_input_to_SRGAN_generator')
    srgan_network = generator(input_gen=t_input_gen, kernel=3, nb=residual_blocks,
                              upscaling_factor=upsampling_factor, feature_size=feature_size, subpixel_NN=subpixel_NN,
                              img_height=img_height, img_width=img_width, img_depth=img_depth, nn=nn,
                              is_train=False, reuse=False)

    # restore g
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="SRGAN_g"))
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_restore))

    for i in range(0, iterations):
        # extract volumes
        xt_total = traindataset.data_true(654 + i)
        xt_mask = traindataset.mask(654 + i)
        normfactor = (np.amax(xt_total[0])) / 2
        x_generator = ((xt_total[0] - normfactor) / normfactor)
        res = 1 / upsampling_factor
        x_generator = x_generator[:, :, :, np.newaxis]
        x_generator = gaussian_filter(x_generator, sigma=1)
        x_generator = zoom(x_generator, [res, res, res, 1], prefilter=False)
        xg_generated = sess.run(srgan_network.outputs, {t_input_gen: x_generator[np.newaxis, :]})
        xg_generated = ((xg_generated + 1) * normfactor)
        volume_real = xt_total[0]
        volume_real = volume_real[:, :, :, np.newaxis]
        volume_generated = xg_generated[0]
        volume_mask = aggregate(xt_mask)
        # compute metrics
        max_gen = np.amax(volume_generated)
        max_real = np.amax(volume_real)
        if max_gen > max_real:
            val_max = max_gen
        else:
            val_max = max_real
        min_gen = np.amin(volume_generated)
        min_real = np.amin(volume_real)
        if min_gen < min_real:
            val_min = min_gen
        else:
            val_min = min_real
        val_psnr = psnr(np.multiply(volume_real, volume_mask), np.multiply(volume_generated, volume_mask),
                        dynamic_range=val_max - val_min)
        array_psnr[i] = val_psnr

        totalpsnr += val_psnr
        val_ssim = ssim(np.multiply(volume_real, volume_mask), np.multiply(volume_generated, volume_mask),
                        dynamic_range=val_max - val_min, multichannel=True)
        array_ssim[i] = val_ssim
        totalssim += val_ssim
        print(val_psnr)
        print(val_ssim)
        # save volumes
        filename_gen = os.path.join(path_volumes, str(i) + 'gen.nii.gz')
        img_volume_gen = nib.Nifti1Image(volume_generated, np.eye(4))
        img_volume_gen.to_filename(filename_gen)
        filename_real = os.path.join(path_volumes, str(i) + 'real.nii.gz')
        img_volume_real = nib.Nifti1Image(volume_real, np.eye(4))
        img_volume_real.to_filename(filename_real)

    print('{}{}'.format('PSNR: ', array_psnr))
    print('{}{}'.format('SSIM: ', array_ssim))
    print('{}{}'.format('Mean PSNR: ', array_psnr.mean()))
    print('{}{}'.format('Mean SSIM: ', array_ssim.mean()))
    print('{}{}'.format('Variance PSNR: ', array_psnr.var()))
    print('{}{}'.format('Variance SSIM: ', array_ssim.var()))
    print('{}{}'.format('Max PSNR: ', array_psnr.max()))
    print('{}{}'.format('Min PSNR: ', array_psnr.min()))
    print('{}{}'.format('Max SSIM: ', array_ssim.max()))
    print('{}{}'.format('Min SSIM: ', array_ssim.min()))
    print('{}{}'.format('Median PSNR: ', np.median(array_psnr)))
    print('{}{}'.format('Median SSIM: ', np.median(array_ssim)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-path_prediction', help='Path to save training predictions')
    parser.add_argument('-path_volumes', help='Path to save test volumes')
    parser.add_argument('-checkpoint_dir', help='Path to save checkpoints')
    parser.add_argument('-checkpoint_dir_restore', help='Path to restore checkpoints')
    parser.add_argument('-residual_blocks', default=6, help='Number of residual blocks')
    parser.add_argument('-upsampling_factor', default=4, help='Upsampling factor')
    parser.add_argument('-evaluate', default=False, help='Number of residual blocks')
    parser.add_argument('-subpixel_NN', default=False, help='Use subpixel nearest neighbour')
    parser.add_argument('-nn', default=False, help='Use Upsampling3D + nearest neighbour, RC')
    parser.add_argument('-feature_size', default=32, help='Number of filters')
    parser.add_argument('-restore', default=None, help='Checkpoint path to restore training')
    args = parser.parse_args()

    if args.evaluate:
        evaluate(upsampling_factor=int(args.upsampling_factor), feature_size=int(args.feature_size),
                 residual_blocks=int(args.residual_blocks), checkpoint_dir_restore=args.checkpoint_dir_restore,
                 path_volumes=args.path_volumes, subpixel_NN=args.subpixel_NN, nn=args.nn, img_width=224,
                 img_height=224, img_depth=152)
    else:
        train(upscaling_factor=int(args.upsampling_factor), feature_size=int(args.feature_size),
              subpixel_NN=args.subpixel_NN, nn=args.nn, residual_blocks=int(args.residual_blocks),
              path_prediction=args.path_prediction, checkpoint_dir=args.checkpoint_dir, img_width=128,
              img_height=128, img_depth=92, batch_size=1, restore=args.restore)
