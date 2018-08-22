"""
Train script that does full training of the model. It saves the model every epoch.

Before training make sure of the following:

1) The global constants are set i.e. NUM_TRAIN_IMGS, NUM_VAL_IMGS, NUM_TEST_IMGS.
2) The images for training, validation and testing should have proper heirarchy
   and proper file names. Details about the heirarchy and file name convention are
   provided in the README.

Command: python train_model.py --log_dir <log_dir_path> --num_epochs <num_of_epochs> --learning_rate <learning_rate> --session_id <session_id> --data_root <path_of_data>
@author: Aditya Vora
Created on Tuesday Dec 5th, 2017 3:15 PM.
"""

import tensorflow as tf
import src.mccnn as mccnn
import src.layers as L
import os
import src.utils as utils
import numpy as np
import matplotlib.image as mpimg
import scipy.io as sio
import time
import argparse
import sys


# Global Constants. Define the number of images for training, validation and testing.
NUM_TRAIN_IMGS = 6000
NUM_VAL_IMGS = 590
NUM_TEST_IMGS = 587

def main(args):
    """
    Main function to execute the training.
    Performs training, validation after each epoch and testing after full epoch training.
    :param args: input command line arguments which will set the learning rate, number of epochs, data root etc.
    :return: None
    """

    sess_path = utils.create_session(args.log_dir, args.session_id)  # Create a session path based on the session id.
    G = tf.Graph()
    with G.as_default():
        # Create image and density map placeholder
        image_place_holder = tf.placeholder(tf.float32, shape=[1, None, None, 1])
        d_map_place_holder = tf.placeholder(tf.float32, shape=[1, None, None, 1])

        # Build all nodes of the network
        d_map_est = mccnn.build(image_place_holder)

        # Define the loss function.
        euc_loss = L.loss(d_map_est, d_map_place_holder)

        # Define the optimization algorithm
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)

        # Training node.
        train_op = optimizer.minimize(euc_loss)

        # Initialize all the variables.
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # For summary
        summary = tf.summary.merge_all()

        with tf.Session(graph=G) as sess:
            writer = tf.summary.FileWriter(os.path.join(sess_path,'training_logging'))
            writer.add_graph(sess.graph)
            sess.run(init)

            #if args.retrain:
            #    utils.load_weights(G, args.base_model_path)


            # Start the epochs
            for eph in range(args.num_epochs):

                start_train_time = time.time()

                # Get the list of train images.
                train_images_list, train_gts_list = utils.get_data_list(args.data_root, mode='train')
                total_train_loss = 0

                # Loop through all the training images
                for img_idx in range(len(train_images_list)):

                    # Load the image and ground truth
                    train_image = np.asarray(mpimg.imread(train_images_list[img_idx]), dtype=np.float32)
                    train_d_map = np.asarray(sio.loadmat(train_gts_list[img_idx])['d_map'], dtype=np.float32)

                    # Reshape the tensor before feeding it to the network
                    train_image_r = utils.reshape_tensor(train_image)
                    train_d_map_r = utils.reshape_tensor(train_d_map)

                    # Prepare feed_dict
                    feed_dict_data = {
                        image_place_holder: train_image_r,
                        d_map_place_holder: train_d_map_r,
                    }

                    # Compute the loss for one image.
                    _, loss_per_image = sess.run([train_op, euc_loss], feed_dict=feed_dict_data)

                    # Accumalate the loss over all the training images.
                    total_train_loss = total_train_loss + loss_per_image

                end_train_time = time.time()
                train_duration = end_train_time - start_train_time

                # Compute the average training loss
                avg_train_loss = total_train_loss / len(train_images_list)

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(eph + 1, args.num_epochs, train_duration))
                print("  Training loss:\t\t{:.6f}".format(avg_train_loss))


                print ('Validating the model...')

                total_val_loss = 0

                # Get the list of images and the ground truth
                val_image_list, val_gt_list = utils.get_data_list(args.data_root, mode='valid')

                valid_start_time = time.time()

                # Loop through all the images.
                for img_idx in xrange(len(val_image_list)):

                    # Read the image and the ground truth
                    val_image = np.asarray(mpimg.imread(val_image_list[img_idx]), dtype=np.float32)
                    val_d_map = np.asarray(sio.loadmat(val_gt_list[img_idx])['d_map'], dtype=np.float32)

                    # Reshape the tensor for feeding it to the network
                    val_image_r = utils.reshape_tensor(val_image)
                    val_d_map_r = utils.reshape_tensor(val_d_map)

                    # Prepare the feed_dict
                    feed_dict_data = {
                        image_place_holder: val_image_r,
                        d_map_place_holder: val_d_map_r,
                    }

                    # Compute the loss per image
                    loss_per_image = sess.run(euc_loss, feed_dict=feed_dict_data)

                    # Accumalate the validation loss across all the images.
                    total_val_loss = total_val_loss + loss_per_image

                valid_end_time = time.time()
                val_duration = valid_end_time - valid_start_time

                # Compute the average validation loss.
                avg_val_loss = total_val_loss / len(val_image_list)

                print("  Validation loss:\t\t{:.6f}".format(avg_val_loss))
                print ("Validation over {} images took {:.3f}s".format(len(val_image_list), val_duration))

                # Save the weights as well as the summary
                utils.save_weights(G, os.path.join(sess_path, "weights.%s" % (eph+1)))
                summary_str = sess.run(summary, feed_dict=feed_dict_data)
                writer.add_summary(summary_str, eph)


            print ('Testing the model with test data.....')

            # Get the image list
            test_image_list, test_gt_list = utils.get_data_list(args.data_root, mode='test')
            abs_err = 0

            # Loop through all the images.
            for img_idx in xrange(len(test_image_list)):

                # Read the images and the ground truth
                test_image = np.asarray(mpimg.imread(test_image_list[img_idx]), dtype=np.float32)
                test_d_map = np.asarray(sio.loadmat(test_gt_list[img_idx])['d_map'], dtype=np.float32)                

                # Reshape the input image for feeding it to the network.
                test_image = utils.reshape_tensor(test_image)
                feed_dict_data = {image_place_holder: test_image}

                # Make prediction.
                pred = sess.run(d_map_est, feed_dict=feed_dict_data)                

                # Compute mean absolute error.
                abs_err += utils.compute_abs_err(pred, test_d_map)

            # Average across all the images.
            avg_mae = abs_err / len(test_image_list)
            print ("Mean Absolute Error over the Test Set: %s" %(avg_mae))
            print ('Finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--retrain', default=False, type=bool)
    #parser.add_argument('--base_model_path', default=None, type=str)
    parser.add_argument('--log_dir', default = './logs', type=str)
    parser.add_argument('--num_epochs', default = 200, type=int)
    parser.add_argument('--learning_rate', default = 0.01, type=float)
    parser.add_argument('--session_id', default = 2, type=int)
    parser.add_argument('--data_root', default='./data/comb_dataset_v3', type=str)

    args = parser.parse_args()

    #if args.retrain:
    #    if args.base_model_path is None:
    #        print "Please provide a base model path."
    #        sys.exit()
    #    else:
    #        main(args)
    #else:
    main(args)
