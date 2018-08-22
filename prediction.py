import argparse
import tensorflow as tf
import src.mccnn as mccnn
import cv2
import src.utils as utils
import numpy as np
import time
import math

def predict(modelpath, videopath):
    G = tf.Graph()
    with G.as_default():
        img_placeholder = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        dm_est = mccnn.build(img_placeholder)
        capture = cv2.VideoCapture()
        success = capture.open(videopath)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not success:
            print "Couldn't open video %s." % videopath
        sess = tf.Session(graph=G)
        with sess.as_default():
            utils.load_weights(G, modelpath)
            for i in xrange(total_frames):
                _, frame = capture.read()
                frame_resized = np.asarray(cv2.resize(frame, dsize=(480,270)), dtype=np.float32)
                #frame_resized = np.asarray(cv2.resize(frame, dsize=(640,480)), dtype=np.float32)
                frame_disp = np.copy(frame_resized)
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)                           # Convert to grayscale
                frame_resized = utils.reshape_tensor(frame_resized)
                start = time.time()
                pred = sess.run(dm_est, {img_placeholder: frame_resized})
                pred = np.reshape(pred, newshape=(pred.shape[1], pred.shape[2]))
                count = np.sum(pred[:])
                end = time.time()
                print "Time for prediction: %.5f secs." % (end - start)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_disp, "Crowd Count: %s" % (math.ceil(count)), (10, 30), font, 0.8, (0, 255, 0), 2)
                pred_disp = np.copy(pred)
                pred_disp = cv2.resize(pred_disp, dsize=(frame_disp.shape[1], frame_disp.shape[0]))
                pmin = np.amin(pred_disp)
                pmax = np.amax(pred_disp)
                pred_disp_n = (pred_disp - pmin) / (pmax - pmin)
                pred_disp_n = pred_disp_n * 255
                pred_disp_n = np.uint8(pred_disp_n)
                pred_disp_color = cv2.applyColorMap(pred_disp_n, cv2.COLORMAP_JET)
                output_image = np.zeros((frame_disp.shape[0], frame_disp.shape[1] * 2, 3), dtype=np.uint8)
                output_image[0:frame_disp.shape[0], 0:frame_disp.shape[1]] = frame_disp
                output_image[0:frame_disp.shape[0], frame_disp.shape[1]:] = pred_disp_color
                output_image = cv2.resize(output_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Display window', output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/weights.comb.npz')
    parser.add_argument('--video_path', type=str)
    args = parser.parse_args()
    predict(args.model_path, args.video_path)
