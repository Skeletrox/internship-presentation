import tensorflow as tf
import numpy as np
import json
rnd = np.random


def prepare_train_data(vals):
    X = []
    Y_lower = []
    Y_upper = []
    for v in vals:
        X.append([
                  v['byte_packet_ratio'],
                  v['bytes_out'],
                  v['byte_rate']
                ])
        c = v['class']
        Y_lower.append([1 if c > 0 else 0])
        Y_upper.append([1 if c > 0.5 else 0])
    return np.asarray(X), np.asarray(Y_lower), np.asarray(Y_upper)


def prepare_test_data(vals):
    X = []
    times = []
    for v in vals:
        X.append([
                  v['byte_packet_ratio'],
                  v['bytes_out'],
                  v['byte_rate']
                ])
        times.append(v['time_avg'])
    return np.asarray(X), np.asarray(times)


def tf_classify_train():

    alpha = 0.05
    num_epochs = 1000

    with open('pb_train.json') as pbt:
        values = json.loads(pbt.read())
    X_in, Y_lower, Y_upper = prepare_train_data(values)
    m = len(X_in)
    n = len(X_in[0])
    X = tf.placeholder(tf.float32, shape=[m, n], name='Inputs')
    Y_L = tf.placeholder(tf.float32, shape=[m, 1], name='Outputs_Lower')
    Y_U = tf.placeholder(tf.float32, shape=[m, 1], name='Outputs_Upper')
    w_lower = tf.Variable(tf.zeros([n, 1]), name='Weights_Lower')
    w_upper = tf.Variable(tf.zeros([n, 1]), name='Weights_Upper')

    pred_lower = tf.matmul(X, w_lower)
    pred_upper = tf.matmul(X, w_upper)

    lower_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_lower, labels=Y_L))
    upper_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_upper, labels=Y_U))

    low_optimizer = tf.train.AdamOptimizer(alpha).minimize(lower_cost)
    up_optimizer = tf.train.AdamOptimizer(alpha).minimize(upper_cost)

    init = tf.global_variables_initializer()

    tf.summary.scalar("Cost_Lower", lower_cost)
    tf.summary.scalar("Cost_Upper", upper_cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter('./logs/multi/train', sess.graph)
        print("Initialized Variables")

        for epoch in range(num_epochs):
            sess.run(low_optimizer, feed_dict={X: X_in, Y_L: Y_lower})
            sess.run(up_optimizer, feed_dict={X: X_in, Y_U: Y_upper})

            if epoch % 50 == 0:
                print ("Finished %s epochs" %(epoch))
                print ("Cost at lower classifier: ", sess.run(lower_cost, feed_dict={X: X_in, Y_L: Y_lower}))
                print ("Cost at upper classifier: ", sess.run(upper_cost, feed_dict={X: X_in, Y_U: Y_upper}))
                print ("======================================")
                merge = tf.summary.merge_all()
                summary = sess.run(merge, feed_dict={X: X_in, Y_L: Y_lower, Y_U: Y_upper})
                train_writer.add_summary(summary, epoch)

        save_path = saver.save(sess, "models/classifier.ckpt")
        print("Model saved in path: %s" % save_path)
    return 0


def tf_classify_predict():
    response = []
    with open('pb_classify.json') as pbc:
        values = json.loads(pbc.read())
    X_test, times = prepare_test_data(values)
    n = len(X_test[0])
    w_lower = tf.Variable(tf.zeros([n, 1]), name='Weights_Lower')
    w_upper = tf.Variable(tf.zeros([n, 1]), name='Weights_Upper')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'models/classifier.ckpt')
        weights_lower = w_lower.eval()
        weights_upper = w_upper.eval()
        for x, t in zip(X_test, times):
            obtained_lower = 1/(1+np.exp(-np.matmul(x, weights_lower)))
            obtained_upper = 1/(1+np.exp(-np.matmul(x, weights_upper)))
            total = (round(obtained_upper[0]) + round(obtained_lower[0]))/2
            if total < 0.25:
                packet_type = 'slow'
            elif total < 0.75:
                packet_type = 'normal'
            else:
                packet_type = 'sparse'
            response.append({
                'result': packet_type,
                'timestamp': t,
                'total': total
            })

    return response
