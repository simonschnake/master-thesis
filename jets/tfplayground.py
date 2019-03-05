import tensorflow as tf
# Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
nbins = 5
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
data = [1., 1., 1., 1., 1., 1.]
indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)
hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
seg = tf.math.unsorted_segment_mean(data, indices, 5)
end = tf.reduce_mean(seg)

with tf.Session() as sess:
    res = sess.run(indices)
    print(res)
    res = sess.run(hist)
    print(res)
    res = sess.run(seg)
    print(res)
    res = sess.run(end)
    print(res)

