from utils import *

def single_report(val_gen, num):
    for i in range(num):
        test_inputs, test_targets, test_seq_len = val_gen.next_batch()
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd = session.run(decoded[0], test_feed)
        detected_list = decode_sparse_tensor(dd)
        for idx, number in enumerate(detected_list):
            print("Test Accuracy:", detected_list[idx])

with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    testi='./exp'
    #saver.restore(session, './model8.24best/LPRtf3.ckpt-25000')
    saver.restore(session, './model/LPRtf3.ckpt-15000')
    test_gen = TextImageGenerator(img_dir=testi,
                                           label_file=None,
                                           batch_size=BATCH_SIZE,
                                           img_size=img_size,
                                           num_channels=num_channels,
                                           label_len=label_len)

    single_report(test_gen,1)

