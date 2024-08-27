#! /usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import tensorflow as tf
import numpy as np


def open_interactive_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


def open_and_close_interactive_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.close()


def open_and_close_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', help='repeat', type=int, default=5)
    parser.add_argument('type', choices=['interactive', 'interactive_close', 'normal'])
    args = parser.parse_args()

    sess_func = open_and_close_session

    if args.type == 'interactive':
        sess_func = open_interactive_session
    elif args.type == 'interactive_close':
        sess_func = open_and_close_interactive_session

    for _ in range(args.num):
        sess_func()
    with tf.Session() as sess:
        print("bytes used=", sess.run(tf.contrib.memory_stats.BytesInUse()))