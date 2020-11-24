import os
import cv2
import pickle
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

""" ------------------------------------------ Embeddings saving ------------------------------------------ """


def save_metadata(embeddings, embeddings_dir, limit=1000):
    tf.logging.info('Saving embeddings metadata...')

    if not os.path.isdir(embeddings_dir):
        os.makedirs(os.path.join(embeddings_dir))

    metadata_path = os.path.join(embeddings_dir, 'metadata.tsv')

    length = len(embeddings['labels'])

    indices = np.random.choice(length, limit)

    embeddings['filenames'] = embeddings['filenames'][indices]
    embeddings['captions'] = embeddings['captions'][indices]
    embeddings['labels'] = embeddings['labels'][indices]
    embeddings['values'] = embeddings['values'][indices]

    with open(metadata_path, 'w+') as f:
        f.write('Index\tCaption\tLabel\tFilename\n')
        for idx in range(limit):
            f.write('{:05d}\t{}\t{}\t{}\n'.format(idx,
                                                  embeddings['captions'][idx],
                                                  embeddings['labels'][idx],
                                                  embeddings['filenames'][idx]))
        f.close()

    with tf.Session() as sess:
        # The embedding variable to be stored
        embedding_var = tf.Variable(np.array(embeddings['values']), name='embeddings')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        embedding = config.train_embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Add metadata to the log
        embedding.metadata_path = metadata_path

        img_data = create_sprite(embeddings['filenames'])

        cv2.imwrite(os.path.join(embeddings_dir, 'sprite.png'), img_data)

        embedding.sprite.image_path = os.path.join(embeddings_dir, 'sprite.png')
        embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[0]])

        writer = tf.summary.FileWriter(embeddings_dir, sess.graph)
        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(embeddings_dir, "model_emb.ckpt"), 1)

    tf.logging.info('Metadata saved.')


def save_pickle(embeddings, pickle_dir, name='all'):
    tf.logging.info('Saving embeddings to pickle...')

    pickle_path = os.path.join(pickle_dir, name + '.pickle')

    pickle.dump(embeddings, open(pickle_path, "wb"))

    tf.logging.info('Pickle saved.')


def read_pickle(pickle_dir, name='all'):
    tf.logging.info('Reading embeddings from pickle...')

    pickle_path = os.path.join(pickle_dir, name + '.pickle')

    return pickle.loads(open(pickle_path, "rb").read())


def save_sqlite(embeddings, sqlite_dir, name='all'):
    """ ------------------------------------------ SQlite saving ------------------------------------------ """

    tf.logging.info('Saving embeddings to sqlite database...')

    sqlite_path = os.path.join(sqlite_dir, name + '.db')

    conn = sqlite3.connect(sqlite_path)

    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE features (
                        feature_id INTEGER PRIMARY KEY,
                        label_id TEXT,
                        label_caption TEXT,
                        feature TEXT                         
                        )''')

    v = [(i, str(embeddings['labels'][i]), embeddings['captions'][i], ', '.join(str(x) for x in embeddings['values'][i]))
         for i in range(len(embeddings['labels']))]

    # Insert a row of data
    c.executemany("INSERT INTO features VALUES (?, ?, ?, ?)", v)

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()

    tf.logging.info('Database saved.')


def create_sprite(filenames, single_image_dim=(32, 32)):
    """Creates the sprite image along with any necessary padding

    Args:
      filenames: list containing the filenames.
      image_dir: string containing the directory of images.
      single_image_dim: tuple with size of one image.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    img_data = []
    for filename in filenames:
        # print(os.path.join(image_dir, filename))
        input_img = cv2.imread(filename)
        input_img_resize = cv2.resize(input_img, single_image_dim)  # you can choose what size to resize your data
        img_data.append(input_img_resize)
    img_data = np.array(img_data)

    # For B&W or greyscale images
    if len(img_data.shape) == 3:
        img_data = np.tile(img_data[..., np.newaxis], (1, 1, 1, 3))

    # img_data = img_data.astype(np.float32)
    # min = np.min(img_data.reshape((img_data.shape[0], -1)), axis=1)
    # img_data = (img_data.transpose((1, 2, 3, 0)) - min).transpose((3, 0, 1, 2))
    # max = np.max(img_data.reshape((img_data.shape[0], -1)), axis=1)
    # img_data = (img_data.transpose((1, 2, 3, 0)) / max).transpose((3, 0, 1, 2))
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(img_data.shape[0])))
    padding = ((0, n ** 2 - img_data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (img_data.ndim - 3)

    img_data = np.pad(img_data, padding, mode='constant', constant_values=0)
    # Tile the individual thumbnails into an image.
    img_data = img_data.reshape((n, n) + img_data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, img_data.ndim + 1)))
    img_data = img_data.reshape((n * img_data.shape[1], n * img_data.shape[3]) + img_data.shape[4:])
    img_data = img_data.astype(np.uint8)

    return img_data


def map_embeddings(embeddings, df=None):

    filenames = embeddings['filenames']
    splits = embeddings['splits']
    values = embeddings['values']
    labels = embeddings['labels']
    captions = embeddings['captions']

    if df is not None:

        splits = df.split.values

        query = df.filename.values

        orig_indices = filenames.argsort()  # indices that would sort the array
        i = np.searchsorted(filenames[orig_indices], query)  # indices in the sorted array
        sorted_indices = orig_indices[i]  # indices with respect to the original array

        filenames = filenames[sorted_indices]
        values = values[sorted_indices]
        labels = labels[sorted_indices]
        captions = captions[sorted_indices]

    train_indices = np.where(splits == 'train')
    test_indices = np.where(splits == 'test')

    return {
        'values': values[train_indices],
        'labels': labels[train_indices],
        'captions': captions[train_indices],
        'filenames': filenames[train_indices],
    }, {
        'values': values[test_indices],
        'labels': labels[test_indices],
        'captions': captions[test_indices],
        'filenames': filenames[test_indices],
    }