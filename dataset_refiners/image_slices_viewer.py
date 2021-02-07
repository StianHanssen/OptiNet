"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""
# Credit: https://matplotlib.org/gallery/event_handling/image_slices_viewer.html
# Modified version
# Used for debugging and checking of HDF5 storage_batches

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import h5py
from time import time

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X.reshape(X.shape[1:])
        _, self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :])
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

# Input shape (depth, height, width)
def display_3d_image(image):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def load_options():
    parser = OptionParser()
    parser.add_option("-n", "--name", dest="batch_name",
                      help="Name of hdf5 batch to get a volume to view",
                      default="datasets/duke_refined/0_storage_batch.hdf5", type=str)
    parser.add_option("-i", "--index", dest="index",
                      help="Index to select a volume in batch",
                      default=0, type=int)
    parser.add_option("-k", "--key", dest="key",
                      help="Key to access data in h5py dataset",
                      default="data", type=str)
    
    options, _ = parser.parse_args()
    return options

if __name__ == '__main__':
    options = load_options()
    dataset = h5py.File(options.batch_name, mode='r')
    data = np.array(dataset[options.key], dtype=np.float32)
    image = data[options.index]
    display_3d_image(image)
    answer = ""
    while answer != 'exit':
        answer = input("Pick another index to view or type 'exit': ")
        if answer.isdigit() or (answer[0] == '-' and answer[1:].isdigit()):
            display_3d_image(data[int(answer)])
        elif answer != 'exit':
            print("Expected index or 'exit'")