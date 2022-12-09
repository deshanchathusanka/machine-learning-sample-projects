from dimensionality_reduction import umap_process
import scipy
import numpy as np

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mat = scipy.io.loadmat('dimensionality_reduction/data/cube.mat')
    img_data = mat['cube']
    reduced_image_dataset = umap_process.umap_reduction(img_data)
    np.save('dimensionality_reduction/data/umap_dataset.npy', reduced_image_dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
