from umap import UMAP
import numpy as np

def umap_reduction(data_np):
    '''
    :param data_np: the input data (m,d)
    :return: dimensionality reduced data  (m,n)
    number of original dimensions = d
    number of reduced dimensions = n
    '''
    process_image_dataset = data_np * 255
    process_image_dataset = np.transpose(process_image_dataset.astype(np.float32), (0, 1, 3, 2))
    flat_image_dataset = process_image_dataset.reshape((224 * 224 * 72, 7))

    print(flat_image_dataset.shape)
    _umap = UMAP(n_components=3, n_neighbors=100)
    umap_image_dataset = _umap.fit_transform(flat_image_dataset)
    print(umap_image_dataset.shape)
    reduced_image_dataset = umap_image_dataset.reshape((224, 224, 72, 3))
    reduced_image_dataset = np.transpose(reduced_image_dataset.astype(np.uint8), (0, 1, 3, 2))
    return reduced_image_dataset
