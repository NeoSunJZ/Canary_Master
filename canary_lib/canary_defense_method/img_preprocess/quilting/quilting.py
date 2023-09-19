import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.transforms import ToTensor


def quilting(image, quilting_size=2, kemeans=16):
    M = quilting_size
    K = kemeans
    image = image.numpy()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    # 获取图像宽和高
    width, height = image.size

    # 将图像分割为N个大小为M x M的小块
    NX = width // M
    NY = height // M
    patches = []
    for i in range(NX):
        for j in range(NY):
            x1 = i * M
            y1 = j * M
            x2 = x1 + M
            y2 = y1 + M
            patch = image.crop((x1, y1, x2, y2))
            patch_array = np.array(patch).flatten().astype('float32')
            patches.append(patch_array)

    # 运行Kmeans聚类
    patches = np.vstack(patches)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(patches)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 构建输出图像
    out_image = Image.new('RGB', (width, height))
    for i in range(NX):
        for j in range(NY):
            patch_index = i * NX + j
            label = labels[patch_index]
            centroid_patch = centroids[label].reshape((M, M, 3)).astype('uint8')
            centroid_image = Image.fromarray(centroid_patch)
            x = i * M
            y = j * M
            out_image.paste(centroid_image, (x, y))
    img = ToTensor()(out_image)
    img = img.permute(1, 2, 0)
    img *= 255
    # 返回输出图像
    return img
