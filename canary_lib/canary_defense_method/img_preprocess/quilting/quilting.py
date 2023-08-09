import numpy as np
from PIL import Image
import faiss
from torchvision.transforms import ToTensor

# 超参数定义
M = 8  # 每个块的大小
K = 5  # Kmeans聚类的簇数


def quilting(image, quilting_size=8, kemeans=5):
    M = quilting_size
    K = kemeans
    image = image.numpy()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    # image = ToPILImage()(image)
    # 读取图像
    # image = Image.open(image_path).convert('RGB')

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
    d = patches.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, K)
    index.train(patches)
    index.add(patches)
    _, I = index.search(patches, 2)

    # 构建输出图像
    out_image = Image.new('RGB', (width, height))
    for i in range(NX):
        for j in range(NY):
            patch_index = i * NX + j
            neighbor_index = I[patch_index][1]
            x = i * M
            y = j * M
            neighbor_patch = patches[neighbor_index].reshape((M, M, 3)).astype('uint8')
            neighbor_image = Image.fromarray(neighbor_patch)
            out_image.paste(neighbor_image, (x, y))
    img = ToTensor()(out_image)
    img = img.permute(1, 2, 0)
    img *= 255
    # 返回输出图像
    return img
