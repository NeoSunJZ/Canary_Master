import base64
from io import BytesIO
from matplotlib import pyplot as plt


def show_img_diff(original_img, adversarial_img):
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img

    difference = difference / abs(difference).max() / 2.0 + 0.5

    plt.imshow(difference, cmap=plt.cm.gray)
    plt.margins(0, 0)

    plt.axis('off')
    plt.tight_layout()
    f = plt.gcf()
    buffer = BytesIO()
    f.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0.0)
    return "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()), "utf-8")
