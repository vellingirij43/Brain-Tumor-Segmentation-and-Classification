import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def Image_Results():
    I = [[301, 306, 322, 335, 450]]
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('GT.npy', allow_pickle=True)
    UNET = np.load('UNET.npy', allow_pickle=True)
    RESUNET = np.load('RESUNET.npy', allow_pickle=True)
    PROPOSED = np.load('PROPOSED.npy', allow_pickle=True)
    for i in range(len(I[0])):
        plt.subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(Images[I[0][i]])
        plt.subplot(2, 3, 2)
        plt.title('GroundTruth')
        plt.imshow(GT[I[0][i]])
        plt.subplot(2, 3, 3)
        plt.title('UNET')
        plt.imshow(UNET[I[0][i]])
        plt.subplot(2, 3, 4)
        plt.title('RESUNET')
        plt.imshow(RESUNET[I[0][i]])
        plt.subplot(2, 3, 5)
        plt.title('PROPOSED')
        plt.imshow(PROPOSED[I[0][i]])
        plt.tight_layout()
        plt.show()
        # cv.imwrite('./Results/Image_Results/' + 'orig-' + str(i + 1) + '.png', Images[I[0][i]])
        # cv.imwrite('./Results/Image_Results/' + 'gt-' + str(i + 1) + '.png', GT[I[0][i]])
        # cv.imwrite('./Results/Image_Results/' + 'unet-' + str(i + 1) + '.png', UNET[I[0][i]])
        # cv.imwrite('./Results/Image_Results/' + 'resunet-' + str(i + 1) + '.png',
        #            RESUNET[I[0][i]])
        # cv.imwrite('./Results/Image_Results/' + 'proposed-' + str(i + 1) + '.png',
        #            PROPOSED[I[0][i]])


def Sample_Images():
    Orig = np.load('Images.npy', allow_pickle=True)
    ind = [40, 50, 80, 100, 150, 200]
    fig, ax = plt.subplots(2, 3)
    plt.suptitle("Sample Images")
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    # plt.show()
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(0 + 1) + '.png', Orig[ind[0]])
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(1 + 1) + '.png', Orig[ind[1]])
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(2 + 1) + '.png', Orig[ind[2]])
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(3 + 1) + '.png', Orig[ind[3]])
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(4 + 1) + '.png', Orig[ind[4]])
    # cv.imwrite('./Results/Sample_Image/' + 'img-' + str(5 + 1) + '.png', Orig[ind[5]])


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
