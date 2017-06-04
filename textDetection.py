import numpy as np
import tensorflow as tf

from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.morphology import closing, square
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border

from tensorflow.python.platform import gfile
from createBottleneck import import_inception, create_img_bottleneck

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shutil
import os
import scipy.misc

LABEL_FILE = './my_label.txt'
GRAPH_FILE = './my_graph.pb'

ECCENTRICYTY = 0.995
SOLIDITY = 0.3
EXTENT = 0.2
EULER = -4
RATIO = 3


class textDetection:
    def __init__(self, image_path):
        print('Image reading...')
        self.image = imread(image_path, as_grey=True)
        self.bw = self.imagePreProcess()
        if os.path.exists('./sample'):
            shutil.rmtree('./sample')
        if not os.path.exists('./sample'):
            os.makedirs('./sample')

    def imagePreProcess(self):
        print('Image Preprocessing...')
        image = denoise_tv_chambolle(self.image, weight=0.1)
        thres = threshold_otsu(image)
        bw = closing(image > thres, square(2))
        clear_border(bw)
        print('Done')
        print('-----')
        return bw

    def getTextCandidate(self):
        print('Getting text candidates from image...')
        label_black = label(self.bw, background=1)
        label_white = label(self.bw, background=0)
        candidateResult = []
        candidatePosition = []
        n = 0

        for region in regionprops(label_black):
            minr, minc, maxr, maxc = region.bbox
            if region.eccentricity > ECCENTRICYTY:
                continue
            if region.solidity < SOLIDITY:
                continue
            if region.extent < EXTENT:
                continue
            if region.euler_number < EULER:
                continue
            if (maxc - minc) / (maxr - minr) > RATIO:
                continue
            if (minr == 0):
                continue
            if (region.area > (0.002 * (self.image.shape[0] * self.image.shape[1]))):
                margin = 0
                minr = minr - margin
                minc = minc - margin
                maxr = maxr + margin
                maxc = maxc + margin
                sample = self.bw[minr:maxr, minc:maxc]
                if sample.shape[0] * sample.shape[1] == 0:
                    continue
                else:
                    sample = resize(sample, (100, int(sample.shape[1] * (100 / sample.shape[0]))))
                    self.sampleSave(sample, str(n))
                    # self.sampleSave(sample, str(n))
                    candidateResult.append('no_text')
                    candidatePosition.append(region.bbox)
                    n = n + 1

        for region in regionprops(label_white):
            minr, minc, maxr, maxc = region.bbox
            if region.eccentricity > ECCENTRICYTY:
                continue
            if region.solidity < SOLIDITY:
                continue
            if region.extent < EXTENT:
                continue
            if region.euler_number < EULER:
                continue
            if (maxc - minc) / (maxr - minr) > RATIO:
                continue
            if (minr == 0):
                continue
            if (region.area > (0.002 * (self.image.shape[0] * self.image.shape[1]))):
                margin = 0
                minr = minr - margin
                minc = minc - margin
                maxr = maxr + margin
                maxc = maxc + margin
                sample = self.bw[minr:maxr, minc:maxc]
                if sample.shape[0] * sample.shape[1] == 0:
                    continue
                else:
                    sample = resize(sample, (100, int(sample.shape[1] * (100 / sample.shape[0]))))
                    sample = 1 - sample
                    self.sampleSave(sample, str(n))
                    # self.sampleSave(sample, str(n))
                    candidateResult.append('no_text')
                    candidatePosition.append(region.bbox)
                    n = n + 1

        self.numMax = np.array(candidatePosition).shape[0]
        need_to_del = False
        delete = []
        for i in range(self.numMax):
            minr1 = candidatePosition[i][0]
            minc1 = candidatePosition[i][1]
            maxr1 = candidatePosition[i][2]
            maxc1 = candidatePosition[i][3]
            for j in range(self.numMax):
                minr2 = candidatePosition[j][0]
                minc2 = candidatePosition[j][1]
                maxr2 = candidatePosition[j][2]
                maxc2 = candidatePosition[j][3]
                if(minc1 >= minc2):
                    if(minr1 >= minr2):
                        if(maxc1 < maxc2):
                            if(maxr1 < maxr2):
                                need_to_del = True
            if need_to_del is True:
                delete.append(i)
            need_to_del = False
        for i in range(np.array(delete).shape[0]):
            index = delete[i] - i
            del candidateResult[index]
            del candidatePosition[index]
            os.remove(os.path.join('./sample', str(delete[i]) + '.jpg'))
        # self.candidate = {'fullscale': np.array(candidateValue),
        #                   'position': np.array(candidatePosition)
        #                   }
        self.candidate = {'position': np.array(candidatePosition),
                          'result': candidateResult}
        num = self.candidate['position'].shape[0]
        print(num, 'candidates found!!!')
        print('-----')

    def showCandidate(self):
        if self.candidate['position'].shape[0] == 0:
            print('Chua co candidate')
            print('Su dung ham getTextCandidate()')
            return
        fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        for i in range(self.candidate['position'].shape[0]):
            minr = self.candidate['position'][i][0]
            minc = self.candidate['position'][i][1]
            maxr = self.candidate['position'][i][2]
            maxc = self.candidate['position'][i][3]
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.set_axis_off()
        plt.show()

    def sampleSave(self, sample, name):
        scipy.misc.imsave(os.path.join('./sample', name + '.jpg'), sample)
        imgSmall = Image.open(os.path.join('./sample', name + '.jpg'))
        sizeImgSmall = imgSmall.size
        sizeImgBig = (128, 128)
        imgBig = Image.new('RGB', sizeImgBig, (255, 255, 255))
        imgBig.paste(imgSmall, (int((sizeImgBig[0] - sizeImgSmall[0]) / 2),
                                int((sizeImgBig[1] - sizeImgSmall[1]) / 2)))
        imgBig.save(os.path.join('./sample', name + '.jpg'))

    def letterClassify(self):
        sess = tf.Session()
        graph, bottleneck_tensor, img_tensor, resized_img_tensor = import_inception(
            './classify_image_graph_def.pb')
        sample_bottleneck = []
        for n in range(self.numMax):
            if os.path.exists(os.path.join('./sample', str(n) + '.jpg')):
                img_data = gfile.FastGFile(os.path.join(
                    './sample', str(n) + '.jpg'), 'rb').read()
                bottleneck = create_img_bottleneck(
                    sess, img_data, img_tensor, bottleneck_tensor)
                sample_bottleneck.append(bottleneck)
        label_lines = [line.rstrip() for line
                       in tf.gfile.GFile(LABEL_FILE)]

        with tf.gfile.FastGFile(GRAPH_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        sess = tf.Session()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        samplePredict = []

        for n in range(np.array(sample_bottleneck).shape[0]):
            predictions = sess.run(softmax_tensor,
                                   {'Input_ts:0': [sample_bottleneck[n]]})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            human_string = label_lines[top_k[0]]
            # score = predictions[0][top_k[0]]
            # print('%s (score = %.5f)' % (human_string, score))
            # print('----------------------------------------------')
            samplePredict.append(human_string)
        self.candidate['result'] = samplePredict
        return self.candidate['result']

    def textReconstruct(self):
        for i in range(self.candidate['position'].shape[0]):
            Ymin = i
            for j in range(self.candidate['position'].shape[0] - i):
                Ytemp = i + j
                if (self.candidate['position'][Ytemp][1] < self.candidate['position'][Ymin][1]):
                    Ymin = Ytemp
            self.candidate['position'][i][0], self.candidate['position'][Ymin][0] = self.candidate['position'][Ymin][0], self.candidate['position'][i][0]
            self.candidate['position'][i][1], self.candidate['position'][Ymin][1] = self.candidate['position'][Ymin][1], self.candidate['position'][i][1]
            self.candidate['position'][i][2], self.candidate['position'][Ymin][2] = self.candidate['position'][Ymin][2], self.candidate['position'][i][2]
            self.candidate['position'][i][3], self.candidate['position'][Ymin][3] = self.candidate['position'][Ymin][3], self.candidate['position'][i][3]
            self.candidate['result'][i], self.candidate['result'][Ymin] = self.candidate['result'][Ymin], self.candidate['result'][i]
        print('Result:')
        print(*self.candidate['result'], sep='')
