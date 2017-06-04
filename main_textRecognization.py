import os
from textDetection import textDetection
from sys import argv

# Prevent unnecessary warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_dir = argv[1]

test = textDetection(img_dir)
test.getTextCandidate()
result = test.letterClassify()
test.showCandidate()
test.textReconstruct()
