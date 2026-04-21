from PIL import Image
from urllib.request import urlopen

import glob
import os

def get_verify(img_file):
    try:
        verify_response = Image.open(img_file).verify()
        # print(f'{verify_response} was returned by verify')
        im = Image.open(img_file)
        # print(f'{im.filename.split("/")[-1]} is a valid image')
        # im.show()
    except Exception:
        print('Invalid image:', img_file)


# get_verify("/home/quachmd/Bureau/depth-correction/knowledge-distillation/data/lingbot/lingbot-depth_pub/RobbyReal/00010_45_library/orbbec_335L_CP2L853000E8/color/0004546.jpg")
# get_verify("test_images/AtoZ.jpg")

dir = '/home/quachmd/Bureau/depth-correction/datasets/lingbot-depth_pub/RobbyReal'

color_pattern = os.path.join(dir, '**', 'color', '*.jpg')
files = sorted(glob.glob(color_pattern, recursive=True))

for i, file in enumerate(files):
    # print(file)
    get_verify(file)
    print('File', i)

print("Done 1")
depth_pattern = os.path.join(dir, '**', 'rawdepth', '*.png')
files = sorted(glob.glob(depth_pattern, recursive=True))

for i, file in enumerate(files):
    print("Depth", i)
    get_verify(file)

