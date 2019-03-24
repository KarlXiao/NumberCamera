import h5py
import argparse
import tqdm
import os
import cv2

parser = argparse.ArgumentParser('scrip to check SVHN data')
parser.add_argument('--type', required=True, type=str, help='data set to parse')

args = parser.parse_args()


def read_list(path):
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                file_list.append(os.path.join(root, file))
    return file_list


def run():
    image_path = os.path.split('data/SVHN/{}/digitStruct.mat'.format(args.type))[0]
    im_list = read_list(image_path)

    f = h5py.File('data/SVHN/{}/digitStruct.mat'.format(args.type), 'r')
    data = f['digitStruct']['bbox']
    with open('data/{}.txt'.format(args.type), 'w') as save:
        for idx in tqdm.trange(len(im_list)):
            im_path = im_list[idx]
            item = data[int(os.path.basename(im_path).strip('.png'))-1].item()
            attrs = {}
            for key in ['label', 'left', 'top', 'width', 'height']:
                attr = f[item][key]
                values = [f[attr.value[i].item()].value[0][0]
                        for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
                attrs[key] = values
            labels = attrs['label']
            length = len(labels)
            if length > 5:
                continue
            left, top, width, height = map(lambda x: [int(i) for i in x],
                                           [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
            left, top, right, bottom = (min(left), min(top),
                                        max(map(lambda x, y: x + y, left, width)),
                                        max(map(lambda x, y: x + y, top, height)))
            center_x, center_y, max_size = ((left + right) / 2.0, (top + bottom) / 2.0,
                                            max(right-left, bottom - top))
            left, top, width, height = (center_x - max_size / 2.0, center_y - max_size / 2.0, max_size, max_size)
            left, top = (left - 0.15 * width, top - 0.15 * height)
            left, top = max(left, 0), max(top, 0)
            max_side = 1.3 * max(width, height)

            save.write('{} {} {} {} {} {}'.format(im_path, int(left), int(top),
                                                  int(left + max_side), int(top + max_side), length))
            for digit in labels:
                if digit == 10:
                    digit = 0
                save.write(' {}'.format(int(digit)))
            save.write('\n')


if __name__ == '__main__':
    run()
