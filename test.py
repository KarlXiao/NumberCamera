import sys
import argparse
from core import *
from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, required=True, help='model directory for finetune training')
parser.add_argument('--txt_file', required=True, type=str, help='path to mat file')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in data loading')

args = parser.parse_args()


def val():
    train_dataset = SVHNLoader(args.txt_file)

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=True)

    model = NumberNet()
    model.eval()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)

    model = model.cuda()

    #######################################################################################
    accuracy = 0
    for iteration, (images, labels, length) in enumerate(data_loader):

        images = Variable(images.cuda())

        l, digits = model(images)

        l_pre = torch.argmax(l.softmax(dim=-1)).item()
        label_pre = torch.argmax(torch.squeeze(digits.softmax(dim=2)), dim=1).cpu().numpy()

        if l_pre == length.item() and (label_pre == labels.numpy()).all():
            accuracy += 1
        # else:
            # print('length:{} labels:{}'.format(l_pre, label_pre))
            # im = np.transpose(images.cpu().numpy()[0, :, :, :], [1, 2, 0])
            # cv2.imshow('Error', im)
            # cv2.waitKey(0)
        count = round(iteration / len(data_loader) * 50)
        sys.stdout.write('[{}/{}: [{}{}]\r'.format(iteration + 1, len(data_loader), '#' * count, ' ' * (50 - count)))

    sys.stdout.write('\n')
    print('Accuracy:{}'.format(accuracy / len(data_loader)))


if __name__ == '__main__':
    val()
