import sys
import argparse
from core import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoint')
parser.add_argument('--resume', type=str, default=None, help='model directory for finetune training')
parser.add_argument('--txt_file', default='data/train.txt', type=str, help='path to mat file')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--epoch', type=int, default=300, help='number of training epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--lr_decay', default=10, type=int, help='learning rate decay rate')
parser.add_argument('--gamma', default=0.9, type=float, help='gamma update for optimizer')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in data loading')
parser.add_argument('--log', default='default', type=str, help='training log')

args = parser.parse_args()


def train():

    best_loss = np.inf
    writer = SummaryWriter(os.path.join('logs', args.log))
    dummy_input = torch.rand(1, 3, 64, 64)

    train_dataset = SVHNLoader(args.txt_file)

    data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  collate_fn=detection_collate, shuffle=True)

    model = NumberNet()
    writer.add_graph(model, (dummy_input, ))

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay, args.gamma)
    #######################################################################################

    for epoch in np.arange(args.epoch):

        writer.add_scalar('Train/learning rate', optimizer.param_groups[0]['lr'], epoch)

        average_loss = 0.0

        for iteration, (images, labels, length) in enumerate(data_loader):

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            length = Variable(length.cuda())

            l, digits = model(images)

            optimizer.zero_grad()
            loss = criterion(l, digits, length, labels)
            loss.backward()
            optimizer.step()

            average_loss = ((average_loss * iteration) + loss.item()) / (iteration + 1)

            writer.add_scalar('Train/loss', loss.item(), epoch * len(data_loader) + iteration)

            count = round(iteration / len(data_loader) * 50)
            sys.stdout.write('[Epoch {}], {}/{}: [{}{}] Avg_loc loss: {:.4}\r'.format(
                epoch, iteration + 1, len(data_loader),'#' * count, ' ' * (50 - count), average_loss))

        sys.stdout.write('\n')

        writer.add_scalar('Train/Global_avg_loss', average_loss, epoch)

        for key, param in model.named_parameters():
            writer.add_histogram(key, param.clone(), epoch)

        if best_loss > average_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.log, 'NumberNet_{}.pth'.format(epoch)))
            print('Epoch: {} model is saved'.format(epoch))

        scheduler.step()


if __name__ == '__main__':

    if not os.path.exists(os.path.join(args.save_dir, args.log)):
        os.makedirs(os.path.join(args.save_dir, args.log))

    train()
