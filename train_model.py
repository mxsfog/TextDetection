import os
import shutil
import time
from enum import Enum
import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import Conf
from Dataset import ImageDataset, train_collate_func, val_test_collate_func
from decoder import ctc_decoder
from Model import CRNN

def main():
    start_epoch = 0
    best_acc = -0.1
    train_dataloader, test_dataloader = load_dataset()


def load_dataset() -> [DataLoader, DataLoader]:

    train_dataset = ImageDataset(dataroot=Conf.mode_config['train']['train_dataroot'],
                                 annotation_file_name=Conf.mode_config['train']['annotation_train_file_name'],
                                 labels_dict=Conf.labels,
                                 image_width=Conf.model_image_width,
                                 image_height=Conf.model_image_height,
                                 mean=Conf.mean,
                                 std=Conf.std,
                                 mode="train")

    test_dataset = ImageDataset(dataroot=Conf.mode_config['test']['train_dataroot'],
                                annotation_file_name=Conf.mode_config['test']['annotation_train_file_name'],
                                labels_dict=Conf.labels,
                                image_width=Conf.model_image_width,
                                image_height=Conf.model_image_height,
                                mean=Conf.mean,
                                std=Conf.std,
                                mode="test")

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=Conf.mode_config['train']['num_workers'],
                                  collate_fn=train_collate_func,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=val_test_collate_func,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    return train_dataloader, test_dataloader

def build_model() -> nn.Module:

    model = CRNN(Conf.model_num_classes)
    model = model.to(device=Conf.device)

    return model

def define_loss(model) -> nn.CTCLoss:
    criterion = nn.CTCLoss()
    criterion = criterion.to(device=Conf.device)

    return criterion

def define_optimizer(model) -> optim.Adadelta:
    optimizer = optim.Adadelta(model.parameters(), Conf.mode_config['train']['model_lr'])

    return optimizer

def train(model: nn.Module,
          train_dataloader: DataLoader,
          criterion: nn.CTCLoss,
          optimizer: optim.RMSprop,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:

    batches = len(train_dataloader),
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch+1}]")

    model.train()
    end = time.time()

    for batch_index, (images, target, target_len) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        current_batch_s = images.size(0)
        images = images.to(device=Conf.device, non_blocking=True)
        target = target.to(device=Conf.device, non_blocking=True)
        target_len = target_len.to(device=Conf.device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        with amp.autocast():
            output = model(images)

            output_log_probas = F.log_softmax(output, 2)
            images_lens = torch.LongTensor([output.size(0)] * current_batch_s)
            target_len = torch.flatten(target_len)

            loss = criterion(output_log_probas, target, images_lens, target_len)

        scaler.scale(loss).backward()
        scaler.scale(optimizer)
        scaler.update()

        losses.update(loss.item(), current_batch_s)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % Conf.mode_config['train']["print_frequency"]:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)


def val(model: nn.Module,
        dataloader: DataLoader,
        epoch: int,
        writer: SummaryWriter,
        mode: str) -> [float, float]:

    model.eval()

    total_cor = 0
    total_files = 0

    with torch.no_grad():
        for batch_index, (_, images, target) in enumerate(dataloader):
            total_files += images.size(0)

            images = images.to(device=Conf.device, non_blocking=True)

            with amp.autocast():
                output = model(images)

            output_log_probas = F.log_softmax(output, 2)
            _, predicted_chars = ctc_decoder(output_log_probas, Conf.chars_dict)

            if ''.join(predicted_chars[0]) == target[0].lower():
                total_cor += 1
    acc = (total_cor / total_files) * 100
    print(f'Accuracy: {acc::2f} %')

    if mode == 'valid' or mode == 'test':
        writer.add_scalar(f"{mode}/Acc", acc, epoch + 1)
    else:
        raise ValueError("Unsupported mode!")


    return acc


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT =3
class AverageMeter(object):
    def __int__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt,
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.2f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.2f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.2f}'
        else:
            raise ValueError(f'Invalid type {self.summary_type}')

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self.get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(''.join(entries))

    def get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches//1))
        fmt = '{:' + str(num_digits) + "d}"
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()