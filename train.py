import json

import os
import torch
import copy
import argparse
import importlib
import numpy as np
import torch.nn as nn
from pathlib import Path

from munch import munchify
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CoNAL
from arguments import get_task_parser, add_train_args, add_model_args

root_dir = Path(os.path.dirname(__file__))


class Train:
    def __init__(self, ):

        print('Loading configurations...')
        self.status = False
        config_file = root_dir / 'conf/music.json'
        print(__file__)
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.args = munchify(config)
        # python train.py --task music --train_data ./data/Music/train --valid_data ./data/Music/valid
        # --device cpu --input_dim 124 --n_class 11 --n_annotator 44 --epochs 800 --batch_size 128
        # train_dir = 'E:\model\CoNAL-pytorch/data/Music/train'
        # valid_dir = 'E:\model\CoNAL-pytorch/data/Music/valid'
        # # Read task argument first, and determine the other arguments
        # task_parser = get_task_parser()
        #
        # task_name = task_parser.parse_known_args()[0].task
        self.current_epoch = 0
        self.pLog = []

        self.task_module = importlib.import_module(f'tasks.{self.args.task}')
        self.task_dataset = getattr(self.task_module, 'Dataset')

        # parser = argparse.ArgumentParser()
        # add_train_args(parser)
        # add_model_args(parser)
        # getattr(self.task_module, 'add_task_args')(parser)
        # self.args = parser.parse_args()

        # Seed settings
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # unzip compressed file
        print('Loading train dataset...')
        self.train_dataset = self.task_dataset(self.args, root_dir / self.args.train_data, is_train=True)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

        print('Loading validation dataset...')
        self.valid_dataset = self.task_dataset(self.args, root_dir / self.args.valid_data)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.args.batch_size)

    def getProgress(self):
        log = copy.deepcopy(self.pLog)
        self.pLog.clear()
        return self.status, self.current_epoch, self.args.epochs, log

    def train(self, ):
        self.status = True
        args = self.args
        print('Building model...')
        classifier = getattr(self.task_module, 'Classifier')(args)
        model = CoNAL(
            args.input_dim,
            args.n_class,
            args.n_annotator,
            classifier,
            annotator_dim=args.n_annotator,
            embedding_dim=args.emb_dim
        )
        model = model.to(args.device)

        # Ignore annotators labeling which is -1
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        optimizer = Adam(model.parameters(), lr=args.lr)

        print('Start training!')
        best_accuracy = 0
        writer = SummaryWriter(args.log_dir)
        for epoch in range(args.epochs):
            self.current_epoch = epoch
            train_loss = 0.0
            train_correct = 0
            model.train()
            for x, y, annotation in self.train_loader:
                model.zero_grad()

                # Annotator embedding matrix (in this case, just a identity matrix)
                annotator = torch.eye(args.n_annotator)

                # Move the parameters to device given by argument
                x, y, annotation, annotator = x.to(args.device), y.to(args.device), annotation.to(
                    args.device), annotator.to(args.device)
                ann_out, cls_out = model(x, annotator)
                # print(ann_out, cls_out)

                # Calculate loss of annotators' labeling
                ann_out = torch.reshape(ann_out, (-1, args.n_class))
                annotation = annotation.view(-1)  # [:ann_out.shape[0]]
                loss = criterion(ann_out, annotation)

                # Regularization term
                confusion_matrices = model.noise_adaptation_layer
                matrices = confusion_matrices.local_confusion_matrices - confusion_matrices.global_confusion_matrix
                for matrix in matrices:
                    loss -= args.scale * torch.linalg.norm(matrix)

                # Update model weight using gradient descent
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate classifier accuracy
                pred = torch.argmax(cls_out, dim=1)
                # print(pred)
                train_correct += torch.sum(torch.eq(pred, y)).item()

            # Validation
            with torch.no_grad():
                valid_correct = 0
                model.eval()
                for x, y in self.valid_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    pred = model(x)
                    pred = torch.argmax(pred, dim=1)
                    valid_correct += torch.sum(torch.eq(pred, y)).item()

            self.pLog.append(
                f'Epoch: {(epoch + 1):4d} | '
                f'Train Loss: {train_loss:.3f} | '
                f'Train Accuracy: {(train_correct / len(self.train_dataset)):.2f} | '
                f'Valid Accuracy: {(valid_correct / len(self.valid_dataset)):.2f}'
            )
            print(
                f'Epoch: {(epoch + 1):4d} | '
                f'Train Loss: {train_loss:.3f} | '
                f'Train Accuracy: {(train_correct / len(self.train_dataset)):.2f} | '
                f'Valid Accuracy: {(valid_correct / len(self.valid_dataset)):.2f}'
            )

            # Save tensorboard log
            if epoch % args.log_interval == 0:
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('train_accuracy', train_correct / len(self.train_dataset), epoch)
                writer.add_scalar('valid_accuracy', valid_correct / len(self.valid_dataset), epoch)

            # Save the model with highest accuracy on validation set
            if best_accuracy < valid_correct:
                best_accuracy = valid_correct
                checkpoint_dir = Path(args.save_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'auxiliary_network': model.auxiliary_network.state_dict(),
                    'noise_adaptation_layer': model.noise_adaptation_layer.state_dict(),
                    'classifier': model.classifier.state_dict()
                }, checkpoint_dir / 'best_model.pth')

                with open(checkpoint_dir / 'args.json', 'w') as f:
                    json.dump(args.__dict__, f, indent=2)

        self.status = False

if __name__ == "__main__":
    t = Train('music')
    t.train()