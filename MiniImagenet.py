import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import csv


class MiniImagenet(Dataset):

    def __init__(self, root, train = True, resize=224, startidx=0):

        self.resize = resize
        self.startidx = startidx

        if train == True:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')

        self.data = []
        self.targets = []

        self.img2label = {}
        j = 0
        for mode in ['train', 'val', 'test']:
            csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
            for i, (k, v) in enumerate(csvdata.items()):
                if train == True:
                    self.data.extend(v[:500])
                    self.targets.extend([1*j for i in range(500)])
                else:
                    self.data.extend(v[500:])
                    self.targets.extend([1 * j for i in range(100)])
                self.img2label[k] = j + self.startidx
                j += 1


    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels


    def __getitem__(self, index):

        path = os.path.join(self.path, self.data[index])
        label = self.img2label[self.data[index][:9]]
        image = self.transform(path)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return len(self.data)
