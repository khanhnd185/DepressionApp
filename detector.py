import os, sys
import torch
import opensmile
import numpy as np
import torch.nn as nn

from model import SCUnimodalTransformer
from utils import load_state_dict, transform, print_eval_info
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, f1_score

# GUI class for the chat
class Detector:
    # constructor method
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depmbt = SCUnimodalTransformer(25 , 256)
        self.header = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )

        self.depmbt = load_state_dict(self.depmbt, 'model/net.pth')
        self.header = load_state_dict(self.header, 'model/classifier.pth')

        self.depmbt = self.depmbt.to(self.device)
        self.header = self.header.to(self.device)

        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02
                                   , feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    def validate(self, loader):
        self.depmbt.eval()
        self.header.eval()

        all_y = None
        all_labels = None
        for batch_idx, data in enumerate(loader):
            feature_audio, feature_video, mask, labels = data
            with torch.no_grad():
                feature_audio = feature_audio.to(self.device)
                feature_video = feature_video.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)

                features = self.depmbt.encode(feature_audio, feature_video, mask)
                y = self.header(features.detach())

                if all_y == None:
                    all_y = y.clone()
                    all_labels = labels.clone()
                else:
                    all_y = torch.cat((all_y, y), 0)
                    all_labels = torch.cat((all_labels, labels), 0)

        all_y = all_y.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        all_labels, all_y = transform(all_labels, all_y)

        f1 = f1_score(all_labels, all_y, average='weighted')
        r = recall_score(all_labels, all_y, average='weighted')
        p = precision_score(all_labels, all_y, average='weighted')
        acc = accuracy_score(all_labels, all_y)
        cm = confusion_matrix(all_labels, all_y)
        eval_return = (0, f1, r, p, acc, cm)

        print_eval_info("", eval_return)

    def inference(self, filename):
        with torch.no_grad():
            x = self.smile.process_file(filename)
            x = x.to_numpy()
            leng = x.shape[0]
            leng = (leng // 100) * 100
            x = x[:leng, :]
            x = x.reshape(leng//100,100,25)
            x = np.sum(x, axis=1)
            x = torch.Tensor(x)
            x = x.unsqueeze(0)
            m = torch.ones(1, x.shape[1])
            
            self.depmbt.eval()
            self.header.eval()
            x = x.to(self.device)
            m = m.to(self.device)
            features = self.depmbt.encode(x, x, m)
            y = self.header(features.detach())

            y = y.squeeze(0)
        out = np.argmax(y.cpu().numpy(), axis=0)

        return out   




def main():
    import pickle
    from torch.utils.data import DataLoader
    from dataset import AudioDVlog, collate_fn

    # with open('../../../Data/DVlog/data_D-Vlog.csv', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()[1:]
    # with open('dvlog.pickle', 'rb') as handle:
    #     dataset = pickle.load(handle)

    #trainset = AudioDVlog(dataset, lines, 'train')
    #validset = AudioDVlog(dataset, lines, 'valid')
    #testset = AudioDVlog(dataset, lines, 'test')
    #trainldr = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True, num_workers=0)
    #validldr = DataLoader(validset, batch_size=32, collate_fn=collate_fn, shuffle=False, num_workers=0)
    #testldr = DataLoader(testset, batch_size=32, collate_fn=collate_fn, shuffle=False, num_workers=0)

    detector = Detector()
    # detector.validate(trainldr)
    # detector.validate(validldr)
    # detector.validate(testldr)

    out = detector.inference('../../../Data/DVlog/wav/ZKJjVV7U2uo.mp4.wav')
    print(out)

if __name__=="__main__":
    main()

