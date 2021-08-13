import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models import TwoStreamAuralVisualSelfCure
from dataloader import Aff2TestDataset
from tqdm import tqdm
import os
import opts

result_path = 'results'


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.indices)

def au_to_str(arr):
    str = "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}".format(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10], arr[11])
    return str

def ex_to_str(arr):
    str = "{:d}".format(arr)
    return str

def va_to_str(v,a):
    str = "{:.3f},{:.3f}".format(v, a)
    return str

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    model_path = opt['model_weight'] # path to the model
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print('cpu selected!')
    # model
    model = TwoStreamAuralVisualSelfCure(num_channels=4)
    modes = model.modes
    # # load the model
    model = model.to(device)
    print('Loading weight from:{}'.format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)

    # load dataset (first time this takes longer)
    dataset = Aff2TestDataset(opt)
    dataset.set_modes(modes)

    # select the frames we want to process (we choose VAL and TEST)
    print('Test set length: ' + str(sum(dataset.test_ids)))
    sampler = SubsetSequentialSampler(np.nonzero(dataset.test_ids)[0])
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=False, drop_last=False)

    output = torch.zeros((len(dataset), 22), dtype=torch.float32)
    #labels = torch.zeros((len(dataset), 17), dtype=torch.float32)

    # run inference
    # takes 5+ hours for test and val on 2080 Ti, with data on ssd
    os.makedirs(result_path, exist_ok=True)
    ex_result_folder = os.path.join(result_path, 'ex')
    au_result_folder = os.path.join(result_path, 'au')
    va_result_folder = os.path.join(result_path, 'va')
    os.makedirs(ex_result_folder, exist_ok=True)
    os.makedirs(au_result_folder, exist_ok=True)
    os.makedirs(va_result_folder, exist_ok=True)
    
    header = {"AU": "AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26", # 0,0,0,0,1,0,0,0
              "VA": "valence,arousal", # 0.602,0.389 or -0.024,0.279
              "EX": "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" # 4
             }
    current_video = None
    bar = tqdm(loader)
    for data in bar:
        # ex_label = data['EX'].float()
        # va_label = data['VA'].float()
        # au_label = data['AU'].float()
        ids = data['Index'].long()
        x = {}
        for mode in modes:
            x[mode] = data[mode].to(device)
        result = model(x)

        if current_video != data['video_id'][0]:
            try:
                ex_writer.close()
                au_writer.close()
                va_writer.close()
            except:
                pass
            current_video = data['video_id'][0]
            bar.set_postfix(current_video=current_video)
            ex_writer = open(os.path.join(ex_result_folder,current_video+'.txt'), "w")
            au_writer = open(os.path.join(au_result_folder,current_video+'.txt'), "w")
            va_writer = open(os.path.join(va_result_folder,current_video+'.txt'), "w")
            ex_writer.write(header['EX'])
            ex_writer.write('\n')
            au_writer.write(header['AU'])
            au_writer.write('\n')
            va_writer.write(header['VA'])
            va_writer.write('\n')

        predict = result.detach().cpu()
        predict_ex = torch.argmax(predict[:, 12:20], dim=1).detach().cpu().numpy()[0]
        pred_v = torch.tanh(predict[:, 20])
        pred_a = torch.tanh(predict[:, 21])
        pred_au = torch.sigmoid(predict[:, :12]).detach().cpu().squeeze().numpy()
        round_au = np.round(pred_au).astype(np.int)
        ex_writer.write(ex_to_str(predict_ex))
        ex_writer.write('\n')
        au_writer.write(au_to_str(round_au))
        au_writer.write('\n')
        va_writer.write(va_to_str(pred_v.item(),pred_a.item()))
        va_writer.write('\n')
        
        output[ids, :] = result.detach().cpu()  # output is EX VA AU
    torch.save({'predictions': output}, os.path.join(result_path, 'inference.pkl'))