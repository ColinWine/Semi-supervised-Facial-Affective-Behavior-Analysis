import warnings
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models import TwoStreamAuralVisualModel, ImageModel, TwoStreamAuralVisualSelfCure
from dataloader import Aff2CompDataset, SubsetSequentialSampler, Prefetcher
from tqdm import tqdm
import os
import time
from sklearn.metrics import f1_score, accuracy_score
from metrics import AccF1Metric, CCCMetric, MultiLabelAccF1
from collections import defaultdict
import opts
from utils import setup_seed, save_checkpoint, AverageMeter
import random
import logging
from teacher import ramps, losses

warnings.filterwarnings("ignore")

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.4*ramps.sigmoid_rampup(epoch, 15.0)  #consistency_rampup = 7.0

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        param_cpu = param.detach().data.cpu()
        ema_param.data.mul_(alpha).add_(1 - alpha, param_cpu.to(torch.device("cuda:1")))
        del param_cpu

def create_ema_model(net):
    # Network definition
    device = torch.device("cuda:1")
    net_cuda = net.to(device)
    for param in net_cuda.parameters():
        param.detach_()

    return net_cuda

@torch.no_grad()
def evaluate(model, loader, loader_iter, device, num_step=1000):
    model.eval()
    bar = tqdm(range(int(num_step)), desc=f'Validation, {model.task}', colour='green', position=0, leave=False)
    metric_ex = AccF1Metric(ignore_index=7)
    metric_va = CCCMetric(ignore_index=-5.0)
    metric_au = MultiLabelAccF1(ignore_index=-1)
    scores = defaultdict()
    for step in bar:
        t1 = time.time()
        try:
            data = next(loader_iter)
        except StopIteration as e:
            print(e)
            loader_iter = iter(loader)
            break
        t2 = time.time()
        data_time = t2 - t1
        label_ex = data['EX'].long().to(device)
        label_ex[label_ex == -1] = 7
        labels = {
            'VA': data['VA'].float().to(device),
            'AU': data['AU'].float().to(device),
            'EX': label_ex,
        }
        x = {}
        for mode in model.modes:
            x[mode] = data[mode].to(device)
        result,attention_weights_ex = model(x)  # batchx22 12 + 8 + 2
        logits_ex = result[:, 12:20]
        logits_au = result[:, :12]
        logits_va = result[:, 20:22] #tanh??

        pred = torch.argmax(logits_ex, dim=1).detach().cpu().numpy().reshape(-1)
        label = label_ex.detach().cpu().numpy().reshape(-1)

        metric_ex.update(pred, label)
        metric_va.update(y_pred=torch.tanh(logits_va).detach().cpu().numpy(), y_true=labels['VA'].detach().cpu().numpy())
        metric_au.update(y_pred=np.round(torch.sigmoid(logits_au).detach().cpu().numpy()), y_true=labels['AU'].detach().cpu().numpy())
        bar.set_postfix(data_fetch_time=data_time)

    acc_ex, f1_ex = metric_ex.get()
    acc_au, f1_au = metric_au.get()
    scores['EX'] = {'EX:acc': acc_ex, 'f1': f1_ex, 'score': 0.67 * f1_ex + 0.33 * acc_ex}
    scores['AU'] = {'AU:acc': acc_au, 'f1': f1_au, 'score': 0.5 * f1_au + 0.5 * acc_au}
    scores['VA'] = {'VA:ccc_v': metric_va.get()[0],'ccc_a': metric_va.get()[1], 'score': metric_va.get()[2]}
    model.train()
    metric_va.clear()
    metric_au.clear()
    metric_ex.clear()
    return scores, loader_iter


def train_mt(args, model, ema_model, dataset, optimizer, epochs, device):
    early_stopper = EarlyStopper(num_trials=args['early_stop_step'], save_path=f'{args["checkpoint_path"]}/best.pth')
    downsample_rate = args.get('downsample_rate')
    downsample = np.zeros(len(dataset), dtype=int)
    downsample[np.arange(0, len(dataset) - 1, downsample_rate)] = 1

    logging.basicConfig(filename=os.path.join(args['exp_dir'],"log.txt"), level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger()
    consistency_criterion = losses.sigmoid_mse_loss
    for epoch in range(epochs):
        random.shuffle(downsample)
        if epoch < args.get('start_epoch'):
            continue
        val_sampler = SubsetSequentialSampler(np.nonzero(dataset.val_ids*downsample)[0], shuffle=True)
        val_loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=val_sampler, num_workers=0,
                                pin_memory=False,
                                drop_last=True)
        train_sampler = SubsetSequentialSampler(np.nonzero(dataset.train_ids*downsample)[0], shuffle=True)
        train_loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=train_sampler, num_workers=0,
                                pin_memory=False,
                                drop_last=False)
        
        print('Training set length: ' + str(sum(dataset.train_ids*downsample)))
        print('Validation set length: ' + str(sum(dataset.val_ids*downsample)))
        
        val_loader_iter = iter(val_loader)
        bar = tqdm(train_loader, desc=f'Training {model.task}, Epoch:{epoch}', colour='blue', position=0, leave=True)
        logging.info(f'Training {model.task}, Epoch:{epoch}')
        t1 = time.time()
        total_loss_record, ex_loss_record,au_loss_record,va_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        total_con_loss_record, ex_con_loss_record,au_con_loss_record,va_con_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        prefetcher = Prefetcher(bar)
        data = prefetcher.next()
        step = -1
        consistency_weight = get_current_consistency_weight(epoch)
        print('consistency_weight',consistency_weight)
        while data is not None:
            step += 1
            t2 = time.time()
            data_time = t2 - t1
            optimizer.zero_grad()
            label_ex = data['EX'].long().to(device)
            label_ex[label_ex == -1] = 7
            labels = {
                'VA': data['VA'].float().to(device),
                'AU': data['AU'].float().to(device),
                'EX': label_ex,
            }

            # ids = data['Index'].long()
            x = {}
            for mode in ['clip', 'audio_features']:
                x[mode] = data[mode].to(device)
            result,attention_weights = model(x)  # batchx22 12 + 8 + 2

            noise_input = {}
            ema_device = torch.device("cuda:1")
            noise_input['clip'] = data['aug'].to(ema_device)
            noise_input['audio_features'] = data['audio_features'].to(ema_device)
            ema_result,attention_ema = ema_model(noise_input)  # batchx22 12 + 8 + 2

            sup_losses,con_losses = model.get_mt_mt_sc_loss(result, attention_weights, labels, ema_result, attention_ema, consistency_criterion) 
            sup_loss = 3*sup_losses[0] + sup_losses[1] + sup_losses[2]
            con_loss = 3*con_losses[0] + con_losses[1] + con_losses[2] + 12*sup_losses[3]

            ex_loss_record.update(sup_losses[0].item())
            au_loss_record.update(sup_losses[1].item())
            va_loss_record.update(sup_losses[2].item())
            total_loss_record.update(sup_loss.item())

            ex_con_loss_record.update(con_losses[0].item())
            au_con_loss_record.update(con_losses[1].item())
            va_con_loss_record.update(con_losses[2].item())
            total_con_loss_record.update(con_loss.item())

            loss = sup_loss + consistency_weight*con_loss
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, step)

            bar.set_postfix(total = total_loss_record.avg, total_con = total_con_loss_record.avg, ex=ex_loss_record.avg, ex_con=ex_con_loss_record.avg,
            au=au_loss_record.avg, au_con=au_con_loss_record.avg, va=va_loss_record.avg, va_con=va_con_loss_record.avg)
            
            t1 = time.time()
            data = prefetcher.next()
        logging.info(f'Total Loss,{total_loss_record.avg}, Ex:{ex_loss_record.avg}, AU:{au_loss_record.avg}, VA:{va_loss_record.avg}')
        logging.info(f'Total Con Loss,{total_con_loss_record.avg}, Ex Con:{ex_con_loss_record.avg}, AU Con:{au_con_loss_record.avg}, VA Con:{va_con_loss_record.avg}')

        save_checkpoint(state=ema_model.state_dict(), filepath=args["checkpoint_path"], filename='latest.pth')
        scores, val_loader_iter = evaluate(ema_model, val_loader, val_loader_iter, torch.device("cuda:1"),
                                            num_step=int(sum(dataset.val_ids*downsample)/(args['batch_size'])))
        score_str = ''
        if model.task == 'ALL':
            total_score = 0
            for task in ['EX','AU','VA']:
                score_dict = scores[task]
                for k, v in score_dict.items():
                    score_str += f'{k}:{v:.3},'
                total_score = total_score + score_dict["score"]
        else:
            score_dict = scores[model.task]
            for k, v in score_dict.items():
                score_str += f'{k}:{v:.3}, '
            total_score = score_dict["score"]
        print(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')
        logging.info(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')
        if not early_stopper.is_continuable(model, total_score):
            print(f'validation: best score: {early_stopper.best_accuracy}')
            logging.info(f'validation: best score: {early_stopper.best_accuracy}')
            break


def main(args):
    setup_seed(args.get('seed'))
    task = args.get('task')
    print(f'Task: {task}')

    # model
    if opt['model_name'] == 'tsav':
        model = TwoStreamAuralVisualSelfCure(num_channels=args['num_channels'], task=task)
        ema_model = TwoStreamAuralVisualSelfCure(num_channels=args['num_channels'], task=task)
    else:
        model = ImageModel(task)
    modes = model.modes
    device = torch.device("cuda:0")
    model = model.to(device)
    ema_model = create_ema_model(ema_model)

    '''
    args['checkpoint_path'] = os.path.join(args['exp_dir'], 'pretrain')
    if args['resume'] and os.path.exists(f'{args["checkpoint_path"]}/latest.pth'):
        print('Loading weight from:{}'.format(f'{args["checkpoint_path"]}/latest.pth'))
        model.load_state_dict(torch.load(f'{args["checkpoint_path"]}/latest.pth'))
    '''
    args['checkpoint_path'] = os.path.join(args['exp_dir'], 'pretrain')
    if args['resume'] and os.path.exists(f'{args["checkpoint_path"]}/latest.pth.tar'):
        print('Loading weight from:{}'.format(f'{args["checkpoint_path"]}/latest.pth.tar'))
        pretrained_dict = torch.load(f'{args["checkpoint_path"]}/latest.pth.tar')['state_dict']
        pretrained_dict.pop('fc.1.weight')
        pretrained_dict.pop('fc.1.bias')
        model.load_state_dict(pretrained_dict,strict= False)
    
    model.train()
    ema_model.train()


    # load dataset (first time this takes longer)
    dataset = Aff2CompDataset(args)

    dataset.set_modes(modes)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    train_mt(args, model, ema_model, dataset, optimizer, epochs=args['epochs'], device=device)


if __name__ == '__main__':
    opt = opts.parse_opt()
    torch.cuda.set_device(opt.gpu_id)
    opt = vars(opt)
    main(opt)
