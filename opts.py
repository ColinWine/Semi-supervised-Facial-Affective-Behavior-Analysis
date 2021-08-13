import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_label_dir', '-lld', type=str, default=r'I:\ABAW_LMDB\lmdb') #Modify this path to lmdb folder, the lmdb folder should contain '.croped_aligned_jpeg' and '.croped_aligned_mask'
    parser.add_argument('--audio_dir', '-ad', type=str, default=r'I:\ABAW_LMDB\extract_audio') #Modify this path to extracted audios
    parser.add_argument('--model_weight', '-mw', type=str, default=r'L:\ABAW\ABAW2021\experiments\MT_TNT_Pretrain_EMA\pretrain\best.pth') #Modify this path to trained weight

    parser.add_argument('--task', '-t', type=str, default='ALL', help='task, one of [ALL, EX, AU, VA]')
    parser.add_argument('--root', '-r', type=str, default=r'I:\ABAW_LMDB\aff2_processed')
    parser.add_argument('--exp_dir', '-ed', type=str, default='experiments/MT_TNT_Pretrain_EMA')
    parser.add_argument('--cache_dir', '-cd', type=str, default='./data')

    parser.add_argument('--modality', '-md', type=str, default='A;V;')

    model = parser.add_argument_group(title='Model Parameters')
    model.add_argument('--dropout_rate', type=float, default=0.2)
    model.add_argument('--num_channels', type=int, default=4)
    model.add_argument('--use_mask', type=bool, default=True)
    model.add_argument('--model_name', '-mn', type=str, default='tsav')

    training = parser.add_argument_group(title='Training Parameters')
    training.add_argument('--seed', default=123, type=int, help='for reproducibility')
    training.add_argument('--learning_rate', '-lr', default=5e-4, type=float, help='the initial learning rate')
    training.add_argument('--n_warmup_steps', type=int, default=0,
                          help='the number of warmup steps towards the initial lr')
    training.add_argument('--grad_clip', type=float, default=-1,
                          help='clip gradients at this value. it will not be used if the value is lower than 0')
    training.add_argument('--weight_decay', type=float, default=5e-5, help='Strength of weight regularization')
    training.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    training.add_argument('-b', '--batch_size', type=int, default=28, help='minibatch size')
    training.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    training.add_argument('--early_stop_step', type=int, default=10)
    training.add_argument('--save_step', type=int, default=1e3)
    training.add_argument('--eval_step', type=int, default=1e3)
    training.add_argument('--val_num_batch', type=int, default=1e2)
    training.add_argument('--resume', action='store_true')
    training.add_argument('--start_epoch', type=int, default=0)

    dataloader = parser.add_argument_group(title='Dataloader Parameters')
    dataloader.add_argument('--n_frames', '--clip_len', type=int, default=8, help='the number of frames per clip')
    dataloader.add_argument('--dilation', type=int, default=6, help='dilation')
    dataloader.add_argument('--downsample_rate', '-ds', type=int, default=100)
    dataloader.add_argument('--audio_len_secs', '-als', type=int, default=10)
    dataloader.add_argument('--audio_shift_secs', '-ass', type=int, default=5)
    dataloader.add_argument('--n_mels', type=int, default=64)
    args = parser.parse_args()

    return args
