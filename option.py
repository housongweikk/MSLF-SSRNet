import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')


parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")

parser.add_argument('--model_name', type=str, default='MSLF', help="model name")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='log\SR_5x5_2x_16\ALL\MSLF\checkpoints\MSLF_5x5_2x_best_model.pth', help="path for pre model ckpt")#更改相应的模型pth文件夹路径
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
# parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--path_test', type=str, default='./Test result/')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=2, type=int, help='Epoch to run [default: 50]')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

parser.add_argument('--alpha', type=float, default=0.7, help='Weight for the loss function')


args = parser.parse_args()



if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1

del args.angRes
