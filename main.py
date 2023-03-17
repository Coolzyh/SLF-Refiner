import argparse
import os
from runner.runner_MTAE import runner_MTAE
from runner.runner_SLF_Refiner import runner_SLF_Refiner
from runner.runner_CNN_Attn import runner_CNN_Attn


def arg_parser():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--run', type=str, default='./results/', help='The runner to execute')
    parser.add_argument('--model_name', type=str, default='SLF_Refiner', help='Select model to run')

    parser.add_argument('--M', type=int, default=4, help='The number of RF nodes')
    parser.add_argument('--P', type=int, default=6, help='Each node has P measurement positions')
    parser.add_argument('--N', type=int, default=12, help='The number of links: M*(M-1)')
    parser.add_argument('--K0', type=int, default=40, help='SLF image dimension K[0]')
    parser.add_argument('--K1', type=int, default=40, help='SLF image dimension K[1]')
    # MTAE parameters
    parser.add_argument('--lambda1_MTAE', type=float, default=100, help='MTAE: loss weight for RSS reconstruction')
    parser.add_argument('--lambda2_MTAE', type=float, default=200, help='MTAE: loss weight for SLF estimation')
    parser.add_argument('--lambda3_MTAE', type=float, default=1, help='MTAE: loss weight for ab parameters estimation')
    parser.add_argument('--lambda4_MTAE', type=float, default=1, help='MTAE: loss weight for noise level prediction')
    parser.add_argument('--n_epochs_MTAE', type=int, default=60)
    # CNN-Attn parameters
    parser.add_argument('--n_epochs_CNN_Attn', type=int, default=60)
    # SLF Refiner paramters
    parser.add_argument('--n_epochs_Refiner', type=int, default=60)

    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='how many epochs to save model before logging training status')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--pretrain', type=bool, default=True, metavar='P',
                        help='whether exists pretrained model (default: True)')
    args = parser.parse_args()
    args.N = int(args.M * (args.M - 1))
    return args


def main():
    args = arg_parser()
    print(args.model_name)
    if not os.path.exists(args.run):
        os.makedirs(args.run)
    if args.model_name == 'SLF_Refiner':
        runner = runner_SLF_Refiner(args)
    elif args.model_name == 'CNN_Attn':
        runner = runner_CNN_Attn(args)
    elif args.model_name == 'MTAE':
        runner = runner_MTAE(args)
    else:
        raise NotImplementedError('Model {} not understood.'.format(args.model_name))

    if not args.pretrain:
        runner.train_save()
        # runner.train_save_with_cnn_attn()
        # runner.train_save_with_mtae()
    else:
        runner.test_model('all')
        runner.test_model('low')
        runner.test_model('mid')
        runner.test_model('high')
        # runner.test_model_with_cnn_attn('all')
        # runner.test_model_with_cnn_attn('low')
        # runner.test_model_with_cnn_attn('mid')
        # runner.test_model_with_cnn_attn('high')
        # runner.test_model_with_mtae('all')
        # runner.test_model_with_mtae('low')
        # runner.test_model_with_mtae('mid')
        # runner.test_model_with_mtae('high')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()