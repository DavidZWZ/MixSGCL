import argparse
from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=str, default='rec', help='mode of training')
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--lr', '-l', type=str, default=0.001, help='learing rate')
    parser.add_argument('--weight_decay', '-w', type=str, default=1e-6, help='weight_decay')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='log suffix')
    parser.add_argument('--config_files', type=str, default='config.yaml', help='config files')
    parser.add_argument('--tem', '-t', type=str, default=None, help='temperature')
    
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, gpu=args.gpu, suffix=args.suffix, lr=args.lr, weight_decay=args.weight_decay, tem=args.tem, config_file_list=config_file_list)