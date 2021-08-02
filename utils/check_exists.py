import os
import time
from load_data.data_loader import save_file_path


def check_path(args):
    anno_path = 'anno/'
    assert (not args.train_dir) == (not args.val_dir)
    assert not (args.train_dir and args.data_dir)

    if not os.path.exists(args.exp):
        os.mkdir(args.exp)

    if not os.path.exists('../anno/'):
        os.mkdir('../anno/')

    if args.anno:
        anno_path = ''.join(['anno/', args.anno, '/'])
        if not os.path.exists(anno_path):
            os.mkdir(anno_path)

    if not os.path.exists('../conf_mat_pic/'):
        os.mkdir('../conf_mat_pic/')

    if args.train_dir and args.val_dir:
        if not (os.path.exists(anno_path + 'train.txt') and os.path.exists(anno_path + 'val.txt')):
            save_file_path(args, phase='train')
            save_file_path(args, phase='val')

    args.anno = anno_path
    if args.data_dir:
        save_file_path(args, split=True, test_size=0.2)

    localtime = time.strftime("%b_%d_%H_%M_%S_%Y", time.localtime())
    log_exp = args.exp + localtime + '/'
    if not os.path.exists(log_exp):
        os.mkdir(log_exp)
    args.exp = log_exp

    print(args)

    with open(log_exp + 'options.txt', 'w') as f:
        f.write(str(args))




