import ast
import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from datetime import datetime
from subprocess import call

try:
    from pytorch_quantization.utils import misc, make_path, hook
    HAS_PTQ = True
except ImportError:
    HAS_PTQ = False
    print("[Warning] pytorch_quantization not available, hardware-only mode active")

from quantize import quantize_model, evaluate
from dataset import get_imagenet, get_cifar10, get_cifar100

def parse_args():
    parser = argparse.ArgumentParser()
    # vgg8 (cifar10) software baseline: 89.66%
    # resnet18 (cifar100) software baseline: 75.59%
    # resnet50 (imagenet) software baseline (quantized): 80.012% (79.972%)
    parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
    parser.add_argument('--model', default='vgg8', help='vgg8|DenseNet40|resnet18|resnet50|swin_t')
    parser.add_argument('--data_path', default='/path/to/datasets/', help='path to saved datasets')
    parser.add_argument('--model_path', default='./models/', help='path to saved models')
    parser.add_argument('--test_name', default='test', help='test name')

    parser.add_argument('--mode', default='TensorRT', help='Quantization mode, only TensorRT supported for now')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use. Only single GPU support for now')
    parser.add_argument('--batch_size', type=int, default=400, help='input batch size for inference (default: 64)')
    parser.add_argument('--calib_batch_size', type=int, default=128, help='input batch size for calibrating quantization (default: 128)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

    # Hardware parameters
    parser.add_argument('--hardware', type=int, default=1, help='run hardware inference simulation')
    parser.add_argument('--ppa', type=int, default=1, help='run power, performance, and area analysis (C++)')
    parser.add_argument('--num_batches', type=int, default=-1, help='number of batches to run in hardware inference simulation, -1 for full dataset')
    parser.add_argument('--sub_array', type=str, default="[128, 128]", help='size of subArray (e.g. 128x128)')
    parser.add_argument('--parallel_read', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
    parser.add_argument('--weight_precision', type=int, default=8, help='number of bits to quantize the weights to (integer)')
    parser.add_argument('--input_precision', type=int, default=8, help='number of bits to quantize the inputs to (integer)')
    parser.add_argument('--dac_precision', type=int, default=1, help='DAC precision (e.g. 1-bit)')
    parser.add_argument('--adc_precision', type=int, default=7, help='ADC precision (e.g. 7-bit)')
    parser.add_argument('--bitcell', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
    parser.add_argument('--mem_type', type=str, default='resistive', help='memory cell type: resisitive | capacitive')
    parser.add_argument('--off_state', type=float, default=6e-3, help='device off state (conductance (S) or capacitance (F))')
    parser.add_argument('--on_state', type=float, default=6e-13*17, help='device on conductance')
    parser.add_argument('--mem_states_file', type=str, default="mem_states.csv", help='path to .csv file containing mean and std for each memory state')
    parser.add_argument('--vdd', type=float, default=1, help='supply voltage')
    parser.add_argument('--read_noise', type=float, default=0.0, help='std of conductance read noise')
    parser.add_argument('--output_noise', type=float, default=0.0, help='std of output voltage noise')
    parser.add_argument('--output_noise_file', type=str, default="", help='path and file to saved output std')

    parser.add_argument('--t', type=float, default=1, help='retention time')
    parser.add_argument('--v', type=float, default=0.00, help='drift coefficient')
    parser.add_argument('--v_list', type=str, default="[0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]", help='drift coefficient')
    parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
    parser.add_argument('--target', type=float, default=0.0, help='drift target for fixed-direction drift, range 0-1')

    parser.add_argument('--rate_stuck_0', type=float, default=0.00, help='rate of cells stuck at 0')
    parser.add_argument('--rate_stuck_1', type=float, default=0.00, help='rate of cells stuck at 1')

    parser.add_argument("--input_calib_method",  type=str, default='max', help='histogram or max')
    parser.add_argument("--weight_calib_method", type=str, default='max', help='histogram or max')
    parser.add_argument("--adc_calib_method",    type=str, default='max', help='histogram or max')
    parser.add_argument("--input_axis", type=int,  default=None)
    parser.add_argument("--weight_axis", type=int, default=0)
    parser.add_argument("--adc_axis",    type=int, default=None)
    parser.add_argument("--adc_quant_method", type=str, default='scale')
    parser.add_argument("--optimize_adc", type=int, default=0)
    parser.add_argument("--adc_enable", type=int, default=1)

    args = parser.parse_args()
    args.sub_array = ast.literal_eval(args.sub_array)
    args.v_list = ast.literal_eval(args.v_list)
    return args


def main():

    args = parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # ── Logging setup: use PTQ utilities if available, else plain fallback ──
    if HAS_PTQ:
        args.logdir = make_path.makepath(
            args,
            ['log_interval', 'test_interval', 'logdir', 'epochs',
             'gpu', 'ngpu', 'debug', 'data_path', 'model_path']
        )
        misc.logger.init(args.logdir, 'test_log_' + current_time)
        args.logger = misc.logger.info
        misc.ensure_dir(args.logdir)
    else:
        args.logdir = os.path.join('logs', current_time)
        os.makedirs(args.logdir, exist_ok=True)
        args.logger = print

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================\n")

    print(f"Running in-memory computing functional simulation of {args.model} on {args.dataset} dataset...")
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    args.data_path += args.dataset + '/'

    if args.dataset == 'imagenet':
        data_loader_quant = get_imagenet(args.calib_batch_size, args.data_path, train=True,  val=False, sample=True, model=args.model)
        data_loader_test  = get_imagenet(args.batch_size,       args.data_path, train=False, val=True,  sample=True, model=args.model)
    elif args.dataset == 'cifar100':
        data_loader_quant = get_cifar100(args.calib_batch_size, args.data_path, train=True,  val=False)
        data_loader_test  = get_cifar100(args.batch_size,       args.data_path, train=False, val=True)
    elif args.dataset == 'cifar10':
        data_loader_quant = get_cifar10(args.calib_batch_size, args.data_path, train=True,  val=False)
        data_loader_test  = get_cifar10(args.batch_size,       args.data_path, train=False, val=True)

    if args.num_batches == -1:
        args.num_batches = len(data_loader_test)

    # ── Hook / record setup: only available with PTQ ──
    if HAS_PTQ:
        hook.make_records(args)
        args.hook = True
    else:
        args.hook = False
    args.write_network = True

    model = quantize_model(args, criterion, data_loader_quant, data_loader_test)

    args.logger("Running inference with detailed hardware simulation...")

    model = model.to(device)

    accuracy = evaluate(model, args, criterion, data_loader_test,
                        num_batches=args.num_batches, print_freq=1)

    if args.ppa:
        print(" --- Hardware Properties --- ")
        print("subArray size:  ", args.sub_array)
        print("parallel read:  ", args.parallel_read)
        print("ADC precision:  ", args.adc_precision)
        print("cell precision: ", args.bitcell)
        print("on/off ratio:   ", args.on_state / args.off_state)

        trace_script = f'./layer_record_{args.model}/trace_command.sh'
        if os.path.exists(trace_script):
            call(["/bin/bash", trace_script])
        else:
            print(f"[Warning] trace_command.sh not found at {trace_script}, skipping PPA call.")

if __name__ == '__main__':
    main()