import matplotlib.pyplot as plt
import numpy as np
import math
import random, sys, os
import asyncio, time
import threading
from datetime import datetime
import argparse
import tqdm
import copy
import time

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

import utils
import compressors


#====================================================================================

class NNConfiguration: pass
class WorkersConfiguration: pass

#====================================================================================

print_lock = threading.Lock()

def dbgprint(wcfg, *args):
    printing_dbg = True
    if printing_dbg == True:
        print_lock.acquire()
        print(f"Worker {wcfg.worker_id}/{wcfg.total_workers}:", *args, flush = True)
        print_lock.release()

def rootprint(*args):
    print_lock.acquire()
    print(f"Master: ", *args, flush = True)
    print_lock.release()

def getAccuracy(model, trainset, batch_size, device):
    avg_accuracy = 0

    dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)           
    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        logits = model(inputs)
        avg_accuracy += (logits.data.argmax(1) == outputs).sum().item()

    avg_accuracy /= len(trainset)
    model.train(prev_train_mode)

    return avg_accuracy

def getLossAndGradNorm(model, trainset, batch_size, device):
    total_loss = 0
    grad_norm = 0

    one_inv_samples = torch.Tensor([1.0/len(trainset)]).to(device)

    dataloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=False,  
        drop_last=False,  
        pin_memory=False,
    )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)

    for p in model.parameters():
        p.grad = None

    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             # move to device

        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network

        loss = one_inv_samples * F.cross_entropy(logits, outputs, reduction='sum')        # compute objective
        loss.backward()                                                                     # compute the gradient (backward-pass)
        total_loss += loss
        
    for p in model.parameters(): 
        grad_norm += torch.norm(p.grad.data.flatten(0))**2
        p.grad = None

    model.train(prev_train_mode)
    return total_loss.item(), grad_norm

#======================================================================================================================================

class WorkerThreadMarina(threading.Thread):
    def __init__(self, wcfg, ncfg):
        threading.Thread.__init__(self)
        self.wcfg = wcfg
        self.ncfg = ncfg

        model = utils.getModel(self.ncfg.model_name, len(self.wcfg.classes))
        model = model.to(wcfg.device) 
        utils.setupAllParamsRandomly(model)
        self.wcfg.model = model
        
        self.iteration = 0

    # compress the gradient related signal
    def compressSignal(self, signal):
        transit_bits = 0
        signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))
        
        signal_flatten = torch.zeros(self.ncfg.D).to(self.wcfg.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = signal[t].flatten(0)
            signal_offset += offset

        signal_flatten = self.wcfg.compressor.compressVector(signal_flatten, self.iteration)
        transit_bits += self.wcfg.compressor.last_need_to_send_advance

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed, transit_bits
    
    def compressSignal_layerwise(self, signal):
        transit_bits = 0
        signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))
        
        signal_flatten = torch.zeros(self.ncfg.D).to(self.wcfg.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = self.wcfg.compressor.compressVector(signal[t].flatten(0), self.iteration)
            transit_bits += self.wcfg.compressor.last_need_to_send_advance
            signal_offset += offset

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed, transit_bits

    # different optimizer to update the local model with g_i
    def updateLocalModel(self, g_i):
        grads = copy.deepcopy(g_i)
        iteration = self.iteration + 1  # add 1 is to make sure nonzero denominator in adam calculation
        
        # learning rate decay
        if iteration < int(self.ncfg.KMax/2):
            lr_decay = 1.0
        elif iteration < int(3*self.ncfg.KMax/4):
            lr_decay = 0.1
        else:
            lr_decay = 0.01
        
        # update the model parameter with recieved aggregated gradient
        for i, param in enumerate(self.wcfg.model.parameters()): 
            grad = grads[i]  # recieve the aggregated (averaged) gradient

            # weight decay
            grad = grad.add(param.data, alpha=self.ncfg.weight_decay)

            # SGD calculation
            if self.ncfg.optim == 'sgd':
                param.data.add_(grad, alpha=-self.ncfg.gamma * lr_decay)
            # SGD+momentum calculation
            elif self.ncfg.optim == 'sgdm':
                buf = self.wcfg.momentum_buffer_list[i]
                buf.mul_(self.ncfg.momentum).add_(grad, alpha=1 - self.ncfg.dampening)
                grad = buf
                param.data.add_(grad, alpha=-self.ncfg.gamma * lr_decay)
            # adam calculation
            elif self.ncfg.optim == 'adam':
                exp_avg = self.wcfg.exp_avgs[i]
                exp_avg_sq = self.wcfg.exp_avg_sqs[i]

                bias_correction1 = 1 - self.ncfg.beta1 ** iteration
                bias_correction2 = 1 - self.ncfg.beta2 ** iteration
                exp_avg.mul_(self.ncfg.beta1).add_(grad, alpha=1 - self.ncfg.beta1)
                exp_avg_sq.mul_(self.ncfg.beta2).addcmul_(grad, grad.conj(), value=1 - self.ncfg.beta2)
                torch.maximum(self.wcfg.max_exp_avg_sqs[i], exp_avg_sq, out=self.wcfg.max_exp_avg_sqs[i])
                denom = (self.wcfg.max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(self.ncfg.eps)

                step_size = self.ncfg.gamma * lr_decay / bias_correction1
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
            # one bit adam calculation
            elif self.ncfg.optim == 'onebitadam':
                exp_avg = self.wcfg.exp_avgs[i]
                exp_avg_sq = self.wcfg.exp_avg_sqs[i]

                bias_correction1 = 1 - self.ncfg.beta1 ** iteration
                bias_correction2 = 1 - self.ncfg.beta2 ** iteration
                exp_avg.mul_(self.ncfg.beta1).add_(grad, alpha=1 - self.ncfg.beta1)
                if iteration < self.ncfg.freeze_iteration:
                    exp_avg_sq.mul_(self.ncfg.beta2).addcmul_(grad, grad.conj(), value=1 - self.ncfg.beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.ncfg.eps)
                
                step_size = self.ncfg.gamma * lr_decay / bias_correction1
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                exit('unknown optimizer: {}'.format(self.ncfg.optim))
 
    # run local training
    def run(self):
        global transfered_bits_by_node
        global fi_grad_calcs_by_node

        dbgprint(self.wcfg, f"START WORKER. IT USES DEVICE", self.wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ", len(self.wcfg.train_set))

        data_iter = iter(self.wcfg.train_loader)
        while True:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.wcfg.train_loader)
                data = next(data_iter)
                
            self.wcfg.input_cmd_ready.acquire()
            self.iteration = self.wcfg.input_iteration_for_cmd

            # in exit state, the worker finish training and exit
            if self.wcfg.cmd == "exit":
                self.wcfg.output_of_cmd = ""
                self.wcfg.cmd_output_ready.release()
                break
            
            # in run state, recieve server's signal and update local model
            if self.wcfg.cmd == "run":
                # recieve server's signal
                signal_recieved = []
                for i, param in enumerate(self.wcfg.model.parameters()):
                    signal_recieved.append(self.wcfg.input_for_cmd[i].clone())
                
                if self.wcfg.compress_strategy in ['markov', 'hybrid1', 'hybrid2', 'hybrid3', 'hybrid4']:
                    self.wcfg.g_in_i = utils.add_params(self.wcfg.g_in_i, signal_recieved)
                    grad = copy.deepcopy(self.wcfg.g_in_i)
                elif self.wcfg.compress_strategy in ['errorFeedback','efhalf']:
                    grad = signal_recieved
                elif self.wcfg.compress_strategy == 'naive':
                    grad = signal_recieved
                else:
                    exit('unknown compression strategy: {}'.format(self.wcfg.compress_strategy))

                # update local model using the recieved (arggregated) gradient
                self.updateLocalModel(grad)
                
            # one step batch training
            prev_train_mode = torch.is_grad_enabled()
            self.wcfg.model.train(True)
            
            inputs, outputs = data
            inputs, outputs = inputs.to(self.wcfg.device), outputs.to(self.wcfg.device)  
            logits = self.wcfg.model(inputs) 
            loss = F.cross_entropy(logits, outputs, reduction='mean')
            
            loss.backward()  
            
            # in init state, the worker initialize local gradient related sequences
            if self.wcfg.cmd == 'init':
                self.wcfg.g_in_i = []
                self.wcfg.g_out_i = []
                self.wcfg.error = []
                
                self.wcfg.momentum_buffer_list = []
                self.wcfg.exp_avgs = []
                self.wcfg.exp_avg_sqs = []
                self.wcfg.max_exp_avg_sqs = []
                    
                for p in self.wcfg.model.parameters(): 
                    self.wcfg.g_in_i.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    self.wcfg.g_out_i.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    self.wcfg.error.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    
                    self.wcfg.momentum_buffer_list.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    self.wcfg.exp_avgs.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    self.wcfg.exp_avg_sqs.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
                    self.wcfg.max_exp_avg_sqs.append(torch.zeros_like(p.grad.data.detach().clone(), requires_grad=False))
    
            # obtain local gradient after one batch
            g_out_next = []
            for p in self.wcfg.model.parameters():
                g_out_next.append(p.grad.data.detach().clone())
                p.grad = None
                
            self.wcfg.model.train(prev_train_mode)
            
            # compress local signal
            if self.wcfg.compress_strategy == 'markov':
                tmp = utils.sub_params(g_out_next, self.wcfg.g_out_i)
                
                if self.wcfg.layerwise == 'flatten':
                    signal_sent, transit_bits = self.compressSignal(tmp)
                elif self.wcfg.layerwise == 'layerwise':
                    signal_sent, transit_bits = self.compressSignal_layerwise(tmp)
                else:
                    exit(f'unknown layerwise compress {self.wcfg.layerwise}')
                
                self.wcfg.g_out_i = utils.add_params(self.wcfg.g_out_i, signal_sent)
            elif self.wcfg.compress_strategy == 'errorFeedback':
                tmp = utils.add_params(g_out_next, self.wcfg.error)
                
                if self.wcfg.layerwise == 'flatten':
                    signal_sent, transit_bits = self.compressSignal(tmp)
                elif self.wcfg.layerwise == 'layerwise':
                    signal_sent, transit_bits = self.compressSignal_layerwise(tmp)
                else:
                    exit(f'unknown layerwise compress {self.wcfg.layerwise}')
                self.wcfg.error = utils.sub_params(tmp, signal_sent)
            elif self.wcfg.compress_strategy == 'naive':
                if self.wcfg.layerwise == 'flatten':
                    signal_sent, transit_bits = self.compressSignal(g_out_next)
                elif self.wcfg.layerwise == 'layerwise':
                    signal_sent, transit_bits = self.compressSignal_layerwise(g_out_next)
                else:
                    exit(f'unknown layerwise compress {self.wcfg.layerwise}')
            else:
                exit('unknown compression strategy: {}'.format(self.wcfg.compress_strategy))
                            
            # broadcast the signal
            self.wcfg.output_of_cmd = copy.deepcopy(signal_sent)
            self.wcfg.cmd_output_ready.release()
            
            # calculate the communication cost when recieving and sending signals
            transfered_bits_by_node[self.wcfg.worker_id, self.iteration] = 2 * transit_bits
            fi_grad_calcs_by_node[self.wcfg.worker_id, self.iteration] = self.ncfg.batch_size
            
        dbgprint(self.wcfg, f"END")

#======================================================================================================================================

def main():
    global transfered_bits_by_node
    global fi_grad_calcs_by_node

    #=======================================================================================================
    # parse arguments
    
    parser = argparse.ArgumentParser(description='run top-k algorithm')
    parser.add_argument('--layerwise', action='store', dest='layerwise', type=str, default='flatten', help='layerwise')
    parser.add_argument('--compressor', action='store', dest='compressor', type=str, default='topk', help='compressor method')
    parser.add_argument('--compress_strategy', action='store', dest='compress_strategy', type=str, default='markov', help='compressor strategy')
    parser.add_argument('--cuda', action='store', dest='cuda', type=str, default='0', help='cuda card')
    parser.add_argument('--optim', action='store', dest='optim', type=str, default='sgd', help='Optimizer')
    parser.add_argument('--momentum', action='store', dest='momentum', type=float, default=0.9, help='momentum coefficient')
    parser.add_argument('--dampening', action='store', dest='dampening', type=float, default=0.0, help='dampening coefficient')
    parser.add_argument('--beta1', action='store', dest='beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', action='store', dest='beta2', type=float, default=0.99, help='beta2 for adam')
    parser.add_argument('--weight_decay', action='store', dest='weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--freeze_epoch', action='store', dest='freeze_epoch', type=int, default=13, help='one bit adam freeze v after this epoch')
    parser.add_argument('--eps', action='store', dest='eps', type=float, default=1e-8, help='eps for adam')

    parser.add_argument('--epoch', action='store', dest='epoch', type=int, default=200, help='number of epoch')
    parser.add_argument('--k', action='store', dest='k', type=float, default=0.01, help='sparcification parameter')
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=5, help='number of workers that will be used')
    parser.add_argument('--lr', action='store', dest='lr', type=float, default=0.1, help='stepsize')
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=128, help='batch size per worker and for GPU')
    
    parser.add_argument('--model', action='store', dest='model', type=str, default='resnet18', help='name of NN architechture')
    parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='CIFAR10', help='name of dataset')
    parser.add_argument('--out_dir', action='store', dest='out_dir', type=str, default='rebuttal_backup4', help='dir to output results')
    
    args = parser.parse_args() 
    
    utils.printTorchInfo()

    cudas = args.cuda.split(',')  
    available_devices = [torch.device(f'cuda:{i}') for i in cudas]
    master_device = available_devices[0]

    random.seed(1)
    torch.manual_seed(1)        # Set the random seed so things involved torch.randn are predictable/repetable 
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    #=======================================================================================================
    # load data
    
    splitted_dataset = utils.getSplitImageDatasets(args.dataset, args.batch_size, args.num_workers)
    train_sets, train_set_full, test_set, train_loaders, test_loaders, classes = splitted_dataset
    
    args.max_iteration = int(args.epoch * len(train_set_full) / (args.batch_size * args.num_workers))
    args.verbose_step = int(args.max_iteration / args.epoch)  # number of iteration for verbose log

    #=======================================================================================================
    # set up configuration for NN
    
    nn_config = NNConfiguration()
    nn_config.dataset = args.dataset                # dataset: CIFAR10
    nn_config.model_name = args.model               # NN architecture: resnet18
    nn_config.kWorkers = args.num_workers           # Number of workers to train model
    nn_config.topK = args.k                         # Number of paramter to send for top K compressor
    nn_config.freeze_iteration = int(args.freeze_epoch * len(train_set_full) / (args.batch_size * args.num_workers))
    # nn_config.freeze_iteration=13

    nn_config.batch_size = args.batch_size
    nn_config.KMax = args.max_iteration
    nn_config.optim = args.optim 
    nn_config.gamma = args.lr
    nn_config.momentum = args.momentum
    nn_config.dampening = args.dampening
    nn_config.weight_decay = args.weight_decay
    nn_config.beta1 = args.beta1
    nn_config.beta2 = args.beta2
    nn_config.eps = args.eps

    #=======================================================================================================
    # load NN model and print out architecture info
    
    print(f"Start training {nn_config.model_name}@{nn_config.dataset} for K={nn_config.KMax} iteration", available_devices)
    master_model = utils.getModel(nn_config.model_name, len(classes))
    master_model = master_model.to(master_device)
    utils.setupAllParamsRandomly(master_model)

    utils.printLayersInfo(master_model, nn_config.model_name)
    nn_config.D = utils.numberOfParams(master_model)
    
    #=======================================================================================================
    # statistics/metrics logged during training
    
    transfered_bits_by_node = np.zeros((nn_config.kWorkers, nn_config.KMax)) # Transfered bits
    fi_grad_calcs_by_node   = np.zeros((nn_config.kWorkers, nn_config.KMax)) # Evaluate number gradients for fi
    train_loss              = []
    test_loss               = []
    train_acc               = []
    test_acc                = []
    fn_train_loss_grad_norm = []
    fn_test_loss_grad_norm  = []
    eval_iteration = []

    #=======================================================================================================
    # set up worker configuration and launch

    worker_tasks = []                           # Worker tasks
    worker_cfgs = []                            # Worker configurations
    for i in range(nn_config.kWorkers):
        worker_cfgs.append(WorkersConfiguration())
        worker_cfgs[-1].worker_id = i
        worker_cfgs[-1].total_workers = nn_config.kWorkers
        worker_cfgs[-1].train_set = train_sets[i]
        worker_cfgs[-1].test_set = test_set
        worker_cfgs[-1].train_set_full = train_set_full
        worker_cfgs[-1].classes = classes

        worker_cfgs[-1].train_loader = train_loaders[i]
        worker_cfgs[-1].test_loader = test_loaders[i]
        worker_cfgs[-1].device = available_devices[i % len(available_devices)]
        
        worker_cfgs[-1].compressor = compressors.Compressor()
        if args.compressor == 'identical':
            worker_cfgs[-1].compressor.makeIdenticalCompressor() 
        elif args.compressor == 'topk':
            worker_cfgs[-1].compressor.makeTopKCompressor(nn_config.topK) 
        elif args.compressor == 'sign':
            worker_cfgs[-1].compressor.makeSignCompressor()
        elif args.compressor == 'onebitsign':
            worker_cfgs[-1].compressor.makeOneBitSignCompressor(int(nn_config.freeze_iteration))
        else:
            exit('unknown compressor: {}'.format(args.compressor))
            
        worker_cfgs[-1].compress_strategy = args.compress_strategy
        worker_cfgs[-1].layerwise = args.layerwise

        worker_cfgs[-1].input_cmd_ready  = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd_output_ready = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd = "init"
        worker_cfgs[-1].input_for_cmd = ""
        worker_cfgs[-1].input_iteration_for_cmd = 0
        worker_cfgs[-1].output_of_cmd = ""
        
        worker_tasks.append(WorkerThreadMarina(worker_cfgs[-1], nn_config))

    for i in range(nn_config.kWorkers):
        worker_tasks[i].start()
        
    #=======================================================================================================
    # start distributed model training

    time_accu = 0
    pi_all=[]
    rootprint(f"Start {nn_config.KMax} iterations of algorithm")
    for iteration in tqdm.tqdm(range(0, nn_config.KMax)):
        #===================================================================================================
        # initialize worker and server
        if iteration == 0:
            # init worker
            for i in range(nn_config.kWorkers):
                worker_cfgs[i].cmd = "init"
                worker_cfgs[i].input_cmd_ready.release()
            
            # init server
            g_in = []
            g_out = []
            error = []
            worker_cfgs[0].cmd_output_ready.acquire()
            for d in worker_cfgs[0].output_of_cmd: 
                g_in.append(torch.zeros_like(d.clone(), requires_grad=False))
                g_out.append(torch.zeros_like(d.clone(), requires_grad=False))
                error.append(torch.zeros_like(d.clone(), requires_grad=False))
            worker_cfgs[0].cmd_output_ready.release()

        #===================================================================================================
        # print acc and loss each epoch
        if iteration % args.verbose_step == 0 and 1==0:
            eval_iteration.append(iteration)
            
            acc = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
            loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
            train_acc.append(acc)
            train_loss.append(loss)
            fn_train_loss_grad_norm.append(grad_norm)
            
            acc  = getAccuracy(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
            loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
            test_acc.append(acc)
            test_loss.append(loss)
            fn_test_loss_grad_norm.append(grad_norm)
            
            # store statistics on the fly
            results = {}
            results["transfered_bits_by_node"] = transfered_bits_by_node
            results["fi_grad_calcs_by_node"] = fi_grad_calcs_by_node
            results["eval_iteration"] = eval_iteration
            results["train_loss"] = train_loss
            results["test_loss"] = test_loss
            results["train_acc"] = train_acc
            results["test_acc"]  = test_acc
            results["fn_train_loss_grad_norm"] = fn_train_loss_grad_norm
            results["fn_test_loss_grad_norm"] = fn_test_loss_grad_norm
            results["compressors"] = worker_cfgs[0].compressor.fullName()
            
            name_pattern = "./{}/experiment_{}_{}_{}_{}_{}_{}_{}_{}_{}_at_{}.bin"
            ser_fname = name_pattern.format(
                args.out_dir, 
                args.layerwise,
                args.compress_strategy, args.compressor,
                args.optim, args.k, args.lr,
                args.num_workers, args.batch_size, 
                args.model, args.dataset
            )
            utils.serialize(results, ser_fname)
            print(f"-- Experiment {nn_config.model_name}@{nn_config.dataset} has been serialised into '{ser_fname}'")
            
            # print out info
            print(f"-- train accuracy: {train_acc[-1]}, train loss: {train_loss[-1]}, train grad norm: {fn_train_loss_grad_norm[-1]}")
            print(f"-- test accuracy: {test_acc[-1]}, test loss: {test_loss[-1]}, test grad norm: {fn_test_loss_grad_norm[-1]}")
            print(f"-- parameter size: {nn_config.D}, communication cost: {transfered_bits_by_node[0, max(0, iteration-1)]}")
            print(f"-- {worker_cfgs[0].layerwise} compressor: {worker_cfgs[0].compressor.fullName()}, compress strategy: {worker_cfgs[0].compress_strategy}")
            print(f"-- optimizer: {nn_config.optim}, batch size: {nn_config.batch_size}, lr: {nn_config.gamma}")
            
        start_time = time.time()
        
        #===================================================================================================
        # recieve signal from workers
        for i in range(nn_config.kWorkers): 
            worker_cfgs[i].cmd_output_ready.acquire()
                
        signal_recieved = worker_cfgs[0].output_of_cmd
        worker_cfgs[0].output_of_cmd = None
        for i in range(1, nn_config.kWorkers): 
            for j in range(len(worker_cfgs[i].output_of_cmd)):
                signal_recieved[j] = signal_recieved[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
            worker_cfgs[i].output_of_cmd = None
        signal_recieved = utils.mult_param(1.0/nn_config.kWorkers, signal_recieved)
        
        # compress and update server state
        if args.compress_strategy == 'markov':
            # update \hat{g}
            g_in = utils.add_params(g_in, signal_recieved)
            # take the difference and compress
            tmp = utils.sub_params(g_in, g_out)
            
            if worker_cfgs[0].layerwise == 'flatten':
                signal_sent, transit_bits = worker_tasks[0].compressSignal(tmp)
            elif worker_cfgs[0].layerwise == 'layerwise':
                signal_sent, transit_bits = worker_tasks[0].compressSignal_layerwise(tmp)
            else:
                exit(f'unknown layerwise compress {worker_cfgs[0].layerwise}')
            
            # update \tilde{g}
            g_out = utils.add_params(g_out, signal_sent)
            
            if iteration % args.verbose_step == 0:
                diff = utils.norm_of_param(utils.sub_params(signal_sent, tmp))
                norm = utils.norm_of_param(tmp)
                pi = 1 - diff/norm
                pi_all.append(pi)
                print(f'-- markov diff={diff}, norm={norm}, pi={pi}, min/max/avg={min(pi_all)}/{max(pi_all)}/{sum(pi_all)/len(pi_all)}')
            
        elif args.compress_strategy == 'errorFeedback':
            tmp = utils.add_params(error, signal_recieved)
            
            if worker_cfgs[0].layerwise == 'flatten':
                signal_sent, transit_bits = worker_tasks[0].compressSignal(tmp)
            elif worker_cfgs[0].layerwise == 'layerwise':
                signal_sent, transit_bits = worker_tasks[0].compressSignal_layerwise(tmp)
            else:
                exit(f'unknown layerwise compress {worker_cfgs[0].layerwise}')
            
            error = utils.sub_params(tmp, signal_sent)
            
            if iteration % args.verbose_step == 0:
                ef_diff = utils.norm_of_param(utils.sub_params(signal_sent, signal_recieved))
                g_norm = utils.norm_of_param(signal_recieved)
                print(f'-- ef diff={ef_diff}, norm={g_norm}')
            
        elif args.compress_strategy == 'naive':
            if worker_cfgs[0].layerwise == 'flatten':
                signal_sent, transit_bits = worker_tasks[0].compressSignal(signal_recieved)
            elif worker_cfgs[0].layerwise == 'layerwise':
                signal_sent, transit_bits = worker_tasks[0].compressSignal_layerwise(signal_recieved)
            else:
                exit(f'unknown layerwise compress {worker_cfgs[0].layerwise}')
            
            if iteration % args.verbose_step == 0:
                naive_diff = utils.norm_of_param(utils.sub_params(signal_sent, signal_recieved))
                g_norm = utils.norm_of_param(signal_recieved)
                print(f'-- naive diff={naive_diff}, norm={g_norm}')
            
        else:
            exit('unknown compression strategy: {}'.format(args.compress_strategy))
        
        # Broadcast to workers
        signal_sent_cp = copy.deepcopy(signal_sent)
        signal_device = {}
        for device_id in range(len(available_devices)):
            signal_loc = []
            for s_i in signal_sent_cp:
                signal_loc.append(s_i.to(available_devices[device_id]))
            signal_device[available_devices[device_id]] = signal_loc

        for i in range(nn_config.kWorkers):
            worker_cfgs[i].cmd = "run"
            worker_cfgs[i].input_for_cmd = signal_device[worker_cfgs[i].device]
            worker_cfgs[i].input_iteration_for_cmd = iteration
            worker_cfgs[i].input_cmd_ready.release()

        time_accu += time.time() - start_time
        if iteration % args.verbose_step == 0:
            print("--- %s; %s seconds ---" % (time.time() - start_time, time_accu/(iteration+1)))
    #===================================================================================
    # finish all work of nodes
    
    for i in range(nn_config.kWorkers):
        worker_cfgs[i].cmd = "exit"
        worker_cfgs[i].input_cmd_ready.release()
    
    for i in range(nn_config.kWorkers):
        worker_tasks[i].join()
    print(f"Master has been finished")
    
    
if __name__ == "__main__":
    main()
