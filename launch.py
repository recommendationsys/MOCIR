#!/usr/bin/python
# encoding: utf-8

import numpy as np
import torch
import logger
from datetime import datetime
import sys
from util import get_objects,set_global_seeds,arg_parser
import data as all_data
# import envs as all_envs
import agents as all_agents
import model as all_FA
import os
def str2bool(str=""):   
    str = str.lower()
    if str.__contains__("yes") or str.__contains__("true") or str.__contains__("y") or str.__contains__("t"):
        return True
    else:
        return False

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    parser = arg_parser()
    parser.add_argument('-seed',type=int, default=2023)
    parser.add_argument('-environment', type=str, default="DataSet")
    parser.add_argument('-data_path',type=str,default="data/original/")
    parser.add_argument('-agent',type=str,default='Train',help="training methods")
    parser.add_argument('-FA',type=str,default="RecModel", help='function approximation')
    parser.add_argument('-T', dest='T', type=int, default=60, help="time_step")
    parser.add_argument('-ST', dest='ST', type=eval, default="[5,10,20,30,40,50,60]", help="evaluation_time_step")
    parser.add_argument('-gpu_no', dest='gpu_no', type=str, default="0", help='which gpu for usage')
    parser.add_argument('-latent_factor', dest='latent_factor', type=int, default=100, help="latent factor")
    parser.add_argument('-learning_rate', dest='learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('-training_epoch', dest='training_epoch', type=int, default=3000, help="training epoch")
    parser.add_argument('-rnn_layer', dest='rnn_layer', type=int, default=2, help="rnn_layer")
    parser.add_argument('-inner_epoch', dest='inner_epoch', type=int, default=50, help="rnn_layer")
    parser.add_argument('-batch', dest='batch', type=int, default=256, help="batch_size")
    parser.add_argument('-gamma', dest='gamma', type=float, default=0.8, help="gamma")
    parser.add_argument('-clip_param', dest='clip_param', type=float, default=0.2, help="clip_param")
    parser.add_argument('-restore_model', dest='restore_model', type=str2bool, default="False", help="")
    parser.add_argument('-num_blocks', dest='num_blocks', type=int, default=2, help="")
    parser.add_argument('-num_heads', dest='num_heads', type=int, default=1, help="")   
    parser.add_argument('-dropout_rate', dest='dropout_rate', type=float, default=0.1, help="")
    parser.add_argument('-meta_path', dest='meta_path', type=list, default=["mgm","mam","mdm"], help="")
    parser.add_argument('-lambta', dest='lambta', type=float, default=0.1, help="")
    return parser

def main(args):
    # arguments
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)  
    args.model = "_".join([args.agent,args.FA,str(args.T)]) 
    # initialization
    set_global_seeds(args.seed) 
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
    # logger
    # dir = './logdata/training methods_function approximation_3_20211206_012831_data_0.01_3_[10, 30, 60, 120]_0.0'
    logger.configure("./log"+args.data_path.split("/")[-2]+"/"+"_".join([args.model,datetime.now().strftime("%Y%m%d_%H%M%S"),args.data_path.split("/")[-2],str(args.learning_rate),str(args.T),str(args.ST),str(args.gamma)]))
    logger.log("Training Model: "+args.model)
    # dataprocess
    datasets = get_objects(all_data)
    dataset = datasets[args.environment](args)
    dataset.dataprocess()
    args.user_num = dataset.user_num
    args.item_num = dataset.item_num
    args.director_num = dataset.director_num

    
    # # environments
    # envs = get_objects(all_envs)    
    # env = envs[args.environment](args)  
    # # ipdb.set_trace()
    # # policy network
    # args.user_num = env.user_num
    # args.item_num = env.item_num
    # args.utype_num = env.utype_num
    # # ipdb.set_trace()
    args.saved_path = os.path.join(os.path.abspath("./"),"saved_path_"+args.data_path.split("/")[-2]+"_"+str(args.FA)+"_"+str(args.learning_rate)+"_"+str(args.agent)+"_"+str(args.seed))


    nets = get_objects(all_FA)
    fa = nets[args.FA](args)
    # f = fa(dataset.mam)
    logger.log("Hype-Parameters: "+str(args))
    
    
    # agents
    agents = get_objects(all_agents)
    agents[args.agent](dataset,fa,args).train()




if __name__ == '__main__':
    # torch.cuda.set_device(0)
    main(sys.argv)