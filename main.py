import argparse
import sys

import torch
from torch import nn, optim

from data import mnist
from model import ConvolutionModel_v1

import re

#Graphics
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style("whitegrid")

import pdb

def load_checkpoint(folderpath,fileName):
    #Check model type
    if 'ConvolutionModel_v1' in fileName:
        model = ConvolutionModel_v1()
    else:
        print('Model type not found')

    #Load model from folderpath
    #pdb.set_trace()
    checkpoint = torch.load(folderpath+'/'+fileName)
    model.load_state_dict(checkpoint)

    return model

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        parser.add_argument('--epochs', default=30)
        parser.add_argument('--batchsize', default=64)
        parser.add_argument('--modelType', default='ConvolutionModel_v1')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[3:])
        print(args)
        
        # Define models, loss-function and optimizer
        if args.modelType == 'ConvolutionModel_v1':
            model = ConvolutionModel_v1()
        else:
            print('Model type not found')
        model.train()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        #Load data
        train_set, _ = mnist(args.batchsize)

        #Training
        epochs = args.epochs

        train_losses = []
        for e in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in train_set:
                
                optimizer.zero_grad()
                
                #pdb.set_trace()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            else:
                train_losses.append(running_loss/len(train_set))
                print('epoch: ',str(e), '/', str(epochs))
                print('Training_loss: ',str(train_losses[e]))
                print('')
        #pdb.set_trace()
        #Save model
        torch.save(model.state_dict(), '/Users/weis/Documents/Skole/DTU/10_semester/ML_Ops/Project/Models/' + args.modelType + '_lr' + str(args.lr) + '_e' + str(args.epochs)+ '_bs' +str(args.batchsize) + '.pth')

        #Plot training loss
        plt.plot(train_losses);
        plt.show()

        


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="/Users/weis/Documents/Skole/DTU/10_semester/ML_Ops/Project/Models")
        parser.add_argument('--modelName', default='ConvolutionModel_v1_lr0.003_e30_bs64.pth')
        
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Load model
        model = load_checkpoint(args.load_model_from,args.modelName)
        model.eval()

        # Extract batch_size
        result = re.search('(.*)_bs(.*).pth', args.modelName)
        #pdb.set_trace()
        batch_size = int(result.group(2))

        # Load data
        _, test_set = mnist(batch_size)

        # Run evaluation
        running_acc = 0
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            
            #get top 1 probs per item
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            
            #Accumulate loss and accuracy
            running_acc += accuracy
        else:
            running_acc = running_acc/len(test_set)
        
        print(f'Test accuracy is: {running_acc*100}%')



if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    