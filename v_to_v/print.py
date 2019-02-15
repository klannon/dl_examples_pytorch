#! /usr/bin/env python

# What is this?  It's a very simple print program to verify whether
# the trained network satisfies the expectations for a solution to the
# linear problem.  This only works for the simple case with 3 nodes in
# 1 hidden layer.

import numpy as np
import torch
import argparse, tarfile, os, tempfile, shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Print out some stuff from our model.  '+
                                     'Hard-coded for simplest model with 3 nodes in hidden '+
                                     'layer.')

    parser.add_argument('model',
                        help='Model tarfile')



    args = parser.parse_args()

    # Untar the model files into a tmp directory
    tempDir = tempfile.mkdtemp()
    tar = tarfile.open(args.model)
    tar.extractall(tempDir)
    tar.close()

    # Build them model
    model = torch.load(os.path.join(tempDir,'model.pkl'))
    print model    
    model.eval()
    
    shutil.rmtree(tempDir)

    # Print the model
    with torch.no_grad():
        weights =  model.state_dict()

        print '\nInput to hidden weights (A):'
        for iRow in range(0,3):
            for iCol in range(0,3):
                print '{:12.8f}  '.format(weights['0.weight'][iCol][iRow].item()), # Weight matrix is transpose of what you'd expect
            print ' '
        print ' ' 

        print 'Hidden Layer Bias Weights (a):'
        for i in range(0,3):
            print '{:12.8f}  '.format(weights['0.bias'][i].item()), 
        print ' '
        print ' '

        print 'Input to hidden weights (B):'
        for iRow in range(0,3):
            for iCol in range(0,3):
                print '{:12.8f}  '.format(weights['2.weight'][iCol][iRow].item()), # Weight matrix is transpose of what you'd expect
            print ' '
        print ' '

        print 'Output Layer Bias Weights (b):'
        for i in range(0,3):
            print '{:12.8f}  '.format(weights['2.bias'][i].item()), 
        print ' '
        print ' '

        xCheck = np.dot(weights['0.weight'].numpy(),weights['2.weight'].numpy())

        print 'Weights Cross-Check A*B (Should be ~I):'
        for iRow in range(0,3):
            for iCol in range(0,3):
                print '{:12.8f}  '.format(xCheck[iCol][iRow]), # Weight matrix is transpose of what you'd expect
            print ' '
        print ' '

        xCheck2 = np.dot(weights['2.weight'].numpy(),weights['0.bias'].numpy())+weights['2.bias'].numpy()

        print 'Bias Cross Check B*a+b (Should be 0):'
        for i in range(0,3):
            print '{:12.8f}  '.format(xCheck2[i]), 
        print ' '
        print ' '


        # First, get the predictions:
        inputs = torch.tensor([[0.1,0.2,0.3],
                               [0.3,0.2,0.1],
                               [0.7,0.8,0.9],
                               [0.9,0.8,0.7],
                               [0.5,0.5,0.5],
                               [0,0,0,],
                               [1,1,1],
                               [-0.5,-0.5,-0.5],
                               [1.5,1.5,1.5]])
        outputs = model(inputs).numpy()

        print 'Checking some test patterns:'
        for input,output in zip(inputs,outputs):
            line = '['
            for i in range(0,3):
                line+='{:12.8f}'.format(input[i])
                if i < 2:
                    line += ','
            line += '] -> ['
            for i in range(0,3):
                line += '{:12.8f}'.format(output[i])
                if i < 2:
                    line += ','
            line += ']'
            print line

