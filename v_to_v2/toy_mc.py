#! /usr/bin/env python

import random, math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import argparse
import json

import numpy as np

# Useful constants
twoPi = 2*math.pi

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate toys of single particle vectors')
    parser.add_argument('nGen', metavar='N', type=int,
                        help='Number of events to generate.')
    parser.add_argument('outfile',
                        help='File name for saving output.  Will be a compressed numpy file (.npz).  Extension will be added for you if ommitted')
    
    parser.add_argument('--cartesian', action="store_true",
                        help='Train network in cartesian coordinates.')

    args = parser.parse_args()

    inputs = np.empty([args.nGen, 3], dtype='float32')
    outputs = np.empty([args.nGen, 3], dtype='float32')

    for iEvt in range(args.nGen):

        if iEvt%1000 == 0:
            print('Generating event {}'.format(iEvt))

        # Make a particle X
        xPt = random.uniform(0,300)
        xPhi = random.uniform(0,twoPi)
        xEta = random.uniform(-2.5,2.5)

        #Start with a null vector and then set pt, eta, phi, and mass
        xVect = ROOT.TLorentzVector(0,0,0,0)
        xVect.SetPtEtaPhiM(xPt,xEta,xPhi,0)

        if args.cartesian:
            inputs[iEvt,0:3] = [xVect.Px(),xVect.Py(), xVect.Pz()]
            outputs[iEvt,0:3] = [xVect.Px()**2,xVect.Py()**2, xVect.Pz()**2]
        else:
            inputs[iEvt,0:3] = [xVect.Pt(),xVect.Eta(), xVect.Phi()]
            outputs[iEvt,0:3] = [xVect.Pt()**2,xVect.Eta()**2, xVect.Phi()**2]
        


    np.savez_compressed(args.outfile,inputs=inputs,outputs=outputs)


    print('Done!')
