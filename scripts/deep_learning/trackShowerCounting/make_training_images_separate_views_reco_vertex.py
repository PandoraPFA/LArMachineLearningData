import numpy as np
import uproot
import math
import copy
import csv
from PIL import Image as pimg
import os

# For now: NC = 0, CC numu = 1, CC nue = 2 (allows clear extension for CC nutau one day)
def pdg_to_target(nuPDG, isCC):
    # NC events first
    if isCC == 0:
        return 0
    # CC numu / anti numu
    if abs(nuPDG) == 14:
        return 1
    # CC nue / anti nue
    if abs(nuPDG) == 12:
        return 2

    return -1

def nu_target_to_string(nu_target):
    if nu_target == 0:
        return 'nc'
    elif nu_target == 1:
        return 'cc_numu'
    else:
        return 'cc_nue'
    

def root_to_images(tree_name, input_file, output_dir, file_index):
    with uproot.open(input_file+":"+tree_name) as tree:
        # We'll make images of size 512 x 512 here from events that are 512 x 512 already
        img_size = 512

        # Run in batches of 100 events and only load the branches that we need
        for b, batch in enumerate(tree.iterate(['nuPDG', 'isCC', 'nuEnergy', 'nuTrueVertexX', 'nuTrueVertexY', 'nuTrueVertexZ', 'nuRecoVertexDriftBinU', 'nuRecoVertexWireBinU', 'nuRecoVertexDriftBinV', 'nuRecoVertexWireBinV', 'nuRecoVertexDriftBinW', 'nuRecoVertexWireBinW', 'nTracksU', 'nShowersU', 'nTracksV', 'nShowersV', 'nTracksW', 'nShowersW', 'pixelRow', 'pixelColumn', 'pixelView', 'pixelCharge'], library='np', step_size=100)):

            # True neutrino information
            nuPDG = batch['nuPDG']
            isCC = batch['isCC']
            nuEnergy = batch['nuEnergy']

            # True vertex
            nuTrueVertexX = batch['nuTrueVertexX']
            nuTrueVertexY = batch['nuTrueVertexY']
            nuTrueVertexZ = batch['nuTrueVertexZ']

            # Reco vertex
            nuRecoVertexDriftBinU = batch['nuRecoVertexDriftBinU'] / float(img_size)
            nuRecoVertexWireBinU = batch['nuRecoVertexWireBinU'] / float(img_size)
            nuRecoVertexDriftBinV = batch['nuRecoVertexDriftBinV'] / float(img_size)
            nuRecoVertexWireBinV = batch['nuRecoVertexWireBinV'] / float(img_size)
            nuRecoVertexDriftBinW = batch['nuRecoVertexDriftBinW'] / float(img_size)
            nuRecoVertexWireBinW = batch['nuRecoVertexWireBinW'] / float(img_size)
            
            # True topology information
            nTracksU = batch['nTracksU']
            nTracksV = batch['nTracksV']
            nTracksW = batch['nTracksW']
            nShowersU = batch['nShowersU']
            nShowersV = batch['nShowersV']
            nShowersW = batch['nShowersW']
            
            # Pixel information
            pixelRows = batch['pixelRow']
            pixelCols = batch['pixelColumn']
            pixelViews = batch['pixelView']
            pixelCharges = batch['pixelCharge']
            
            # Loop over the events
            for e in range(len(nuPDG)):
                nu_target = pdg_to_target(nuPDG[e], isCC[e])
                if nu_target < 0:
                    continue;
            
                # Apply truth cuts
                if nuEnergy[e] > 10.0:
                    continue;
                if math.fabs(nuTrueVertexX[e]) > 310.0:
                    continue;
                if math.fabs(nuTrueVertexY[e]) > 550.0:
                    continue;
                if nuTrueVertexZ[e] < 50.0 or nuTrueVertexZ[e] > 1244:
                    continue;

                nu_string = nu_target_to_string(nu_target)
            
                viewImages = [np.zeros((img_size, img_size), dtype='float'), np.zeros((img_size, img_size), dtype='float'), np.zeros((img_size, img_size), dtype='float')]
                for i in range(len(pixelRows[e])):
                    if pixelRows[e][i] >= img_size:
                        continue
                    if pixelCols[e][i] >= img_size:
                        continue
                    charge = pixelCharges[e][i]
                    if charge > 10:
                        charge = 10
                    viewImages[pixelViews[e][i] - 4][pixelRows[e][i]][pixelCols[e][i]] = pixelCharges[e][i] * 25.5
            
                # Save the images for each view separately as greyscale png files
                png_image_u = pimg.new('L',(img_size,img_size))
                png_image_v = pimg.new('L',(img_size,img_size))
                png_image_w = pimg.new('L',(img_size,img_size))
                image_name_u = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_u_' + str(e) + '.png' 
                image_name_v = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_v_' + str(e) + '.png' 
                image_name_w = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_w_' + str(e) + '.png' 
                for x in range(0, img_size):
                    for y in range(0, img_size):
                        png_image_u.putpixel((y,x), int(viewImages[0][x][y]))
                        png_image_v.putpixel((y,x), int(viewImages[1][x][y]))
                        png_image_w.putpixel((y,x), int(viewImages[2][x][y]))
                png_image_u.save(image_name_u,"PNG")
                png_image_v.save(image_name_v,"PNG")
                png_image_w.save(image_name_w,"PNG")
            
                # U view csv file
                truth_name_u = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_u_' + str(e) + '.csv'
                with open(truth_name_u, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([image_name_u, nuRecoVertexDriftBinU[e], nuRecoVertexWireBinU[e], nu_target, nTracksU[e], nShowersU[e]])
                    print([image_name_u, nuRecoVertexDriftBinU[e], nuRecoVertexWireBinU[e], nu_target, nTracksU[e], nShowersU[e]])

                # V view csv file
                truth_name_v = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_v_' + str(e) + '.csv'
                with open(truth_name_v, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([image_name_v, nuRecoVertexDriftBinV[e], nuRecoVertexWireBinV[e], nu_target, nTracksV[e], nShowersV[e]])

                # W view csv file
                truth_name_w = output_dir + nu_string + '_' + str(file_index) + '_' + str(b) + '_w_' + str(e) + '.csv'
                with open(truth_name_w, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([image_name_w, nuRecoVertexDriftBinW[e], nuRecoVertexWireBinW[e], nu_target, nTracksW[e], nShowersW[e]])
            
                #print('Created files', image_name, 'and', truth_name)
            print(' - Finished batch', b)

if __name__ == '__main__':
    root_to_images('pixels','pandoraEvents/trk_5_hits/trainingFile_nue_set_1.root','images/trk_5_hits/',1)
    root_to_images('pixels','pandoraEvents/trk_5_hits/trainingFile_nue_set_2.root','images/trk_5_hits/',3)
    root_to_images('pixels','pandoraEvents/trk_5_hits/trainingFile_nue_set_3.root','images/trk_5_hits/',5)
    root_to_images('pixels','pandoraEvents/trk_5_hits/trainingFile_nue_set_4.root','images/trk_5_hits/',7)
    root_to_images('pixels','pandoraEvents/trk_5_hits/trainingFile_nue_set_5.root','images/trk_5_hits/',9)
