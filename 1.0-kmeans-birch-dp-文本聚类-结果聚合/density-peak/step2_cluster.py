#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import *
from cluster import *
from groupClusterLabel import *

def plot(data, density_threshold, distance_threshold, auto_select_dc = False):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	dpcluster = DensityPeakCluster()
	rho, delta, nneigh, clusterLabel = dpcluster.cluster(load_paperdata, data, density_threshold, distance_threshold, auto_select_dc = auto_select_dc)
	logger.info(str(len(dpcluster.ccenter)) + ' center as below')
	for idx, center in dpcluster.ccenter.items():
		logger.info('%d %f %f' %(idx, rho[center], delta[center]))
	plot_rho_delta(rho, delta)   #plot to choose the threthold
	plot_rhodelta_rho(rho,delta)
	plot_cluster(dpcluster)
	resultLabel=dict2Array(clusterLabel)
	groupClusterLabel(resultLabel,4,100)

#将获得的cluster字典转化成list    
def dict2Array(clusterLabel):
    resultLabel=[]
    for label in clusterLabel.values():
        resultLabel.append(label)
    return resultLabel


if __name__ == '__main__':
	#plot('./data/data_in_paper/example_distances.dat', 20, 0.1,False)
	#plot('./data/data_others/spiral_distance.dat',8,5,False)
	#plot('./data/data_others/aggregation_distance.dat',15,4.5,False)
	#plot('./data/data_others/flame_distance.dat',4,7,False)
	#plot('./data/data_others/jain_distance.dat',12,10,False)
	plot('../../data/results/weightPCA2dat.dat',136.727,1.14386,False)
    