from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
from load_data import load_EOD_data, load_relation_data
import argparse
import numpy as np
import pickle
from scipy.stats import spearmanr

#Issue: not sue that the dates are aligned.  Do they actualy need aligned dates?

def save_eod_data(data_path, market_name, tickers, fname) :

    eod_data, masks, ground_truth, base_price = load_EOD_data(data_path, market_name, tickers, steps=1)
    cap_mask = np.prod(masks, axis=0)
    logging.info("In intersection mask, there are {} out of {} masked entries, a proportion of {}".format(len(cap_mask) - sum(cap_mask), len(cap_mask),( len(cap_mask) - sum(cap_mask) )/len(cap_mask) ) )
    data_dict =  {"eod_data":eod_data, "masks": masks, "ground_truth": ground_truth, "base_price": base_price, "market_name" : market_name, "tickers": tickers, "cap_mask": cap_mask}
    #if we use the cap mask, then we mask out half the days, which is unacceptable 
    
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

def get_relation_data(data_path ='../data/2013-01-01', market_name = "NASDAQ", relation_name = "sector_industry"):
    rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
    rpath = os.path.join(data_path, '..', 'relation', relation_name, market_name + rname_tail[relation_name])
    relation_tensor, mask = load_relation_data(rpath)    
    return relation_tensor
    
def get_index_to_industry( data_path ='../data', market_name = "NASDAQ" ):
    rpath = os.path.join(data_path,  'relation', "sector_industry", market_name + "_index_to_industry.npy")
    return np.load(rpath).item() #since dict is stored in array


def get_industry_correlatons(abs_cor_mat):
    assert np.all(abs_cor_mat >= 0)
    num_stocks = len(abs_cor_mat)
    r_tensor = get_relation_data()
    index_to_industry = get_index_to_industry()
    index_to_correlation ={}
    for index in index_to_industry:
        # plt.matshow(r_tensor[:,:,index])
        # plt.show()
        r_tensor_slice = r_tensor[:,:,index] #slice of relationships pertaining to a given industry
        v = r_tensor_slice.sum(axis=0)
        v[v>0]=1
        # v shows us which stocks are in the industry
        r_tensor_slice = r_tensor_slice - np.diag(v) #remove self references
        mean = np.sum((r_tensor_slice * abs_cor_mat) ) / np.sum(r_tensor_slice)
        std =( np.sum(r_tensor_slice * (abs_cor_mat - mean)**2  ) / np.sum(r_tensor_slice))**2
        index_to_correlation[index] = (mean,std,index_to_industry[index])

    #actually I want the vector of actual correlations for each industry use np,nonzero and advanced indexing to do this
    return index_to_correlation













def make_stock_correlation_matrix(d):
    #d is data dict as specified above
    num_stocks = len( d["ground_truth"])    
    cor_mat = np.ones([num_stocks, num_stocks]) * float("nan")

    #cant use matrix multiplication because of inconsistent masking
    for i in tqdm(range(num_stocks)):
        for j in range(i,num_stocks):
            val = correlate_returns(d,i,j)
            cor_mat[i,j] = val
            cor_mat[j,i] = val
    return cor_mat

def write_stock_correlation_matrix(d):
    cor_mat = make_stock_correlation_matrix(d)
    np.save("../cor_data/correlation_matrix",cor_mat)


def make_fully_masked_correlation_matrix(d):
    gt = d["ground_truth"]
    # import pdb; pdb.set_trace()
    masked_gt = gt[:,d["cap_mask"].astype(int)==1]
    rho_mat, p_mat = spearmanr(masked_gt.transpose()) #wantw the stocks to be in the columns
    return rho_mat

def write_fully_masked_correlation_matrix(d):
    fm_cor_mat = make_fully_masked_correlation_matrix(d)
    np.save("../cor_data/fm_correlation_matrix",fm_cor_mat)
    return fm_cor_mat







def correlate_returns(d, index1, index2):
    logging.info("computing correlation between linear daily returns of {} and {} ".format( d["tickers"][index1]  , d["tickers"][index2]  ))
    m1 = d["masks"][index1,:]
    m2 = d["masks"][index2,:]
    m_combined = m1*m2
    assert(sum(m_combined) > 10)
    v1 = d["ground_truth"][index1]
    v2 = d["ground_truth"][index2]
    v1_masked = v1[m_combined == 1]
    v2_masked = v2[m_combined == 1]
    rho, pval = spearmanr(v1, v2)
    logging.info("Spearman correlation is {} pval of rejecting null hypothesis that uncorrelated i {}".format(rho,pval))
    return rho


    
 
def test_cor_load():
    data_path =  '../data/2013-01-01'
    market_name = 'NASDAQ' 
    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv' 
    tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname), dtype=str, delimiter='\t', skip_header=False)
    fname = "test_eod_file"
    save_eod_data(data_path, market_name, tickers, fname)

def test_cor_matshow(fname = "test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    plt.matshow(d["masks"])
    plt.show()

def test_cor_compute(fname = "test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    cor_mat = correlate_returns(d,0,50)
    
    


def test_cor_mat(fname="test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    write_stock_correlation_matrix(d)

def test_fm_cor_mat(fname="test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    mat = write_fully_masked_correlation_matrix(d)
    plt.matshow(mat)
    plt.show()
   

def test_fm_industry_cor(fname="test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    mat = make_fully_masked_correlation_matrix(d)
    abs_cor_mat = abs(mat)
    index_to_cor = get_industry_correlatons(abs_cor_mat)
    print(index_to_cor)
 
    


if  __name__ == "__main__":
    test_fm_industry_cor()
    # logging.getLogger().setLevel(logging.INFO)
    # get_relation_data()
    # test_fm_cor_mat()
    #test_cor_matshow()
    # test_cor_load()
    # test_cor_compute()
    #print(get_industry_names())
