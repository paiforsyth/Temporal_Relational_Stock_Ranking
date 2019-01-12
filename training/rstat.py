import pandas as pd
import itertools
import copy
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



def compute_relationship_correlations(abs_cor_mat, relationship_matrix):
    dim = len(abs_cor_mat)
    rr_mat = relationship_matrix * np.tri(dim,dim,-1)
    dex1, dex2 = rr_mat.nonzero()
    cor_vec = abs_cor_mat[dex1,dex2]
    return cor_vec

def get_wikidata_correltions(abs_cor_mat):
    num_stocks = len(abs_cor_mat)
    r_tensor = get_relation_data(relation_name="wikidata")
    index_to_industry = get_index_to_industry()
    index_to_correlation ={}
    for index in index_to_industry:
        r_tensor_slice = r_tensor[:,:,index] #slice of relationships pertaining to a given industry
        cor = compute_relationship_correlations(abs_cor_mat, r_tensor_slice)
          


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

        v = np.diag(r_tensor_slice) 
        company_indexes = v.nonzero()[0]#since return is a tuple
        pair_indexes = list(zip(*list(itertools.combinations(company_indexes.tolist(),2 )))) 
        #import pdb; pdb.set_trace()

        cors = abs_cor_mat[pair_indexes[0], pair_indexes[1]] #1d array consisting the of the correlations between
        industry = index_to_industry[index]
        index_to_correlation[index] = {"industry":industry, "correlations": cors, "mean": np.mean(cors), "std": np.std(cors), "company_indexes": company_indexes, "num_companies": sum(v) }
        # if index_to_correlation[index]["mean"] > max_cor:
            # max_cor = index_to_correlation[index]["mean"]
            # max_cor_dex = index 
    
    #add an industry consisting of all stocks
    flat_cor_mat = abs_cor_mat.reshape(-1)
    index_to_correlation[-1] = {"industry": "all", "correlations": flat_cor_mat, "mean": np.mean(flat_cor_mat), "std": np.std(flat_cor_mat) , "company_indexes": list(range(num_stocks)), "num_companies": num_stocks }

    #compute data for related and unrelated stocks
    related = np.sum(r_tensor, axis=2)
    related[related>0] = 1
    r_cors = compute_relationship_correlations(abs_cor_mat, related)
    index_to_correlation[-2] =  {"industry": "related", "correlations": r_cors, "mean": np.mean(r_cors), "std": np.std(r_cors) , "company_indexes":float("NaN") , "num_companies": float("NaN") }
    print("mean absolute correlation of related stocks is {0:.4f} with std {1:.4f}".format(np.mean(r_cors), np.std(r_cors)))

    unrelated = np.ones(related.shape) - related
    u_cors = compute_relationship_correlations(abs_cor_mat, unrelated)
    index_to_correlation[-3] =  {"industry": "related", "correlations": u_cors, "mean": np.mean(u_cors), "std": np.std(u_cors) , "company_indexes":float("NaN") , "num_companies": float("NaN") }
    print("mean absolute correlation of unrelated stocks is {0:.4f} with std {1:.4f}".format(np.mean(u_cors), np.std(u_cors) ))



    

    # import pdb; pdb.set_trace()
    df = pd.DataFrame(index_to_correlation).transpose()
    # mi = index_to_correlation[max_cor_dex]
    ma = index_to_correlation[-1]
    # print("maximum correlation is the {} industry which has mean {} std {} ".format(mi["industry"], mi["mean"], mi["std"] ))
    print("overall correlation is mean {0:.4f} std {1:.4f}".format(ma["mean"], ma["std"] ))


    df = df.sort_values(by = "mean", ascending=False)
    return df



def plot_industry_data(index, d,legend=True, market_name=""):
    index_to_industry = get_index_to_industry()
    industry_name = index_to_industry[index]
    r_tensor = get_relation_data()
    r_tensor_slice = r_tensor[:,:,index] #slice of relationships pertaining to a given industry
    v = np.diag(r_tensor_slice) 
    company_indexes = v.nonzero()[0]#since return is a tuple
    price_data = d["eod_data"][:,:,-1]
    industry_price_data = price_data[company_indexes ,:]
    industry_price_data[industry_price_data == 1.1 ] =float("NaN") #This is the dummy value for missing days
    tickers = d["tickers"]
    labels = [tickers[dex] for dex in company_indexes  ]
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    for i in range(len(industry_price_data)) :
        ax.plot(industry_price_data[i,:])

    # ax.plot(y=industry_price_data.transpose())
    if legend:
        ax.legend(labels)
    plt.title(market_name + industry_name)
    plt.show()











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
    import pdb; pdb.set_trace()
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

def test_plot( fname = "test_eod_file" ):
     with open(fname,'rb') as f:
        d = pickle.load(f)
     plot_industry_data(32, d, legend=True, market_name = "NASDAQ_")
    

def test_cor_compute(fname = "test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    cor_mat = correlate_returns(d,0,50)
    
def test_industry_cor(fname="test_eod_file"):
    with open(fname,'rb') as f:
        d = pickle.load(f)
    mat = np.load("../cor_data/correlation_matrix.npy")
    abs_cor_mat = abs(mat)
    # import pdb; pdb.set_trace()
    df = get_industry_correlatons(abs_cor_mat)
    # print(df.loc[:,["industry", "mean", "num_companies"]])
    df.to_excel("../cor_data/industry/NASDAQ.xlsx")
    # fig,ax = plt.subplots()
    # ax.hist(df.loc[-1,"correlations"] )
    # plt.title("all corelations")

    # fig,ax = plt.subplots()
    # ax.hist(df.loc[-2,"correlations"] )
    # plt.title("related corelations")

    # fig,ax = plt.subplots()
    # ax.hist(df.loc[-3,"correlations"] )
    # plt.title("unrelated corelations")

    # plt.show()
    

    
    return df
    


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
    df = get_industry_correlatons(abs_cor_mat)
    return df
    #print(index_to_cor)
 
    


if  __name__ == "__main__":
    test_plot()
    # df = test_industry_cor()
    # import pdb; pdb.set_trace()
    # df = test_fm_industry_cor()
    # logging.getLogger().setLevel(logging.INFO)
    # get_relation_data()
    # test_fm_cor_mat()
    #test_cor_matshow()
    # test_cor_load()
    # test_cor_compute()
    #print(get_industry_names())
    # test_cor_mat()
