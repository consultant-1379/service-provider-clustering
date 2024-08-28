import numpy as np
import pandas as pd
import scipy.spatial as sp
import math
import pickle
from sklearn.neighbors import KNeighborsClassifier
from collections import deque
from time import time
import os
#from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)

class serviceProviderClustering():
    """
    Implementation of the service provider clustering model using decision tree for predictions.

    """
    #class constants
    _splist =     [ 'sp_google', 'sp_googleplay', 'sp_samsungapps', 'sp_facebook', 'sp_snapchat',
            'sp_instagram', 'sp_twitter', 'sp_pinterest', 'sp_tumblr', 'sp_flickr', 'sp_googleplus', 'sp_vine',
            'sp_weheartit', 'sp_picasa', 'sp_linkedin', 'sp_yahoo', 'sp_outlook', 'sp_gmail', 'sp_hotmail',
            'sp_microsoftlive', 'sp_microsoft', 'sp_youtube', 'sp_netflix', 'sp_vimeo', 'sp_dailymotion',
            'sp_ustream', 'sp_hulu', 'sp_skype', 'sp_viber', 'sp_whatsapp', 'sp_nimbuzz', 'sp_amazon',
            'sp_ebay', 'sp_officeonline', 'sp_adobe', 'sp_foursquare', 'sp_wikipedia']
    _orderedMSP = ['MSP_Google', 'MSP_SocMed', 'MSP_Mail', 'MSP_Video', 'MSP_Call', 'MSP_Purchase', 'MSP_Prof', 'MSP_Info']

    _mspDefDict = {'MSP_Google':  ['sp_google', 'sp_googleplay', 'sp_samsungapps'],
                   'MSP_SocMed':  ['sp_facebook', 'sp_snapchat', 'sp_instagram', 'sp_twitter',
                       'sp_pinterest', 'sp_tumblr', 'sp_flickr', 'sp_googleplus', 'sp_vine', 'sp_weheartit', 'sp_picasa', 'sp_linkedin'],
                   'MSP_Mail':    ['sp_yahoo', 'sp_outlook', 'sp_gmail', 'sp_hotmail', 'sp_microsoftlive', 'sp_microsoft'],
                   'MSP_Video':   ['sp_youtube', 'sp_netflix', 'sp_vimeo', 'sp_dailymotion', 'sp_ustream', 'sp_hulu'],
                   'MSP_Call':    ['sp_skype', 'sp_viber', 'sp_whatsapp', 'sp_nimbuzz'],
                   'MSP_Purchase':['sp_amazon', 'sp_ebay'],
                   'MSP_Prof':    ['sp_linkedin', 'sp_officeonline', 'sp_adobe', 'sp_microsoft'],
                   'MSP_Info':    ['sp_foursquare', 'sp_wikipedia']
                  }

    _mspPowers =  {'MSP_Google'  :.1,
                   'MSP_SocMed'  :.1,
                   'MSP_Mail'    :.015,
                   'MSP_Video'   :.2,
                   'MSP_Call'    :.25,
                   'MSP_Purchase':.000001,
                   'MSP_Prof'    :.00001,
                   'MSP_Info'    :.0000001
                  }

    _label_dict = {-1 : 'Outliers',
                0 : 'Shopper Power Users',
                1 : 'Shoppers',
                2 : 'Data-less Users',
                3 : 'Emailers',
                4 : 'Information Sharers',
                5 : 'Business Power Users',
                6 : 'Information Consumers'
               }

    #resources
    SCALER = "scl.pkl"
    CLUSTERER = "dt.pkl"

    def __init__(self, scaler_pkl = SCALER, clusterer_pkl = CLUSTERER):
        """
        init method

        Input:  scaler_pkl    - pkl object of the scaler used to scale down the training data between 0 and 1
                custerer_pkl  - pkl object of the decision tree used to cluster the training data into 8 different labels
        """
        t = time()    #start timer
        with open(scaler_pkl, "rb") as f:
            self._scaler = pickle.load(f)
        with open(clusterer_pkl, "rb") as f:
            self._clusterer = pickle.load(f)
        self._preds = deque([])
        self._count = 0

    def _prepare_data(self, scaler_model, data2, scale = True):
            """
            This method reduces the data and create 8 MSP (Master Service Provider) features out of the original ones. It applies an exponential transform to the columns and it scale them.

            Input:  data           - input anonymized data where each sp has its own column.
                    scaler_model   - scaler used to scale down the data between 0 and 1 (roughly)
                    scale          - boolean used to specify wether or not to scale the data using the scaler

            Output: prepared_data  - output anonymized data where each MSP (of the 8 MSP) has its own column and the imsi (anonymized) is preserved for identification.
            """
            #create a copy of the dataframe
            data = pd.DataFrame(data2.astype(float), columns=self._splist)

            #mesure its length
            length = data.shape[0]

            #instantiate the output dataframe
            prepared_data = pd.DataFrame(np.zeros((length,8)))
            prepared_data.columns = self._orderedMSP
            #prepared_data["imsi"] = prepared_data.index.tolist()
            #create a list of interesting columns
            #fill up the output dataframe
            for col in data.columns.tolist():
                if col in self._mspDefDict["MSP_Google"]:
                    prepared_data["MSP_Google"]       =  prepared_data["MSP_Google"]   + data[col]
                elif col in self._mspDefDict["MSP_SocMed"]:
                    prepared_data["MSP_SocMed"]       =  prepared_data["MSP_SocMed"]   + data[col]
                elif col in self._mspDefDict["MSP_Mail"]:
                    prepared_data["MSP_Mail"]         =  prepared_data["MSP_Mail"]     + data[col]
                elif col in self._mspDefDict["MSP_Video"]:
                    prepared_data["MSP_Video"]        =  prepared_data["MSP_Video"]    + data[col]
                elif col in self._mspDefDict["MSP_Call"]:
                    prepared_data["MSP_Call"]         =  prepared_data["MSP_Call"]     + data[col]
                elif col in self._mspDefDict["MSP_Purchase"]:
                    prepared_data["MSP_Purchase"]     =  prepared_data["MSP_Purchase"] + data[col]
                elif col in self._mspDefDict["MSP_Prof"]:
                    prepared_data["MSP_Prof"]         =  prepared_data["MSP_Prof"]     + data[col]
                elif col in self._mspDefDict["MSP_Info"]:
                    prepared_data["MSP_Info"]         =  prepared_data["MSP_Info"]     + data[col]

            #apply an exponential transform to the data in order unskew it
            for mspp in self._mspPowers.items():
                msp = mspp[0]
                power = mspp[1]
                prepared_data.loc[prepared_data.loc[:,msp]>0,msp] = np.power(np.array(prepared_data.loc[prepared_data[msp]>0,msp]), power)
            #scale each feature using the object's scaler fitted on the training data
            if scale:
                prepared_data = scaler_model.transform(prepared_data)
            else:
                prepared_data = prepared_data
            return prepared_data

    def predict(self, data, features_names):
        """
        This method predict the cluster of new (unseen by algorithm) subscribers.

        Input: data            - input data taken in the form of a pd.DataFrame (either vector or matrix)

        Output: clusters (int) - The predicted cluster for a subscriber based on the pre-trained model.
                                 Note that a value of -1 is returned for outliers, and 0 and more for specific in-cluster.
        """
        #cells = data['cellid_set_count_1d'].copy()
        #sli = data['sli_1d'].copy()
        #data_cons = data['data_dl_dy_avg_14d'].copy()
        imsi = data[:,0].reshape(1,-1)
        ts = data[:,-1].reshape(1,-1)
        data = self._prepare_data(self._scaler, data[:,1:-1])
        clusters = self._clusterer.predict(data).reshape(1,-1)
        return np.concatenate((imsi,clusters,clusters,np.array([[self._label_dict[clusters[0][0]]]]),np.array([[1]]),ts), axis=1)

    def predict_str(self, data_str):
        """
        Wrapper for predict(), using a single string parameter instead of a dataframe
        Usage e.g. predict_str('Header\nPinterest 4\nGoogle Play 114\nSkype 5\n')
        """
        return self.predict(pd.read_csv(pd.compat.StringIO(data_str)))

    def get_scaler(self):
        """
        This method returns the scaler used on the training data

        Input: None

        Output: self._scaler    -The scaler used on the training data
        """
        return self._scaler
