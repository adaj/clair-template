import numpy as np
import pandas as pd
# Utility functions (and db analysis calls later on)


def type_encoding(dic):
    new_dic = {}
    for k in dic.keys():
        if 'lags_' in k:
            continue
        obj = dic[k]
        if isinstance(obj,  (np.int32, np.int64)):
            new_dic[k] = int(obj)
        elif isinstance(obj,  (np.float32, np.float64)):
            new_dic[k] = float(obj)
        elif isinstance(obj, pd.Timestamp):
            new_dic[k] = str(dic['timestamp'].asm8)[:26]
        elif isinstance(obj, np.ndarray):
            tolist = []
            for x in obj.tolist():
                if isinstance(obj,  (np.int32, np.int64)):
                    tolist.append(int(obj))
                elif isinstance(obj,  (np.float32, np.float64)):
                    tolist.append(float(obj))
            new_dic[k] = tolist
        elif isinstance(obj, dict):
            new_dic[k] = type_encoding(obj)
        elif isinstance(obj, list):
            tolist = []
            for i in obj:
                if isinstance(i,  (np.int32, np.int64)):
                    tolist.append(int(i))
                elif isinstance(i,  (np.float32, np.float64)):
                    tolist.append(float(i))
                else:
                    tolist.append(i)
            new_dic[k] = tolist
        else:
            new_dic[k] = obj
    return new_dic


def remove_none_keys(dic):
    new_dic = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            new_dic[key] = {}
            for k in dic[key].keys():
                if dic[key][k] is not None:
                    new_dic[key][k] = dic[key][k]
        elif dic[key] is not None:
            new_dic[key] = dic[key]
    return new_dic
