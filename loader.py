import numpy
import os
def _arraylike_copy(arr):
    """Duplicate content of arr into a numpy array.

     Examples
     --------
     >>> X_npy = numpy.array([1, 2, 3])
     >>> numpy.alltrue(_arraylike_copy(X_npy) == X_npy)
     True
     >>> _arraylike_copy(X_npy) is X_npy
     False
     >>> numpy.alltrue(_arraylike_copy([1, 2, 3]) == X_npy)
     True
     """
    if type(arr) != numpy.ndarray:
        return numpy.array(arr)
    else:
        return arr.copy()

def str_to_timeseries(ts_str,delimiter1="|", delimiter2=",", classpos=-1):
    dimensions = ts_str.split(delimiter1)
    ts = [dim_str.split(delimiter2) for dim_str in dimensions]
    if classpos == 0:
        cl = ts[0].pop(0)
    elif classpos == -1:
        cl = ts[-1].pop(-1)
    else:
        cl = None   
        
    #return ts, cl
    return to_time_series(numpy.transpose(ts)), cl


def to_time_series(ts, remove_nans=False):
    ts_out = _arraylike_copy(ts)
    if ts_out.ndim == 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out
    

def to_time_series_dataset(dataset, dtype=numpy.float):
    if numpy.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts)) for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = numpy.zeros((n_ts, max_sz, d), dtype=dtype) + numpy.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out
    
def load_timeseries_txt(fname,delimiter1="|", delimiter2=",", classpos=-1):
    dataset = []
    cl  = []
    fp = open(fname, "rt")
    for row in fp.readlines():
        ts, cli = str_to_timeseries(row,delimiter1=delimiter1, delimiter2=delimiter2, classpos=classpos)
        dataset.append(ts)
        try:
            cl.append(float(cli))
        except:
            cl.append(cli)
    fp.close()
    return to_time_series_dataset(dataset), cl
    
def ts_size(ts):
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and not numpy.any(numpy.isfinite(ts_[sz - 1])):
        sz -= 1
    return sz

def load_dataset(ds_name, path=None, delimiter1="|", delimiter2=",", classpos=-1, only_train=False):
    if path is None:
        path = "data/ucr"
    directory = "%s/%s" % (path, ds_name)
    fname_train, fname_test = None, None
    for fname in os.listdir(directory):
        if fname.endswith("_TRAIN.csv"):
            fname_train = os.path.join(directory, fname)        
        elif fname.endswith("_TEST.csv") and not only_train:
            fname_test = os.path.join(directory, fname)
            
    print(fname_train)
    x_train, y_train = load_timeseries_txt(fname_train, delimiter1=delimiter1, delimiter2=delimiter2, classpos=classpos)
    if  only_train:
        return x_train, y_train
    else:
        x_test, y_test = load_timeseries_txt(fname_test, delimiter1=delimiter1, delimiter2=delimiter2, classpos=classpos)
        return x_train, y_train, x_test, y_test



#X_train, y_train, X_test, y_test = load_dataset('ERing',path='/Users/rcarlini/Downloads/a/MultivariateTSCProblems/')






#d = [] 
#for z in a:
#     d.append(cydtw.dtw(x[20],z))




#f = open('/Users/rcarlini/Downloads/a/MultivariateTSCProblems/Cricket/Cricket_TEST.csv')
#while ts_str = f.readline():
#    dimensions = ts_str.split("|")
    
    
