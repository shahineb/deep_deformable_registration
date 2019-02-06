import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import distance
from sklearn.metrics import f1_score, mean_squared_error


def pipeline(model, x, y, segmentation=True):
    # metrics for segmentation (x = one array )
    # metrics for registration (x = two arrays )
    # model = model to be tested with loaded weights
    # 
    results = pd.DataFrame()
    if not segmentation:
        y1, y2 = model.predict(x)
        results['MSE'] = [mean_squared_error(y.reshape(1, -1), y1.reshape(1, -1))]
        # results['Mutual_info']=
        results['Cross_cor'] = [np.linalg.norm(signal.correlate(y, y1))]
    else:
        y_pred = model.predict(x)
        results['Dice'] = [f1_score(y.reshape(1, -1), y_pred.reshape(1, -1))]
        results['Hausdorff'] = [distance.directed_hausdorff(y, y_pred)]
    return results


def pipeline_test_set(model, gen_val, model_name, segmentation=True):
    # gen_val is a generator
    # For registration : [[src,tgt],[tgt,zeros]]
    # For segmentation : [src,tgt]
    results = pd.DataFrame()
    for val_i in gen_val:
        if not segmentation:
            results = pd.concat([results, pipeline(model, val_i[0], val_i[0][1], segmentation)])
        else:
            results = pd.concat([results, pipeline(model, val_i[0], val_i[1], segmentation)])
    results.mean().to_csv(model_name+'.csv')
    return None

