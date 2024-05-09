from data_class import patient, embeddings, vitalsigns_name, labs_name, diagnosis_name
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from keras.models import load_model
from keras.models import Model
from MIMIC_IV_HAIM_API import get_single_chest_xray_embeddings, get_biobert_embedding_from_events_list
import torchxrayvision as xrv

ecg_model = load_model("./pretrained_models/ecg_diagnosis/model/model.hdf5", compile=False)

def ts_vitalsigns_embedding(vitalsigns):
    vitalsigns_items = vitalsigns.groupby(by = "itemid")
    metrics = ['max', 'min', 'mean', 'variance', 'meandiff', 'meanabsdiff', 'maxdiff', 'sumabsdiff', 'diff', 'npeaks', 'trend']
    vitalsigns_ts_emb = [item_name.replace(" ","_")+"_"+metric for item_name in vitalsigns_name.values() for metric in metrics]
    event_dict = {}
    for ax_i, (itemid, sub_df) in enumerate(vitalsigns_items):
        item_name = vitalsigns_name[itemid]
        event_dict[item_name.replace(" ","_")] = sub_df.value.values
    
    def get_ts_emb(event_dict):
        ts_emb = {k:0 for k in vitalsigns_ts_emb}
        #Compute the following features
        for event, series  in event_dict.items():
            if len(series) >0: #if there is any event
                ts_emb[event+'_max'] = np.max(series)
                ts_emb[event+'_min'] = np.min(series)
                ts_emb[event+'_mean'] = np.mean(series)
                ts_emb[event+'_variance'] = np.var(series)
                diff = np.diff(series)
                ts_emb[event+'_meandiff'] = np.mean(diff)#average change
                ts_emb[event+'_meanabsdiff'] = np.mean(np.abs(diff))
                ts_emb[event+'_maxdiff'] = np.max(np.abs(diff))
                ts_emb[event+'_sumabsdiff'] = np.sum(np.abs(diff))
                ts_emb[event+'_diff'] = series[-1]-series[0]
                #Compute the n_peaks
                peaks,_ = signal.find_peaks(series) #, threshold=series.median()
                ts_emb[event+'_npeaks'] = len(peaks)
                #Compute the trend (linear slope)
                if len(series)>1:
                    ts_emb[event+'_trend']= np.polyfit(np.arange(len(series)), series, 1)[0] #fit deg-1 poly
                else:
                     ts_emb[event+'_trend'] = 0
        return ts_emb
    
    emb = get_ts_emb(event_dict)
    
    return pd.Series(emb)

def ecg_embedding(ecg_sig):
    sig_len = 5000
    fs = 500
    N = int(sig_len/fs * 400)
    pad_width = int((4096-N)/2)
    
    ecg_sig[np.isnan(ecg_sig)] = 0
    def preprocessing(sig):
        return  np.pad(signal.resample(sig,N), pad_width = pad_width)
    
    ecg_sig = np.array([preprocessing(sig) for sig in ecg_sig.transpose()]).transpose()
    input = ecg_sig.reshape(1,ecg_sig.shape[0],ecg_sig.shape[1])
    
    prediction_embeddings = ecg_model.predict(input)
    prediction_embeddings = prediction_embeddings.squeeze()
    diagnosis = ["1dAVb", "RBBB", "LBBB", "SB", "AF","ST"]
    
    model_embedding = Model(inputs=ecg_model.input, outputs=ecg_model.layers[-3].output) 
    resnet_embeddings = model_embedding.predict(input)
    
    resnet_embeddings = np.max(resnet_embeddings[0],0)
    
    emb = {k.replace(" ", "_"):v for k,v in zip(diagnosis,prediction_embeddings)}
    emb.update({"ecg_dense_%d"%(i):v for i, v in enumerate(resnet_embeddings)})
    
    return pd.Series(emb)


def cxr_img_emb(cxr_img):
    densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(cxr_img)
    pathologies = xrv.datasets.default_pathologies
    
    emb = {k.replace(" ", "_"):v for k,v in zip(pathologies,prediction_embeddings)}
    emb.update({"cxr_dense_%d"%(i):v for i, v in enumerate(densefeature_embeddings)})
    
    return pd.Series(emb)


def cxr_note_emb(cxr_note):
    note_list = [cxr_note]
    note_weights = pd.Series([1])
    note_embeddings, _, _ = get_biobert_embedding_from_events_list(note_list,note_weights)
    
    emb = {"textemb_%d"%(i):v for i, v in enumerate(note_embeddings)}
    
    return pd.Series(emb)



if __name__ == "__main__":
    subjects_flag = []
    subjects_list = np.genfromtxt("sample_subjects.csv")
    subjects_list = subjects_list.astype(int)
    for i, subject_id in enumerate(subjects_list):
        emb = embeddings()
        with open("./sample_patient/%d.pkl"%(subject_id), "rb") as f:
            data = pickle.load(f)
        subjects_flag.append({"ID":"S%03d"%(i),"flag":data.expire_flag})
        tabular = {}
        tabular["age"] = data.age
        tabular["gender"] = 1 if data.gender=="M" else 0
        tabular.update({ccs_name.replace(" ","_"):data.ed_diagnosis[ccs_id] for ccs_id,ccs_name in diagnosis_name.items()})
        tabular.update({item_name.replace(" ","_"):data.labs[item_id] if item_id in data.labs else 0 for item_id,item_name in labs_name.items()})
        
        emb.tabular_emb = pd.Series(tabular)
        emb.timeseries_emb = ts_vitalsigns_embedding(data.vitalsigns)
        emb.note_emb = cxr_note_emb(data.cxr_note)
        emb.image_emb = cxr_img_emb(data.cxr_img)
        emb.signal_emb = ecg_embedding(data.ecg)
        
        with open("./embeddings/S%03d.pkl"%(i), "wb") as f:
            pickle.dump(emb, f)
    
    subjects_flag_df = pd.DataFrame(subjects_flag)
    subjects_flag_df.to_csv("./embeddings/flag.csv",index=False)
