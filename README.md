
## Wrap-up
-  Slides (Code): https://dhlab-tseng.github.io/multimodal_demo/MMLforEHR.slides.html
-  Notebook: https://dhlab-tseng.github.io/multimodal_demo/MMLforEHR.html
-  Slides: https://docs.google.com/presentation/d/1QeXEGVmBTPQino78FAUfwduMeUtf7UlfINPIQM3ouZc/edit?usp=sharing
  
## Code
### Embeddings Generation
#### HAIM API : [MIMIC_IV_HAIM_API.py](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/MIMIC_IV_HAIM_API.py)  
> Embeddings Generation is based on HAIM[1].  
> Clone repository in https://github.com/lrsoenksen/HAIM/blob/main/MIMIC_IV_HAIM_API.py

#### Time series VitalSigns Embedding : [timeseries_embedding.ipynb](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/timeseries_embedding.ipynb)  
> Time series data are processed by generating statistical metrics on each of the time-dependent to produce embeddings.[1]    
  
#### ECG Signal Embedding : [ecg_embedding.ipynb](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/ecg_embedding.ipynb)  
> Signal inputs are processed using a pre-trained CNN model[9] to extract embeddings that are the model output and dense features.[1]  
> Pre-Trained ECG Model[9] : [Automatic ECG diagnosis](https://github.com/antonior92/automatic-ecg-diagnosis)  
> [Model download link](https://zenodo.org/records/3765717)  

#### CXR Image Embedding : [cxr_embedding.ipynb](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/cxr_embedding.ipynb)  
> Image inputs are processed using a pre-trained CNN model[7] to extract embeddings that are the model output and dense features.[1]   
> Pre-Trained CXR Model[7] : [XRay: TorchXRayVision](https://github.com/mlmed/torchxrayvision)  

#### CXR Diagnosis Note Embedding : [note_embedding.ipynb](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/note_embedding.ipynb)  
> Natural language inputs are processed using a pre-trained transformer model[8] to generate text embeddings.[1]  
> Pre-Trained Language Model[8] : [Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT?tab=readme-ov-file)  
> [Model download link](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?e=1&dl=0)  

### Mortality Prediction : [prediction.ipynb](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/prediction.ipynb)  
> In hospital mortality prediction using fusion embeddings from 5 modalities data(tabular/time series/signal/note/image).

## Data
> This work uses the MIMIC-IV Clinical Database[2][3] to get demographics, laboratory events, vital signs, and ED diagnosis.   
> The ECG signal is from MIMIC-IV-ECG[5].  
> The CXR Image and diagnosis note are from MIMIC-CXR-JPG[4] and MIMIC-IV-Note[6].  

Defined in [data_class.py](https://github.com/DHLab-TSENG/multimodal_demo/blob/main/data_class.py)  
#### Patient class structure
> Patient object contains clinical raw data from a single subject.  
> patient object save as  pickle file, file path is "./sample_patient/[subject_id].pkl".  

```python
class patient:
    def __init__(self):
        self.subject_id = None
        self.hamd_id = None
        self.icu_stay_id = None
        self.ed_stay_id = None
        self.gender = None
        self.age = None
        self.expire_flag = None
        self.ed_diagnosis = None
        self.labs = None
        self.vitalsigns = None
        self.cxr_study_id = None
        self.cxr_img = None
        self.cxr_note = None
        self.ecg_study_id = None
        self.ecg = None
```
```python
#read patient obj
import pickle
from data_class import patient
subject_id = ...
with open("./sample_patient/%s.pkl"%(subject_id), "rb") as f:
    data = pickle.load(f)
print(data.subject_id)

# 12 leads ECG: 2d-array(5000,12), 500Hz * 10 sec
plt.plot(data.ecg[:,0])

# CXR img: 2d-array(224,224)
plt.imshow(data.cxr_img)

# VitalSigns: dataframe(itemid, value, delta_hrs)
print(data.vitalsigns)

# CXR Note: str
print(data.cxr_note)
```

#### Embeddings class structure
> Embeddings object contains embedding from different modalities with single subject.  
> emb_extraction.py is to convert patient object into embeddings object.  
> embeddings object save as  pickle file, file path is "./embeddings/[ID].pkl".  
```python
class embeddings:
    def __init__(self):
        self.tabular_emb = None
        self.timeseries_emb = None
        self.signal_emb = None
        self.image_emb = None
        self.note_emb = None
```

### Reference
1. [Soenksen, L.R., Ma, Y., Zeng, C. et al. Integrated multimodal artificial intelligence framework for healthcare applications. npj Digit. Med. 5, 149 (2022). https://doi.org/10.1038/s41746-022-00689-4.](https://www.nature.com/articles/s41746-022-00689-4)
2. [Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67.](https://physionet.org/content/mimiciv/2.2/)
3. [Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). MIMIC-IV-ED (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72.](https://physionet.org/content/mimic-iv-ed/2.2/)
4. [Johnson, A., Lungren, M., Peng, Y., Lu, Z., Mark, R., Berkowitz, S., & Horng, S. (2024). MIMIC-CXR-JPG - chest radiographs with structured labels (version 2.1.0). PhysioNet. https://doi.org/10.13026/jsn5-t979.](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
5. [Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. https://doi.org/10.13026/4nqg-sb35.](https://physionet.org/content/mimic-iv-note/2.2/)
6. [Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). PhysioNet. https://doi.org/10.13026/1n74-ne17.](https://physionet.org/content/mimic-iv-note/2.2/)
7. [Cohen, Joseph Paul, et al. TorchXRayVision: A library of chest X-ray datasets and models. International Conference on Medical Imaging with Deep Learning. PMLR, 2022.](https://arxiv.org/abs/2111.00595)
8. [Alsentzer, Emily, et al. Publicly available clinical BERT embeddings. arXiv preprint arXiv:1904.03323 (2019).](https://arxiv.org/abs/1904.03323)
9. [Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4](https://www.nature.com/articles/s41467-020-15432-4.)




