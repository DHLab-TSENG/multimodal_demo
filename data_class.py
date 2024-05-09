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


class embeddings:
    def __init__(self):
        self.tabular_emb = None
        self.timeseries_emb = None
        self.signal_emb = None
        self.image_emb = None
        self.note_emb = None


vitalsigns_name = {
    220045: 'heart rate',
    220179: 'systolic blood pressure',
    220180: 'diastolic blood pressure',
    220210: 'respiratory rate',
    220277: 'blood oxygen'
    }

labs_name = {
    50810: 'hematocrit',
    50868: 'anion gap',
    50882: 'bicarbonate',
    50893: 'total calcium',
    50902: 'chloride',
    50912: 'creatinine',
    50931: 'glucose',
    50960: 'magnesium',
    50970: 'phosphate',
    50971: 'potassium',
    50983: 'sodium',
    51006: 'urea nitrogen',
    51222: 'hemoglobin',
    51248: 'mean corpuscular hemoglobin',
    51249: 'mean corpuscular hemoglobin concentration',
    51250: 'mean corpuscular volume',
    51256: 'neutrophils',
    51265: 'platelet count',
    51277: 'red blood cell distribution width',
    51279: 'red blood cells',
    51301: 'white blood cells'
    }

diagnosis_name = {
    2:"Septicemia",
    122:"Pneumonia",
    98:"Essential hypertension",
    259:"Residual codes",
    133:"Other lower respiratory disease",
    106:"Cardiac dysrhythmias",
    109:"Acute cerebrovascular disease",
    55:"Fluid and electrolyte disorders",
    108:"Congestive heart failure",
    117:"Other circulatory disease",
    2603:"E Codes Fall",
    2620:"E Codes unspecified",
    49:"Diabetes mellitus without complication"
    }

