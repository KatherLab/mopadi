from configs.templates import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def lung_mil():
    conf = MILconfig()
    conf.nr_feats = 512
    conf.num_epochs = 50
    conf.sab_ln = False
    conf.target_label = "Type"
    conf.target_dict = {"Lung_squamous_cell_carcinoma": 0, 
                        "Lung_adenocarcinoma": 1}
    return conf

def crc_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 150
    conf.sab_ln = False
    conf.target_label = "isMSIH"
    conf.target_dict = {"nonMSIH": 0, "MSIH": 1}
    return conf

def brca_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 200
    conf.sab_ln = False
    #conf.target_label = "PIK3CA_driver"
    conf.target_label = "BRCA_Pathology"
    #conf.target_label = "TP53"HRD_binary
    #conf.target_label = "HRD_binary"
    #conf.target_dict = {"HRD_negative": 0, "HRD_positive": 1}
    #conf.target_dict = {"WT": 0, "MUT": 1}
    #conf.target_dict = {"NODRIVER": 0, "DRIVER": 1}
    conf.target_dict = {"IDC": 0, "ILC": 1}
    return conf

def liver_types_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.target_label = "Type"
    conf.num_epochs = 140
    conf.sab_ln = False
    conf.target_dict = {"hcc": 0, "cca": 1}
    return conf
