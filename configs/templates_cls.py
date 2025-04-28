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

def default_linear_clf(config):
    conf = default_autoenc(config)
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_znormalize = True
    conf.name = 'linear_clf'

    train_config = config.get('train', {})
    data_config = train_config.get('data', {})
    model_config = train_config.get('model', {})
    
    conf.batch_size = model_config.get('batch_size', 32)
    conf.lr = model_config.get('lr', 1e-3)
    conf.total_samples = model_config.get('total_samples', 300_000)
    conf.load_pretrained_autoenc = True

    conf.id_to_cls = list(data_config.get('id_to_cls', None))
    conf.attr_path = data_config.get('attr_path', None)
    conf.test_patients_file_path = data_config.get('test_patients_file_path', None)
    conf.split = data_config.get('split', 'none')
    conf.max_tiles_per_patient = data_config.get('max_tiles_per_patient', None)
    conf.cohort_size_threshold = data_config.get('cohort_size_threshold', 'none')
    conf.do_normalize = data_config.get('do_normalize', True)
    conf.do_resize = data_config.get('do_resize', False)
    conf.to_tensor = data_config.get('to_tensor', True)
    conf.img_size = data_config.get('img_size', 224)
    conf.process_only_zips = data_config.get('process_only_zips', False)
    conf.cache_pickle_tiles_path = data_config.get('cache_pickle_tiles_path', None)
    conf.cache_cohort_sizes_path = data_config.get('cache_cohort_sizes_path', None)

    return conf
