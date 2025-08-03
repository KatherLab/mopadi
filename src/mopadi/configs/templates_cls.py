from mopadi.configs.templates import *


def lung_pretrained_mil():
    conf = MILconfig()
    conf.nr_feats = 512
    conf.num_epochs = 50
    conf.target_label = "Type"
    conf.target_dict = {"LUAD": 0, "LUSC": 1}
    return conf

def crc_pretrained_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 150
    conf.target_label = "isMSIH"
    conf.target_dict = {"nonMSIH": 0, "MSIH": 1}
    return conf

def brca_type_pretrained_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 200
    conf.target_label = "BRCA_Pathology"
    conf.target_dict = {"IDC": 0, "ILC": 1}
    return conf

def brca_e2_pretrained_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.num_epochs = 200
    conf.target_label = "is_E2"
    conf.target_dict = {"No": 0, "Yes": 1}
    return conf

def liver_types_pretrained_mil():
    conf = MILconfig()    
    conf.nr_feats = 512
    conf.target_label = "Type"
    conf.num_epochs = 140
    conf.target_dict = {"LIHC": 0, "CHOL": 1}
    return conf


def default_mil_conf(config):
    conf = MILconfig()

    data_config = config.get('data', {})
    conf.data_dirs = list(data_config.get('data_dirs', None))
    conf.test_patients_file_path = data_config.get('test_patients_file_path', None)
    conf.split = data_config.get('split', 'none')
    conf.max_tiles_per_patient = data_config.get('max_tiles_per_patient', None)
    conf.cohort_size_threshold = data_config.get('cohort_size_threshold', None)
    conf.do_normalize = data_config.get('do_normalize', True)
    conf.do_resize = data_config.get('do_resize', False)
    conf.to_tensor = data_config.get('to_tensor', True)
    conf.img_size = data_config.get('img_size', 224)
    conf.process_only_zips = data_config.get('process_only_zips', False)
    conf.cache_pickle_tiles_path = data_config.get('cache_pickle_tiles_path', None)
    conf.cache_cohort_sizes_path = data_config.get('cache_cohort_sizes_path', None)

    mil_config = config.get('mil_classifier', {})
    conf.base_dir = config.get('base_dir', 'checkpoints/no_name')

    conf.nr_feats = mil_config.get('nr_feats', 512)
    conf.num_epochs = mil_config.get('num_epochs', 100)
    conf.target_label = mil_config.get('target_label', None)
    conf.target_dict = mil_config.get('target_dict', None)
    conf.nr_folds = mil_config.get('number_of_folds', 5)
    conf.fname_index = mil_config.get('fname_index', 3)
    conf.feat_path = mil_config.get('feat_path_train', None)
    conf.feat_path_test = mil_config.get('feat_path_test', None)
    conf.clini_table = mil_config.get('clini_table', None)
    conf.out_dir = os.path.join(conf.base_dir, 'mil_classifier_' + conf.target_label)
    
    conf.man_amps = list(mil_config.get('manipulation_levels', [0.2, 0.4, 0.6, 0.8]))
    conf.images_dir = mil_config.get('images_dir', None)
    conf.patients = mil_config.get('patients', None)
    conf.filename = mil_config.get('filename', None)
    conf.nr_top_tiles = mil_config.get('nr_top_tiles', 5)

    conf.use_pretrained = mil_config.get('use_pretrained', False)
    conf.pretrained_autoenc_name = mil_config.get('pretrained_autoenc_name', None)
    conf.pretrained_clf_name = mil_config.get('pretrained_clf_name', None)

    if conf.use_pretrained:
        assert conf.pretrained_autoenc_name is not None, "Pretrained autoencoder name must be provided if use_pretrained is True"
        assert conf.pretrained_clf_name is not None, "Pretrained MIL model name must be provided if use_pretrained is True"
        pretrained_mil_conf = None
        if conf.pretrained_autoenc_name == "crc_512_model":
            conf.pretrained_autoenc_conf = tcga_crc_autoenc()
            if conf.pretrained_clf_name == "msi":
                pretrained_mil_conf = crc_pretrained_mil()
        elif conf.pretrained_autoenc_name == "brca_512_model":
            conf.pretrained_autoenc_conf = tcga_brca_autoenc()
            if conf.pretrained_clf_name == "e2_center":
                pretrained_mil_conf = brca_e2_pretrained_mil()
            elif conf.pretrained_clf_name == "type":
                pretrained_mil_conf = brca_type_pretrained_mil()
        elif conf.pretrained_autoenc_name == "pancancer_model":
            conf.pretrained_autoenc_conf = pancancer_autoenc()
            if conf.pretrained_clf_name == "lung":
                pretrained_mil_conf = lung_pretrained_mil()
            elif conf.pretrained_clf_name == "liver":
                pretrained_mil_conf = liver_types_pretrained_mil()
        assert pretrained_mil_conf is not None, f"Pretrained MIL model not found. Please check whether '{conf.pretrained_clf_name}' MIL model is available for the selected autoencoder: {conf.pretrained_autoenc_name}."
        
        if conf.target_dict is None:
            conf.target_dict = pretrained_mil_conf.target_dict
        if conf.target_label is None:
            conf.target_label = pretrained_mil_conf.target_label

        if pretrained_mil_conf.target_dict != conf.target_dict:
            print(f"Target dict did not match the selected pretrained model. Changing the target dict to: {pretrained_mil_conf.target_dict}")
            conf.target_dict = pretrained_mil_conf.target_dict

        if pretrained_mil_conf.target_label != conf.target_label:
            print(f"Target label {conf.target_label} did not match the selected pretrained model. Changing the target label to: {pretrained_mil_conf.target_label}")
            conf.target_label = pretrained_mil_conf.target_label
    else:
        conf.pretrained_autoenc_conf = None

    return conf

def default_linear_clf(config):
    data_config = config.get('data', {})
    model_config = config.get('linear_classifier', {})
    target = model_config.get('target', '')
    use_pretrained_clf = model_config.get('use_pretrained_clf', False)

    if use_pretrained_clf:
        print("Using default 'Texture (NCT100k)' model.")
        conf = texture100k_autoenc()
        conf.train_mode = TrainMode.manipulate
        conf.manipulate_znormalize = True
        conf.name = 'linear_clf' + '_' + target
        conf.id_to_cls = list(model_config.get('classes', None))
    else:
        conf = default_autoenc(config)
        conf.load_pretrained_autoenc = True
        conf.train_mode = TrainMode.manipulate
        conf.manipulate_znormalize = True
        conf.name = 'linear_clf' + '_' + target
        conf.id_to_cls = list(model_config.get('classes', None))
        conf.attr_path = model_config.get('attr_path', None)
        conf.batch_size = model_config.get('batch_size', 32)
        conf.lr = float(model_config.get('lr', 1e-3))
        conf.total_samples = model_config.get('total_samples', 300000)
        
    conf.feats_infer_path = model_config.get('feats_infer_path', None)
    return conf


def pretrain_texture100k_autoenc():
    conf = texture100k_autoenc()
    conf.pretrain = PretrainConfig(
        name='Texture',
        path=f'{ws_path}/mopadi/checkpoints/texture100k/last.ckpt',
    )
    conf.feats_infer_path = f'{ws_path}/mopadi/checkpoints/texture100k/latent.pkl'
    return conf


def texture100k_linear_cls():
    conf = texture100k_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "texture"
    conf.manipulate_znormalize = True
    conf.feats_infer_path = f"{ws_path}/mopadi/checkpoints/texture100k/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.pretrain = PretrainConfig(
        "Texture100k",
        f"{ws_path}/mopadi/checkpoints/texture100k/{texture100k_autoenc().name}/last.ckpt",
    )
    conf.name = "texture100k_linear_cls"
    return conf
