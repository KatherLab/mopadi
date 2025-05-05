from mopadi.configs.templates import *


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
    conf.pancancer_type = mil_config.get('pancancer_type', None)

    return conf

def default_linear_clf(config):
    conf = default_autoenc(config)
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_znormalize = True

    data_config = config.get('data', {})
    model_config = config.get('linear_classifier', {})
    target = model_config.get('target', '')

    conf.name = 'linear_clf' + '_' + target
    conf.load_pretrained_autoenc = True

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

    conf.id_to_cls = list(model_config.get('classes', None))
    conf.attr_path = model_config.get('attr_path', None)
    conf.batch_size = model_config.get('batch_size', 32)
    conf.lr = float(model_config.get('lr', 1e-3))
    conf.total_samples = model_config.get('total_samples', 300000)

    return conf
