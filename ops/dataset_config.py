'''
Description        : The dataset config for S2D (following the registry mechanism in TSM codebase). 
'''

import os

ROOT_DATASET = '../datasets/'  


def return_something(modality, raw_sample_rate=1):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-frames'
        if raw_sample_rate != 1:
            sr = int(raw_sample_rate)
            root_data = root_data + '-sr{}'.format(sr)
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
        label_map = None
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix, None


def return_somethingv2(modality, raw_sample_rate=1):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        if raw_sample_rate != 1:
            sr = int(raw_sample_rate)
            root_data = root_data + '-sr{}'.format(sr)
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
        label_map = 'something/v2/category.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix, label_map


def return_kinetics(modality, raw_sample_rate=1):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics400/frames'
        if raw_sample_rate != 1:
            sr = int(raw_sample_rate)
            root_data = root_data + '_sr{}'.format(sr)
        filename_imglist_train = 'kinetics400/labels/train_videofolder_cvdf.txt'
        filename_imglist_val = 'kinetics400/labels/val_videofolder_cvdf.txt'
        label_map = 'kinetics400/labels/kinetics_label_map.txt'
        prefix = '{:06d}.jpg' # for cvdfoundation data
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix, label_map

def return_mini_kinetics(modality, raw_sample_rate=1):
    filename_categories = 200 
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics400/frames_mini'
        if raw_sample_rate != 1:
            sr = int(raw_sample_rate)
            root_data = root_data + '_sr{}'.format(sr)
        filename_imglist_train = 'kinetics400/labels/train_videofolder_mini.txt'
        filename_imglist_val = 'kinetics400/labels/val_videofolder_mini.txt'
        label_map = 'kinetics400/labels/mini/mini-kinetics_label_map.txt'
        prefix = '{:06d}.jpg' # for cvdfoundation data
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix, label_map

def return_dataset(dataset, modality, raw_sample_rate=1):
    dict_single = {'something': return_something, 'somethingv2': return_somethingv2,
                   'mini-kinetics': return_mini_kinetics, 'kinetics': return_kinetics}
    
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix, label_map_file = dict_single[dataset](modality, raw_sample_rate)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    label_map_file = os.path.join(ROOT_DATASET, label_map_file)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix, label_map_file
