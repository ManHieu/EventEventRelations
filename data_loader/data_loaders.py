from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
from sklearn.model_selection import train_test_split
from data_loader.document_reader import tsvx_reader, tml_reader, i2b2_xml_reader
from utils.tools import *
from itertools import combinations
import sys
import torch
import numpy as np
import tqdm
import random


class Reader(object):
    def __init__(self, typ):
        self.type = typ
    
    def format_reader(self, dir_name, file_name):
        if self.type == 'tsvx':
            return tsvx_reader(dir_name, file_name)
        elif self.type == 'tml':
            return tml_reader(dir_name, file_name)
        elif self.type == 'i2b2_xml':
            return i2b2_xml_reader(dir_name, file_name)

def loader(dir_name, typ):
    reader = Reader(typ)
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    corpus = []
    for file_name in tqdm.tqdm(onlyfiles):
        if typ == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.format_reader(dir_name, file_name)
                corpus.append(my_dict)
            else:
                pass
        else:
            my_dict = reader.format_reader(dir_name, file_name)
            corpus.append(my_dict)

    train_set, test_set = train_test_split(corpus, train_size=0.8, test_size=0.2)
    train_set, validate_set = train_test_split(train_set, train_size=0.75, test_size=0.25)
    return train_set, test_set, validate_set
    
def joint_constrained_loader(dataset, downsample, batch_size):
    train_set_HIEVE = []
    valid_set_HIEVE = []
    test_set_HIEVE = []
    train_set_MATRES = []
    valid_set_MATRES = []
    test_set_MATRES = []
    train_set_I2B2 = []
    valid_set_I2B2 = []
    test_set_I2B2 = []

    def get_data_train(my_dict):
        train_data = []
        eids = my_dict['event_dict'].keys()
        triple_events = list(combinations(eids, 3))
        for triple in triple_events:
            x, y, z = triple
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']
            z_sent_id = my_dict['event_dict'][z]['sent_id']

            x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
            y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])
            z_sent = padding(my_dict["sentences"][z_sent_id]["roberta_subword_to_ID"])

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]
            z_position = my_dict["event_dict"][z]["roberta_subword_id"]

            x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
            y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
            z_sent_pos = padding(my_dict["sentences"][z_sent_id]["roberta_subword_pos"], pos = True)

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            yz = my_dict["relation_dict"].get((y, z))
            zy = my_dict["relation_dict"].get((z, y))
            xz = my_dict["relation_dict"].get((x, z))
            zx = my_dict["relation_dict"].get((z, x))

            candidates = [[str(x), str(y), str(z), x_sent, y_sent, z_sent, x_position, y_position, z_position, x_sent_pos, y_sent_pos, z_sent_pos, xy, yz, xz],
                        [str(y), str(x), str(z), y_sent, x_sent, z_sent, y_position, x_position, z_position, y_sent_pos, x_sent_pos, z_sent_pos, yx, xz, yz],
                        [str(y), str(z), str(x), y_sent, z_sent, x_sent, y_position, z_position, x_position, y_sent_pos, z_sent_pos, x_sent_pos, yz, zx, yx],
                        [str(z), str(y), str(x), z_sent, y_sent, x_sent, z_position, y_position, x_position, z_sent_pos, y_sent_pos, x_sent_pos, zy, yx, zx],
                        [str(x), str(z), str(y), x_sent, z_sent, y_sent, x_position, z_position, y_position, x_sent_pos, z_sent_pos, y_sent_pos, xz, zy, xy],
                        [str(z), str(x), str(y), z_sent, x_sent, y_sent, z_position, x_position, y_position, z_sent_pos, x_sent_pos, y_sent_pos, zx, xy, zy]]
            for candidate in candidates:
                if None in candidate[-3:]:
                    candidates.remove(candidate)
            train_data.extend(candidates)
            return train_data
    
    def get_data_test(my_dict):
        test_data = []
        eids = my_dict['event_dict'].keys()
        pair_events = list(combinations(eids, 2))
        undersmp_ratio = 0.4
        for pair in pair_events:
            x, y = pair
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']

            x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
            y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]

            x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
            y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            candidates = [[str(x), str(y), str(x), x_sent, y_sent, x_sent, x_position, y_position, x_position, x_sent_pos, y_sent_pos, x_sent_pos, xy, xy, xy],
                        [str(y), str(x), str(y), y_sent, x_sent, y_sent, y_position, x_position, y_position, y_sent_pos, x_sent_pos, y_sent_pos, yx, yx, yx]]
            for candidate in candidates:
                if None in candidate[-3:]:
                    candidates.remove(candidate)
            test_data.extend(candidates)
        return test_data

    if dataset in ["HiEve", "Joint"]:
        # ========================
        #       HiEve Dataset
        # ========================
        dir_name = "./datasets/hievents_v2/processed/"
        train, test, validate = loader(dir_name, 'tsvx')
        undersmp_ratio = 0.4
        for my_dict in train:
            train_data = get_data_train(my_dict)
            for item in train_data:
                if item[-3]==3 and item[-2]==3:
                    pass
                elif None in item[-3:]:
                    if random.uniform(0, 1) < downsample:
                        item.append(0) # 0 is HiEve
                        train_set_HIEVE.append(item)
                else:
                    item.append(0)
                    train_set_HIEVE.append(item)
        
        for my_dict in test:
            test_data = get_data_test(my_dict)
            for item in test_data:
                if item[-3]==3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        item.append(0)
                        test_set_HIEVE.append(item)
                else:
                    item.append(0)
                    test_set_HIEVE.append(item)
        
        for my_dict in validate:
            valid_data = get_data_test(my_dict)
            for item in valid_data:
                if item[-3]==3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        item.append(0)
                        valid_set_HIEVE.append(item)
                else:
                    item.append(0)
                    valid_set_HIEVE.append(item)
    
    if dataset in ["MATRES", "Joint"]:
        # ========================
        #       MATRES Dataset
        # ========================
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        train = []
        test = []
        validate = []
        aquaint = loader(aquaint_dir_name, 'tml')
        timebank = loader(timebank_dir_name, 'tml')
        platinum = loader(platinum_dir_name, 'tml')
        for subset in aquaint:
            validate.extend(subset)
        for subset in timebank:
            train.extend(subset)
        for subset in platinum:
            test.extend(subset)
        
        for my_dict in train:
            train_data = get_data_train(my_dict)
            for item in train_data:
                item.append(1) # 1 is MATRES
                train_set_MATRES.append(item)

        for my_dict in test:
            test_data = get_data_test(my_dict)
            for item in test_data:
                item.append(1) # 1 is MATRES
                test_set_MATRES.append(item)
        
        for my_dict in validate:
            validate_data = get_data_train(my_dict)
            for item in validate_data:
                item.append(1) # 1 is MATRES
                valid_set_MATRES.append(item)

    if dataset in ["I2B2", "Joint"]:
        # ========================
        #       I2B2 Dataset
        # ========================
        dir_name = "datasets/i2b2_2012/2012-07-15.original-annotation.release"
        train, test, validate = loader(dir_name, 'i2b2_xml')
        
        for my_dict in train:
            train_data = get_data_train(my_dict)
            for item in train_data:
                item.append(2) # 2 is I2B2
                train_set_I2B2.append(item)

        for my_dict in test:
            test_data = get_data_test(my_dict)
            for item in test_data:
                item.append(2) # 2 is I2B2
                test_set_I2B2.append(item)
        
        for my_dict in validate:
            validate_data = get_data_train(my_dict)
            for item in validate_data:
                item.append(2) # 2 is I2B2
                valid_set_I2B2.append(item)
        
    # ==============================================================
    #      Use DataLoader to convert to Pytorch acceptable form
    # ==============================================================
    if dataset == "MATRES":
        num_classes = 4
        train_dataloader_MATRES = DataLoader(EventDataset(train_set_MATRES), batch_size=batch_size, shuffle = True)
        valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
        test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True) 
        return train_dataloader_MATRES, valid_dataloader_MATRES, test_dataloader_MATRES, None, None, None, None, num_classes 
    elif dataset == "HiEve":
        num_classes = 4
        train_dataloader_HIEVE = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
        valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle = True)    
        test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True) 
        return train_dataloader_HIEVE, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, None, None, num_classes
    if dataset == "I2B2":
        num_classes = 3
        train_dataloader_I2B2 = DataLoader(EventDataset(train_set_I2B2), batch_size=batch_size, shuffle = True)
        valid_dataloader_I2B2 = DataLoader(EventDataset(valid_set_I2B2), batch_size=batch_size, shuffle = True)    
        test_dataloader_I2B2 = DataLoader(EventDataset(test_set_I2B2), batch_size=batch_size, shuffle = True) 
        return train_dataloader_I2B2, , None, None, None, None, valid_dataloader_I2B2, test_dataloader_I2B2, num_classes 
    elif dataset == "Joint":
        num_classes = 8
        train_set_HIEVE.extend(train_set_MATRES)
        train_set_HIEVE.extend(train_set_I2B2)
        train_dataloader = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
        valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
        test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True)
        valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle = True)    
        test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True)
        valid_dataloader_I2B2 = DataLoader(EventDataset(valid_set_I2B2), batch_size=batch_size, shuffle = True)    
        test_dataloader_I2B2 = DataLoader(EventDataset(test_set_I2B2), batch_size=batch_size, shuffle = True) 
        return train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, valid_dataloader_I2B2, test_dataloader_I2B2, num_classes
    else:
        raise ValueError("Currently not supporting this dataset! -_-'")
    

    


