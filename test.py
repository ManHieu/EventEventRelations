if __name__ == '__main__':
    from data_loader.document_reader import tbd_tml_reader
    dir_name = ".\\datasets\\TimeBank-dense\\train\\"
    file_name = "ABC19980108.1830.0711.tml"
    my_dict = tbd_tml_reader(dir_name, file_name)
    print(my_dict)