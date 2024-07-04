from os import getcwd, mkdir

current_path = getcwd()

def mk_augmented_dir():
    try:
        mkdir(data_path + r'\train_augmented')
        print('train_augmented folder succesfully created.')
    except FileExistsError:
        print('train_augmented folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(data_path + r'\val_augmented')
        print('val_augmented folder succesfully created.')
    except FileExistsError:
        print('val_augmented folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(data_path + r'\train_augmented\yes')
        print('train_augmented\yes folder succesfully created.')
    except FileExistsError:
        print('train_augmented\yes folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(data_path + r'\train_augmented\no')
        print('train_augmented\no folder succesfully created.')
    except FileExistsError:
        print('train_augmented\no folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(data_path + r'\val_augmented\yes')
        print('val_augmented\yes folder succesfully created.')
    except FileExistsError:
        print('val_augmented\yes folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(data_path + r'\val_augmented\no')
        print('val_augmented\no folder succesfully created.')
    except FileExistsError:
        print('val_augmented\no folder already exists.')
    except PermissionError:
        pass