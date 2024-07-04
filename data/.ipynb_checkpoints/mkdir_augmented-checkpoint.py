from os import getcwd, mkdir

current_path = getcwd() + r'\data\brain_tumor_dataset'

def mk_augmented_dir():
    try:
        mkdir(current_path + r'\train_augmented')
        print(r'train_augmented folder succesfully created.')
    except FileExistsError:
        print(r'train_augmented folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(current_path + r'\val_augmented')
        print(r'val_augmented folder succesfully created.')
    except FileExistsError:
        print(r'val_augmented folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(current_path + r'\train_augmented\yes')
        print(r'train_augmented\yes folder succesfully created.')
    except FileExistsError:
        print(r'train_augmented\yes folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(current_path + r'\train_augmented\no')
        print(r'train_augmented\no folder succesfully created.')
    except FileExistsError:
        print(r'train_augmented\no folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(current_path + r'\val_augmented\yes')
        print(r'val_augmented\yes folder succesfully created.')
    except FileExistsError:
        print(r'val_augmented\yes folder already exists.')
    except PermissionError:
        pass

    try:
        mkdir(current_path + r'\val_augmented\no')
        print(r'val_augmented\no folder succesfully created.')
    except FileExistsError:
        print(r'val_augmented\no folder already exists.')
    except PermissionError:
        pass

    return None