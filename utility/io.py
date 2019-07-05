import pandas as pd
import zipfile
import os
import shutil


# Read CSV file with pandas
def read_data(file_path):
    df = pd.read_csv(file_path)

    return df.values


# Unzip file
def unzip(file_path, save_path):
    zip_ref = zipfile.ZipFile(file=file_path, mode='r')
    zip_ref.extractall(save_path)
    zip_ref.close()
    return


# Zip files
def zip_files(file_paths, save_path):
    zip_ref = zipfile.ZipFile(file=save_path, mode='w')

    for file_path in file_paths:
        zip_ref.write(file_path, arcname=os.path.basename(file_path))

    zip_ref.close()
    return


# Zip folder
def zip_folder(folder_path, save_path):
    zip_ref = zipfile.ZipFile(file=save_path, mode='w')

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.normpath(file_path) != os.path.normpath(save_path):
                zip_ref.write(
                    filename=file_path,
                    arcname=os.path.join(os.path.basename(root), file)
                )

    zip_ref.close()
    return


# Copy files
def copy_files(file_paths, save_path):
    for file_path in file_paths:
        shutil.copyfile(file_path, os.path.join(save_path, os.path.basename(file_path)))
    return


# Delete files
def delete_files(file_paths):
    return


# Cut files
def cut_files(file_paths, save_path):
    return


def save_tf_record(save_path):
    return


import preprocessor
import utility
# zip_files([os.path.join(os.getcwd(), "io.py"),
#            os.path.join(os.getcwd(), "network.py"),
#            os.path.join(preprocessor.PATH, "__init__.py")],
#           save_path=os.path.join(preprocessor.PATH, "target.zip"))

# zip_files(["io.py",
#            "network.py"],
#           save_path=os.path.join(os.getcwd(), "target.zip"))

# zip_folder('D:/workspace/ANHTT/Sign Letter Recognition', save_path=os.path.join(os.getcwd(), "target.zip"))


# print(os.getcwd())