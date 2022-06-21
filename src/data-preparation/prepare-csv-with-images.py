import os

melanoma_source_folder = "/Users/yauhenbichel/Downloads/DermMel/DermMel/Melanoma"
notmelanoma_source_folder = "/Users/yauhenbichel/Downloads/DermMel/DermMel/NotMelanoma"
melanoma_file_path = "gs://YOUR-GCS-BUCKET/dataset_derm/Melanoma/"
not_melanoma_file_path = "gs://YOUR-GCS-BUCKET/dataset_derm/NotMelanoma/"

images_csv = open("images.csv", "w")

for (dirpath, dirnames, filenames) in os.walk(melanoma_source_folder):
    for file in filenames:
        images_csv.write(str(melanoma_file_path + file + ", Melanoma") + os.linesep)

for (dirpath, dirnames, filenames) in os.walk(notmelanoma_source_folder):
    for file in filenames:
        images_csv.write(str(not_melanoma_file_path + file + ", NotMelanoma") + os.linesep)
