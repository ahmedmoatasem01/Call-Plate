from sklearn.model_selection import train_test_split
import cv2
import os
import yaml


#################################################################
### Split the Data
#################################################################

root_dir = "datasets/car-number-plate/"
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

def file_paths(root, valid_formats):
    "get the full path to each image/label in the dataset"
    file_paths = []

    # loop over the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # loop over the filenames in the current directory
        for filename in filenames:
            # extract the file extension from the filename
            extension = os.path.splitext(filename)[1].lower()

            # if the filename has a valid extension we build the full 
            # path to the file and append it to our list
            if extension in valid_formats:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

    return file_paths


image_paths = file_paths(root_dir + "images", valid_formats[:3])
label_paths = file_paths(root_dir + "labels", valid_formats[-1])

# split the data into training, validation and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)


def write_to_file(images_path, labels_path, X):
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for img_path in X:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1]
        image = cv2.imread(img_path)
        cv2.imwrite(os.path.join(images_path, f"{img_name}{img_ext}"), image)

        # Dynamically get label path from image path
        label_file_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(label_file_path):
            with open(os.path.join(labels_path, f"{img_name}.txt"), "w") as f:
                with open(label_file_path, "r") as label_file:
                    f.write(label_file.read())
        else:
            print(f"[WARNING] Label not found for image: {img_path}")

write_to_file("datasets/images/train", "datasets/labels/train", X_train)
write_to_file("datasets/images/valid", "datasets/labels/valid", X_val)
write_to_file("datasets/images/test", "datasets/labels/test", X_test)


#################################################################
### Create a YAML file
#################################################################

# Create a dictionary with the paths to the train, valid, and test sets
data = {
    "path": "../datasets", # dataset root dir (you can also use the full path to the `datasets` folder)
    "train": "images/train", # train images (relative to 'path')
    "val": "images/valid", # val images (relative to 'path')
    "test": "images/test", # test images (optional)

    # Classes
    "names":["number plate"]
}

# write the dictionary to a YAML file
with open("number-plate.yaml", "w") as f:
    yaml.dump(data, f)
