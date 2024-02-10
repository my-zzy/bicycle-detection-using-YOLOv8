import os


alpaca_id = '/m/0pcr'
bicycle_id = '/m/0199g'

train_bboxes_filename = "C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\prepare_data\\oidv6-train-annotations-bbox.csv"
# os.path.join('.', 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = "C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\prepare_data\\validation-annotations-bbox.csv"
test_bboxes_filename = "C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\prepare_data\\test-annotations-bbox.csv"

image_list_file_path = "C:\\Users\\zzy\\Desktop\\project\\train-yolov8\\prepare_data\\image_list_file.txt"

image_list_file_list = []
for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [bicycle_id] and id not in image_list_file_list:
                image_list_file_list.append(id)
                with open(image_list_file_path, 'a') as fw:
                    fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))
            line = f.readline()

        f.close()
