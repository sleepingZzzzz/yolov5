import xml.etree.ElementTree as ET
import os

# 你只想保留的类别名
target_class = "vehicle"  # 或你实际xml里的类别名

def convert(size, box):
    x_center = (box[0] + box[2]) / 2.0 / size[0]
    y_center = (box[1] + box[3]) / 2.0 / size[1]
    width = (box[2] - box[0]) / size[0]
    height = (box[3] - box[1]) / size[1]
    return [x_center, y_center, width, height]

def convert_annotation(xml_path, yolo_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    with open(yolo_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls != target_class:
                continue  # 跳过非目标类别
            xmlbox = obj.find('bndbox')
            b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
            bb = convert((w, h), b)
            out_file.write(f"0 {' '.join([str(a) for a in bb])}\n")  # YOLO类别编号为0

# 示例批量转换
xml_dir = "Annotations"
yolo_dir = "labels"
os.makedirs(yolo_dir, exist_ok=True)
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue
    xml_path = os.path.join(xml_dir, xml_file)
    yolo_path = os.path.join(yolo_dir, xml_file.replace(".xml", ".txt"))
    convert_annotation(xml_path, yolo_path)