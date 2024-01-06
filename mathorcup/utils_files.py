import os
from pathlib import Path
import xml.etree.ElementTree as ET

"""
return PATH
"""


def get_current_path():
    return Path.cwd()


def get_file_list(directory):
    file_list = []
    for path in directory.glob("**/*"):
        if path.is_file():
            file_list.append(path)
    return file_list


def get_files_with_extension(directory, extension):
    path = Path(directory)
    files = list(path.glob(f"*.{extension}"))
    return files


def conv_xml2txt(annotation_path) -> list:
    """
    XML标注转归一化TXT标注
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    annotations = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        # 计算标注框的中心点和宽高的归一化值
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # 构建转换后的标注字符串
        annotation = f"{obj.find('name').text} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        annotations.append(annotation)

    return annotations


def get_xml_annotated_info(annotation_path) -> dict:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # 获取图像大小和矩形
    info = {
        "width": int(root.find(".//width").text),
        "height": int(root.find(".//height").text),
        "xmin": int(root.find(".//xmin").text),
        "ymin": int(root.find(".//ymin").text),
        "xmax": int(root.find(".//xmax").text),
        "ymax": int(root.find(".//ymax").text),
    }

    return info
