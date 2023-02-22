import json
import xml.etree.ElementTree as ET

def data_to_xml(image_width, image_height, image_filename, boxes, classes):
  annotation = ET.Element('annotation')

  filename = ET.SubElement(annotation, 'filename')
  filename.text = image_filename

  size = ET.SubElement(annotation, 'size')

  width = ET.SubElement(size, 'width')
  width.text = image_width

  height = ET.SubElement(size, 'height')
  height.text = image_height

  depth = ET.SubElement(size, 'depth')
  depth.text = '3'

  for i in range(len(classes)):

    object = ET.SubElement(annotation, 'object')

    name = ET.SubElement(object, 'name')
    name.text = classes[i].lower()

    bndbox = ET.SubElement(object, 'bndbox')

    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = boxes[i][0]
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = boxes[i][1]

    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = boxes[i][2]
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = boxes[i][3]


  b_xml = ET.tostring(annotation)

  image_filename = image_filename[:-4]
  with open("annotations/" + image_filename + ".xml", "wb") as f:
    f.write(b_xml)
  return None


