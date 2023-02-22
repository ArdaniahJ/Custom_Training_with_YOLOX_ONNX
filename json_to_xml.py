# Convert the json to xml document
with open('/content/Fashion_Dataset_Annotations.json') as f:
  data = json.load(f)

for i in range(len(data)):

  image_width = str(data[i]["width"])
  image_height = str(data[i]["height"])
  image_filename = str(data[i]["file_name"])

  boxes = []
  classes = []
  for j in range(len(data[i]['annotations'])):
    boxes.append([str(data[i]['annotations'][j]["bbox"]["xmin"]), str(data[i]['annotations'][j]["bbox"]['ymin']),
                 str(data[i]['annotations'][j]["bbox"]["xmax"]), str(data[i]['annotations'][j]["bbox"]["ymax"])])
    
    try:
      classes.append(data[i]['annotations'][j]["classes"][0])
    except:
      print("i got here")
      classes.append("Ripe")

  data_to_xml(image_width, image_height, image_filename, boxes, classes)