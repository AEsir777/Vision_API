# returns the label/object
import io
import os
from google.cloud import vision
from google.cloud.vision_v1 import types

credential_path = "key.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = vision.ImageAnnotatorClient()


file_name = 'tomato.jpg'
folder_path = r'C:\Users\AEsir\Desktop\schoolwork\computer science\hackthon'


with io.open(os.path.join(folder_path,file_name),'rb') as image_file:

    content = image_file.read()
image = vision.Image(content=content)

objects = client.object_localization(image=image).localized_object_annotations

print('Number of objects found: {}'.format(len(objects)))
for object_ in objects:
    print('\n{} (confidence: {})'.format(object_.name, object_.score))
    print('Normalized bounding polygon vertices: ')
    for vertex in object_.bounding_poly.normalized_vertices:
        print(' - ({}, {})'.format(vertex.x, vertex.y))

response = client.label_detection(image=image)
labels = response.label_annotations
print('Labels:')

for label in labels:
    print(label.description)
