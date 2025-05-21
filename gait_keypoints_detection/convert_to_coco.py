import json
import os

def labelme_to_coco(labelme_json_path, image_id=1, annotation_id=1, category_id=1):
    # Carica il file LabelMe
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    # Estrai le informazioni dell'immagine
    file_name = os.path.basename(data['imagePath'])
    width = data['imageWidth']
    height = data['imageHeight']
    
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }
    
    # Definisci l'ordine dei keypoint che vuoi utilizzare
    # In questo esempio consideriamo quattro keypoint: "top", "bottom", "outer", "inner"
    keypoints_order = ['top', 'bottom', 'outer', 'inner']
    keypoints = []
    
    # Per ciascun keypoint, cerca la corrispondente annotazione in LabelMe
    for kp in keypoints_order:
        point = None
        for shape in data['shapes']:
            if shape['label'] == kp:
                # Presupponiamo che ogni shape contenga un solo punto
                point = shape['points'][0]
                break
        if point is None:
            raise ValueError(f"Annotazione mancante per il keypoint: {kp}")
        x, y = point
        # Aggiungi la tripla [x, y, v] (v = 2 per indicare che è visibile)
        keypoints.extend([x, y, 2])
    
    num_keypoints = len(keypoints_order)
    
    # Calcola una bounding box che racchiude tutti i keypoint
    xs = [shape['points'][0][0] for shape in data['shapes']]
    ys = [shape['points'][0][1] for shape in data['shapes']]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "bbox": [x_min, y_min, bbox_width, bbox_height],
        "area": bbox_width * bbox_height,
        "iscrowd": 0
    }
    
    return image_info, annotation_info

# Esempio di utilizzo:
if __name__ == "__main__":
    

    labelme_file = "/Users/giovanni/Desktop/Tesi di Laurea/database/annotations/ear_dx/1_1.json"  # Sostituisci con il percorso corretto
    image_info, annotation_info = labelme_to_coco(labelme_file)
    
    coco_dict = {
        # "info": {
        #     "description": "Dataset per il rilevamento dei landmark dell'orecchio",
        #     "version": "1.0",
        #     "year": 2023,
        #     "contributor": "Tuo Nome",
        #     "date_created": "2023-09-30"
        # },
        # "licenses": [
        #     {
        #         "id": 1,
        #         "name": "Attribution-NonCommercial",
        #         "url": "http://creativecommons.org/licenses/by-nc/4.0/"
        #     }
        # ],
        "images": [image_info],
        "annotations": [annotation_info],
        "categories": [
            {
                "id": 1,
                "name": "ear",
                "supercategory": "ear",
                "keypoints": ["top", "bottom", "left", "right"],
                "skeleton": [[1, 2], [2, 3], [3, 4]]
            }
        ]
    }
    
    output_file = "output_coco.json"
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    
    print(f"File COCO creato: {output_file}")







# import json
# import os

# def labelme_to_coco(labelme_json_path, image_id=1, annotation_id=1, category_id=1):
#     """
#     Converte un file LabelMe in formato COCO.
    
#     Parametri:
#       - labelme_json_path: percorso al file JSON di LabelMe
#       - image_id: ID da assegnare all'immagine
#       - annotation_id: ID da assegnare all'annotazione
#       - category_id: ID della categoria (per esempio 1 per "ear")
      
#     Ritorna:
#       Una tupla (image_info, annotation_info) che potrà essere usata per creare il file COCO.
#     """
#     with open(labelme_json_path, 'r') as f:
#         data = json.load(f)
    
#     # Estrai informazioni dell'immagine
#     file_name = os.path.basename(data["imagePath"])
#     width = data["imageWidth"]
#     height = data["imageHeight"]
    
#     image_info = {
#         "id": image_id,
#         "file_name": file_name,
#         "width": width,
#         "height": height,
#         "date_captured": "",
#         "license": 1,
#         "coco_url": "",
#         "flickr_url": ""
#     }
    
#     # Definisci l'ordine dei keypoint che vuoi annotare
#     keypoints_order = ["top", "bottom", "left", "right"]
#     keypoints = []
    
#     # Per ogni keypoint definito, cerca la corrispondente annotazione in "shapes"
#     for kp in keypoints_order:
#         point = None
#         for shape in data["shapes"]:
#             if shape["label"].lower() == kp.lower():
#                 # Assumiamo che ogni shape contenga un solo punto
#                 point = shape["points"][0]
#                 break
#         if point is None:
#             raise ValueError(f"Annotazione mancante per il keypoint: {kp}")
#         x, y = point
#         # Aggiungi la tripla [x, y, 2] (2 indica che il keypoint è visibile)
#         keypoints.extend([x, y, 2])
    
#     num_keypoints = len(keypoints_order)
    
#     # Calcola la bounding box che racchiude tutti i keypoint
#     xs = []
#     ys = []
#     for shape in data["shapes"]:
#         # Considera solo le annotazioni per i keypoint che ci interessano
#         if shape["label"].lower() in [k.lower() for k in keypoints_order]:
#             pt = shape["points"][0]
#             xs.append(pt[0])
#             ys.append(pt[1])
#     x_min = min(xs)
#     y_min = min(ys)
#     x_max = max(xs)
#     y_max = max(ys)
#     bbox_width = x_max - x_min
#     bbox_height = y_max - y_min

#     annotation_info = {
#         "id": annotation_id,
#         "image_id": image_id,
#         "category_id": category_id,
#         "keypoints": keypoints,
#         "num_keypoints": num_keypoints,
#         "bbox": [x_min, y_min, bbox_width, bbox_height],
#         "area": bbox_width * bbox_height,
#         "iscrowd": 0
#     }
    
#     return image_info, annotation_info

# if __name__ == "__main__":
#     # Specifica il percorso del file LabelMe
#     labelme_file = "path/to/your_labelme_annotation.json"  # Sostituisci con il percorso corretto
    
#     # Converte il file LabelMe in formato COCO
#     image_info, annotation_info = labelme_to_coco(labelme_file, image_id=1, annotation_id=1, category_id=1)
    
#     coco_output = {
#         "info": {
#             "description": "Dataset per il rilevamento dei landmark dell'orecchio",
#             "version": "1.0",
#             "year": 2023,
#             "contributor": "Tuo Nome",
#             "date_created": "2023-09-30"
#         },
#         "licenses": [
#             {
#                 "id": 1,
#                 "name": "Attribution-NonCommercial",
#                 "url": "http://creativecommons.org/licenses/by-nc/4.0/"
#             }
#         ],
#         "images": [image_info],
#         "annotations": [annotation_info],
#         "categories": [
#             {
#                 "id": 1,
#                 "name": "ear",
#                 "supercategory": "ear",
#                 "keypoints": ["top", "bottom", "left", "right"],
#                 "skeleton": [[1, 2], [2, 3], [3, 4]]
#             }
#         ]
#     }
    
#     output_path = "output_coco.json"
#     with open(output_path, "w") as outfile:
#         json.dump(coco_output, outfile, indent=4)
    
#     print(f"File COCO creato: {output_path}")