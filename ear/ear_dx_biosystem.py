import sys
import os

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import EarPreProcessing
from yolo_detection.yolo_detection import Yolo
from post_processing.post_processing import EarPostProcessing
from features_extraction_classes.fisherfaces import FisherFaceExtractor
from matching_classes.matching import Matching
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint


if __name__ == '__main__':
    ear_config = load_config('config/ear_dx_config.yaml')

    if ear_config.browse_path:
        ear_config.data_dir = browse_path('Select the database folder')
        ear_config.save_path = browse_path('Select the folder where images and plots will be saved')

        if ear_config.detector == 'yolo':
            ear_config.yolo.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')

        if ear_config.save_groundtruths:
            ear_config.groundtruths_path = browse_path('Select the folder where groundtruths will be saved')

    images_data = LoadData()
    images, image_names, image_paths = images_data.load_images(ear_config, 'ear_dx')

    pre_processing = EarPreProcessing(ear_config)

    yolo = Yolo(ear_config, 'ear_dx')

    post_processing = EarPostProcessing(ear_config)

    if ear_config.use_checkpoint:
        checkpoint = load_checkpoint('checkpoint_ear_dx.json')
        start_index = checkpoint['current_index'] if checkpoint else 0

    subjects = {}

    max_width = 0
    max_height = 0

    widths = []
    heights = []

    post_processed_ear_images = []

    for current_index, (image, image_name, image_path) in enumerate(zip(images, image_names, image_paths)):
        # Pre-Processing and Segmentation phases
        # -----------------------------------------------------------
        
        # Extract the subject number from the image name
        subject = image_name.split('_')[0]

        print("Image name:", image_name)

        if subject not in subjects:
            subjects[subject] = {
                'acquisition_name': [], 
                'template': []
            }

        pre_processed_ear_image = pre_processing.pre_processing_image(image.copy())

        if ear_config.save_image.pre_processed:
            save_image(ear_config, 'ear_dx', pre_processed_ear_image, image_name, 'pre_processed_ear_dx_image')

        if ear_config.detector == 'yolo':
            pre_processed_ear_image_path = path_extractor(ear_config, 'ear_dx', image_name, 'pre_processed_ear_dx_image')
            detected_image, bounding_box = yolo.predict_bounding_box(pre_processed_ear_image_path)
        elif ear_config.detector == 'None':
            bounding_box = [0, 0, 1, 1]
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if ear_config.save_image.detected:
            save_image(ear_config, 'ear_dx', detected_image, image_name, 'detected_ear_dx_bounding_box')

        post_processed_ear_image, shape = post_processing.post_processing_image(pre_processed_ear_image.copy(), bounding_box)

        post_processed_ear_images.append(post_processed_ear_image)

        widths.append(shape[1])
        heights.append(shape[0])

        if max_width < shape[1]:
            max_width = shape[1]
        
        if max_height < shape[0]:
            max_height = shape[0]

        if ear_config.save_image.post_processed:
            save_image(ear_config, 'ear_dx', post_processed_ear_image, image_name, 'post_processed_ear_dx_image')

        if ear_config.features_extractor == 'fisherface':
            subjects[subject]['acquisition_name'].append(image_name)
            subjects[subject]['template'].append(post_processed_ear_image)       
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        # Salva il checkpoint dopo ogni combinazione
        if ear_config.use_checkpoint:
            save_checkpoint('checkpoint_ear_dx.json', current_index + 1)

    # print("Max width: ", max_width)
    # print("Max height: ", max_height)

    # mean_width = int(statistics.mean(widths))
    # mean_height = int(statistics.mean(heights))

    # print("Max width: ", mean_width)
    # print("Max height: ", mean_height)

    # padded_ear_images = []

    # for post_processed_ear_image in post_processed_ear_images:
    #     # padded_ear_images.append(post_processing.add_padding(post_processed_ear_image, max_width, max_height))
    #     # padded_ear_images.append(post_processing.resize_image(post_processed_ear_image, mean_width, mean_height))
    #     padded_ear_images.append(post_processing.resize_image(post_processed_ear_image, 512, 512))

    # # Itera su subjects
    # padded_ear_images_index = 0  # Indice per tracciare la posizione nei processed ear images

    # for subject in subjects.keys():
    #     subjects[subject]['template'] = []

    #     num_acquisitions = len(subjects[subject]['acquisition_name'])

    #     # Aggiungi le padded images al soggetto a blocchi di num_acquisitions
    #     subjects[subject]['template'].extend(
    #         padded_ear_images[padded_ear_images_index:padded_ear_images_index + num_acquisitions]
    #     )

    #     if ear_config.save_image.padded:
    #         for i in range(num_acquisitions):
    #             save_image(ear_config, 'ear_dx', padded_ear_images[i + padded_ear_images_index], f"{subject}_{i + 1}", "padded_ear_dx_image")

    #     # Aggiorna l'indice per il prossimo soggetto
    #     padded_ear_images_index += num_acquisitions

    if ear_config.features_extractor == 'fisherface':
        fisherface_extractor = FisherFaceExtractor(ear_config)

        # fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects, max_width, max_height)
        # fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects, mean_width, mean_height)
        fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects, ear_config.post_processing.image_size, ear_config.post_processing.image_size)

        # Itera su subjects e assegna i templates (fisherfaces)
        fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces

        for subject in subjects.keys():
            subjects[subject]['template'] = []

            num_acquisitions = len(subjects[subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects[subject]['template'].extend(
                fisherfaces[fisherfaces_index:fisherfaces_index + num_acquisitions]
            )

            if ear_config.save_image.features_extracted:
                for i in range(num_acquisitions):
                    save_image(ear_config, 'ear_dx', visual_fisherfaces[i + fisherfaces_index], f"{subject}_{i + 1}", f"{ear_config.features_extractor}_ear_dx_image")

            # Aggiorna l'indice per il prossimo soggetto
            fisherfaces_index += num_acquisitions
    
    ear_matching = Matching(ear_config, 'ear_dx')

    far, fa, t_imp = ear_matching.calculate_far(subjects)
    frr, fr, t_legit = ear_matching.calculate_frr(subjects)
    accuracy = ear_matching.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Matching ear dx metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    ear_matching.calculate_roc_and_det(subjects)
    ear_matching.far_vs_frr(subjects)