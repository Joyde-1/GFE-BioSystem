import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import EarPreProcessing
from yolo_detection.yolo_detection import Yolo
from post_processing.post_processing import EarPostProcessing
from features_extraction_classes.fisherfaces import FisherFaceExtractor
from metrics_classes.verification import Verification
from metrics_classes.recognition_closed_set import RecognitionClosedSet
from metrics_classes.recognition_open_set import RecognitionOpenSet
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

    if ear_config.features_extraction.fisherfaces.load_model:
        fisherface_extractor = FisherFaceExtractor(ear_config)

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
        
        # if ear_config.save_image.detected:
        #     save_image(ear_config, 'ear_dx', detected_image, image_name, 'detected_ear_dx_bounding_box')

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

        if ear_config.features_extractor == 'fisherface' and ear_config.features_extraction.fisherfaces.load_model:
            ear_template = fisherface_extractor.extract_fisherface(np.array(post_processed_ear_image))
            ear_template_vis = fisherface_extractor.extract_visual(ear_template, ear_config.post_processing.image_size, ear_config.post_processing.image_size)
        elif ear_config.features_extractor == 'fisherface' and not ear_config.features_extraction.fisherfaces.load_model:
            subjects[subject]['acquisition_name'].append(image_name)
            subjects[subject]['template'].append(post_processed_ear_image)       

            # Salva il checkpoint dopo ogni combinazione
            if ear_config.use_checkpoint:
                save_checkpoint('checkpoint_ear_dx.json', current_index + 1)

            continue
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if ear_config.save_image.features_extracted:
            save_image(ear_config, 'ear_dx', ear_template_vis, image_name, f"{ear_config.features_extractor}_ear_dx_image")
        
        subjects[subject]['acquisition_name'].append(image_name)
        subjects[subject]['template'].append(ear_template) 

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

    if ear_config.features_extractor == 'fisherface' and not ear_config.features_extraction.fisherfaces.load_model:
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

    # Verification phase
    ear_verification = Verification(ear_config, 'ear_dx')

    far, fa, t_imp, ms_far, thr_far = ear_verification.calculate_far(subjects)
    frr, fr, t_legit, ms_frr, thr_frr = ear_verification.calculate_frr(subjects)
    accuracy = ear_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Ear dx verification task metrics:", sep='\n')
    print(f"FAR: {far:.4f} %")
    print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    print(f"FRR: {frr:.4f} %")
    print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    print(f"Accuracy: {accuracy:.4f} %")

    ear_verification.calculate_roc_and_det(subjects)
    ear_verification.far_vs_frr(subjects)

    # Recognition closed-set phase
    ear_recognition_closed_set = RecognitionClosedSet(ear_config, 'ear_dx')

    rank1, rank5, mAP, t_ms, tps = ear_recognition_closed_set.evaluate_kfold(subjects, max_rank=20)

    print("", "Ear dx recognition (closed-set) task metrics:", sep='\n')
    print(f"Rank-1 medio: {rank1:.4f}%")
    print(f"Rank-5 medio: {rank5:.4f}%")
    print(f"mAP medio: {mAP:.4f}%")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    # Recognition open-set phase
    ear_recognition_open_set = RecognitionOpenSet(ear_config, 'ear_dx')

    fp, fn, t_ms, tps = ear_recognition_open_set.fpir_fnir(subjects, threshold=0.35)
    fpir, fnir, eer, eer_th = ear_recognition_open_set.fpir_fnir_curve(subjects)
    fpir_arr, fnir_arr, eer, eer_th = ear_recognition_open_set.det_curve(subjects)
    fpir_arr, fnir_arr, dir_arr = ear_recognition_open_set.dir_fpir_curve(subjects)

    print("", "Ear dx recognition (open-set) task metrics:", sep='\n')
    print(f"FPIR: {fp:.4f} %")
    print(f"FNIR: {fn:.4f} %")
    print(f"EER: {eer:.4f} %")
    print(f"threshold: {eer_th:.4f} %")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")
