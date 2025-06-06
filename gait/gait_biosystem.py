import sys
import os

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import GaitPreProcessing
from pre_processing.gait_keypoints_detection import GaitKeypointsDetection
from yolo_pose_detection.yolo_pose_detection import YoloPose
from features_extraction.gait_embedding_extraction import GaitEmbeddingExtraction
from matching_classes.matching import Matching
from utils import load_config, browse_path, save_image, path_extractor, load_checkpoint, save_checkpoint


if __name__ == '__main__':
    gait_config = load_config('config/gait_config.yaml')

    if gait_config.browse_path:
        gait_config.data_dir = browse_path('Select the database folder')
        gait_config.save_dir = browse_path('Select the folder where images and plots will be saved')

    frames_data = LoadData()
    frame_sequences, frame_sequences_names, frame_sequences_paths = frames_data.load_frames(gait_config, 'gait')

    pre_processing = GaitPreProcessing(gait_config)

    gait_keypoints_detector = GaitKeypointsDetection(gait_config)

    gait_yolo_pose_detector = YoloPose(gait_config, 'gait')

    gait_embedding_extractor = GaitEmbeddingExtraction(gait_config)

    # matching = Matching(gait_config)

    if gait_config.use_checkpoint:
        checkpoint = load_checkpoint('checkpoint_gait.json')
        start_index = checkpoint['current_index'] if checkpoint else 0

    subjects = {}

    for current_index, (frame_sequence, frame_sequence_name, frame_sequence_path) in enumerate(zip(frame_sequences, frame_sequences_names, frame_sequences_paths)):
        # Pre-Processing and Segmentation phases
        # -----------------------------------------------------------
        
        # Extract the subject number from the image name
        subject = frame_sequence_name.split('_')[0]

        print("Frame sequence name:", frame_sequence_name)

        if subject not in subjects:
            subjects[subject] = {
                'acquisition_name': [], 
                'template': []
            }

        # print("frame_sequence:", len(frame_sequence))
        # print("frame_sequence_path:", len(frame_sequence_path))
        # print("subject:", subject)

        pre_processed_frame_sequence = []

        for i, frame in enumerate(frame_sequence):
            # pre_processed_frame, pre_processed_frame_to_save = pre_processing.pre_processing_frame(frame.copy())
            pre_processed_frame = pre_processing.pre_processing_frame(frame.copy())

            if gait_config.save_image.pre_processed:    # TODO: da rimuovere l'if se necessariamente deve esserci l'immagine salvata
                # save_image(gait_config, 'gait', pre_processed_frame_to_save, frame_sequence_name, 'pre_processed_frame_sequence', i + 1)
                save_image(gait_config, 'gait', pre_processed_frame, frame_sequence_name, 'pre_processed_frame_sequence', i + 1)

            pre_processed_frame_sequence.append(pre_processed_frame)
        
        keypoints_sequence = []

        # Gait Keypoints Detection phase
        # -----------------------------------------------------------
        for i, pre_processed_frame in enumerate(pre_processed_frame_sequence):
            # keypoints, frame_with_detected_keypoints = gait_keypoints_detector.detect_keypoints(pre_processed_frame)

            pre_processed_frame_path = path_extractor(gait_config, 'gait', frame_sequence_name, 'pre_processed_frame_sequence', i + 1)
            keypoints, frame_with_detected_keypoints = gait_yolo_pose_detector.predict_bounding_box(pre_processed_frame_path)

            if gait_config.save_image.detected_keypoints:
                save_image(gait_config, 'gait', frame_with_detected_keypoints, frame_sequence_name, 'detected_frame_keypoints_sequence', i + 1)

            keypoints_sequence.append(keypoints)

        keypoints_sequence = pre_processing.pre_processing_keypoints(keypoints_sequence)

        gait_embedding = gait_embedding_extractor.extract_embedding(keypoints_sequence)

        subjects[subject]['acquisition_name'].append(frame_sequence_name)
        subjects[subject]['template'].append(gait_embedding)

        # Salva il checkpoint dopo ogni combinazione
        if gait_config.use_checkpoint:
            save_checkpoint('checkpoint_face.json', current_index + 1)
    
    matching_gait = Matching(gait_config, 'gait')

    far, fa, t_imp = matching_gait.calculate_far(subjects)
    frr, fr, t_legit = matching_gait.calculate_frr(subjects)
    accuracy = matching_gait.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Matching face metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_gait.calculate_roc_and_det(subjects)
    matching_gait.far_vs_frr(subjects)