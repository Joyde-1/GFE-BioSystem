import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import GaitPreProcessing
# from pre_processing.gait_keypoints_detection import GaitKeypointsDetection
from yolo_pose_detection.yolo_pose_detection import YoloPose
from features_extraction.gait_embedding_extraction import GaitEmbeddingExtraction
from metrics_classes.verification import Verification
from metrics_classes.recognition_closed_set import RecognitionClosedSet
from metrics_classes.recognition_open_set import RecognitionOpenSet
from utils import load_config, browse_path, save_image, path_extractor, load_checkpoint, save_checkpoint


if __name__ == '__main__':
    gait_config = load_config('config/gait_config.yaml')

    if gait_config.browse_path:
        gait_config.data_dir = browse_path('Select the database folder')
        gait_config.save_dir = browse_path('Select the folder where images and plots will be saved')

    frames_data = LoadData()
    frame_sequences, frame_sequences_names, frame_sequences_paths, all_subject_ids, all_sequence_names, all_frame_names = frames_data.load_frames(gait_config, 'gait')

    pre_processing = GaitPreProcessing(gait_config)

    # gait_keypoints_detector = GaitKeypointsDetection(gait_config)

    gait_yolo_pose_detector = YoloPose(gait_config, 'gait')

    gait_embedding_extractor = GaitEmbeddingExtraction(gait_config)

    if gait_config.use_checkpoint:
        checkpoint = load_checkpoint('checkpoint_gait.json')
        start_index = checkpoint['current_index'] if checkpoint else 0

    subjects = {}

    # for current_index, (frame_sequence, frame_sequence_name, frame_sequence_path) in enumerate(zip(frame_sequences, frame_sequences_names, frame_sequences_paths)):
    for current_index, (frame_sequence, frame_sequence_name, frame_sequence_path, subject_ids, sequence_names, frame_names) in enumerate(zip(frame_sequences, frame_sequences_names, frame_sequences_paths, all_subject_ids, all_sequence_names, all_frame_names)):
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
        pre_processing_params_sequence = []

        for i, frame in enumerate(frame_sequence):
            # pre_processed_frame, pre_processed_frame_to_save = pre_processing.pre_processing_frame(frame.copy())
            pre_processed_frame, pre_processing_params = pre_processing.pre_processing_frame(frame.copy())

            if gait_config.save_image.pre_processed:
                # save_image(gait_config, 'gait', pre_processed_frame_to_save, frame_sequence_name, 'pre_processed_frame_sequence', i + 1)
                save_image(gait_config, 'gait', pre_processed_frame, frame_sequence_name, 'pre_processed_frame_sequence', i + 1)

            pre_processed_frame_sequence.append(pre_processed_frame)
            pre_processing_params_sequence.append(pre_processing_params)
        
        keypoints_sequence = []

        # Gait Keypoints Detection phase
        # -----------------------------------------------------------
        for i, (pre_processed_frame, pre_processing_params, frame, subject_id, sequence_name, frame_name) in enumerate(zip(pre_processed_frame_sequence, pre_processing_params_sequence, frame_sequence, subject_ids, sequence_names, frame_names)):
        # for i, (pre_processed_frame, pre_processing_params) in enumerate(zip(pre_processed_frame_sequence, pre_processing_params_sequence)):
            # keypoints, frame_with_detected_keypoints = gait_keypoints_detector.detect_keypoints(pre_processed_frame)

            pre_processed_frame_path = path_extractor(gait_config, 'gait', frame_sequence_name, 'pre_processed_frame_sequence', i + 1)
            keypoints, frame_with_detected_keypoints = gait_yolo_pose_detector.predict_keypoints(pre_processed_frame_path)

            if gait_config.save_image.detected_keypoints:
                save_image(gait_config, 'gait', frame_with_detected_keypoints, frame_sequence_name, 'detected_frame_keypoints_sequence', i + 1)

            pre_processed_keypoints = pre_processing.pre_processing_keypoints(keypoints, pre_processing_params)

            # Visualize keypoints on the original frame
            frame_with_pre_processed_keypoints = pre_processing.visualize_keypoints(pre_processed_keypoints, frame)  

            if gait_config.save_image.post_processed:
                save_image(gait_config, 'gait', frame_with_pre_processed_keypoints, frame_sequence_name, 'post_processed_frame_with_keypoints', i + 1)

            if gait_config.save_detected_keypoints:
                pre_processing.save_keypoints(pre_processed_keypoints, subject_id, sequence_name, frame_name)

            keypoints_sequence.append(pre_processed_keypoints)

        processed_keypoints_sequence = pre_processing.pre_processing_keypoints_sequence(keypoints_sequence)
        
        gait_embedding = gait_embedding_extractor.extract_embedding(processed_keypoints_sequence)

        subjects[subject]['acquisition_name'].append(frame_sequence_name)
        subjects[subject]['template'].append(gait_embedding)

        # Salva il checkpoint dopo ogni combinazione
        if gait_config.use_checkpoint:
            save_checkpoint('checkpoint_gait.json', current_index + 1)

    # Verification phase
    gait_verification = Verification(gait_config, 'gait')

    far, fa, t_imp, ms_far, thr_far = gait_verification.calculate_far(subjects)
    frr, fr, t_legit, ms_frr, thr_frr = gait_verification.calculate_frr(subjects)
    accuracy = gait_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Gait verification task metrics:", sep='\n')
    print(f"FAR: {far:.4f} %")
    print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    print(f"FRR: {frr:.4f} %")
    print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    print(f"Accuracy: {accuracy:.4f} %")

    gait_verification.calculate_roc_and_det(subjects)
    gait_verification.far_vs_frr(subjects)

    # Recognition closed-set phase
    gait_recognition_closed_set = RecognitionClosedSet(gait_config, 'gait')

    rank1, rank5, mAP, t_ms, tps = gait_recognition_closed_set.evaluate_kfold(subjects, max_rank=20)

    print("", "Gait recognition (closed-set) task metrics:", sep='\n')
    print(f"Rank-1 medio: {rank1:.4f}%")
    print(f"Rank-5 medio: {rank5:.4f}%")
    print(f"mAP medio: {mAP:.4f}%")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    # Recognition open-set phase
    gait_recognition_open_set = RecognitionOpenSet(gait_config, 'gait')

    fp, fn, t_ms, tps = gait_recognition_open_set.fpir_fnir(subjects, threshold=0.35)
    fpir, fnir, eer, eer_th = gait_recognition_open_set.fpir_fnir_curve(subjects)
    fpir_arr, fnir_arr, eer, eer_th = gait_recognition_open_set.det_curve(subjects)
    fpir_arr, fnir_arr, dir_arr = gait_recognition_open_set.dir_fpir_curve(subjects)

    print("", "Gait recognition (open-set) task metrics:", sep='\n')
    print(f"FPIR: {fp:.4f} %")
    print(f"FNIR: {fn:.4f} %")
    print(f"EER: {eer:.4f} %")
    print(f"threshold: {eer_th:.4f} %")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")
