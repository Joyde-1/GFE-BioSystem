# inference_gait.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown

def draw_keypoints(img, keypoints, radius=3, color=(0,255,0), thickness=-1):
    """
    Disegna ogni keypoint su img (x,y,v).
    """
    # for x, y, v in keypoints:
    #     if v > 0:
    #         cv2.circle(img, (int(x), int(y)), radius, color, thickness)
    for keypoints in keypoints:   
        # print("KEYPOINTS: ", keypoints)
        if keypoints[2] > 0:
            cv2.circle(img, (int(keypoints[0]), int(keypoints[1])), radius, color, thickness)

def main():
    # --- 1) Configura paths e device ---
    config_file     = '/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem/gait_keypoints_detection/config/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    checkpoint_file = '/Users/giovanni/Desktop/Tesi di Laurea/model_checkpoints/gait/best_coco_AP_epoch_90.pth'
    device          = 'mps'

    # directory di input / output
    imgs_dir    = '/Users/giovanni/Desktop/Tesi di Laurea/splitted_gait_keypoints_database/test/frames/'       # es. splitted_gait_keypoints_database/test/frames
    vis_dir     = '/Users/giovanni/Desktop/Tesi di Laurea/images/gait/keypoints_detection/'               # es. gait_inference_vis/
    os.makedirs(vis_dir, exist_ok=True)

    # --- 2) Inizializza il modello ---
    model = init_model(
        config_file, checkpoint_file, device=device,  
        cfg_options=dict(device=device)  # se serve iniettare via cfg‐options
    )

    # Lista dove mettere i risultati
    keypoints_sequence = []

    # --- 3) Itera sulle immagini ---
    for fn in tqdm(sorted(os.listdir(imgs_dir)), desc='Processing images'):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(imgs_dir, fn)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Top‐down: usa il bbox che copre tutta l’immagine letterbox (256×192)
        # Se le tue immagini sono già letterbox 256×192, puoi fare:
        bbox = np.array([[0, 0, w, h]], dtype=np.float32)

        # --- 4) Inferenza ---
        pose_results = inference_topdown(
            model,
            img,
            bboxes=bbox,
            # data_mode='topdown',
            bbox_format='xyxy',
            # return_heatmap=False
        )
        # pose_results è una lista di dict, per noi [0]['keypoints'] è Nx3

        # print("POSE RESULTS: ", pose_results)
        # print("POSE RESULTS[0]: ", pose_results[0])
        # print("POSE RESULTS[0].pred_instances: ", pose_results[0].pred_instances)
        # print("POSE RESULTS[0].pred_instances.keypoints: ", pose_results[0].pred_instances.keypoints)
        # print("POSE RESULTS[0].pred_instances.keypoints: ", pose_results[0].pred_instances.keypoints[0])

        # Estrai il primo (e unico) set di keypoints
        # kps = pose_results[0]['keypoints']  # numpy array (17,3)
        kps = pose_results[0].pred_instances.keypoints[0]  # numpy array (17,2)
        kps_scores = pose_results[0].pred_instances.keypoint_scores[0]  # numpy array (17,1)

        kp_xyv = []
        # print("KEYPOINTS: ", kps.tolist())

        for kp, kp_score in zip(kps.tolist(), kps_scores.tolist()):
            # print(f"Keypoint: {kp}, Score: {kp_score}")
            kp_xyv.append([kp[0], kp[1], kp_score])

        keypoints_sequence.append(kp_xyv)
        # print("KEYPOINTS SEQUENCE: ", keypoints_sequence)

        # --- 5) Visualizzazione e salvataggio ---
        vis_img = img.copy()
        draw_keypoints(vis_img, kp_xyv, radius=4, color=(0,255,0), thickness=-1)
        out_path = os.path.join(vis_dir, fn)
        cv2.imwrite(out_path, vis_img)

    # --- 6) Salva la sequenza di keypoint su disco ---
    np.savez(
        os.path.join(vis_dir, 'gait_keypoints_sequence.npz'),
        keypoints_sequence=keypoints_sequence
    )

    print(f"Fatte inferenze su {len(keypoints_sequence)} immagini.")
    print(f"Visualizzazioni salvate in {vis_dir}")
    print(f"Keypoint sequence salvata in {os.path.join(vis_dir,'gait_keypoints_sequence.npz')}")

if __name__ == '__main__':
    main()