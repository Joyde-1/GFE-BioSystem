# configs/ear_landmark_config.py

import mmpose.visualization

_base_ = ['/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem/mmpose/configs/_base_/default_runtime.py']

visualizer = dict(type='DefaultVisualizer')

device = 'mps'

# Tipo di dataset e percorso base del dataset
dataset_type = 'COCOKeypoints'
data_root = '/Users/giovanni/Desktop/Tesi di Laurea/splitted_gait_keypoints_database/ear_dx/'

# Parametri di normalizzazione (i valori sono quelli standard di ImageNet)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Pipeline di preprocessing per il training
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=['image_file', 'center', 'scale', 'rotation']
    )
]

# Pipeline per validazione e test
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation']
    )
]

data = dict(
    samples_per_gpu=8,  # Batch size per GPU
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labels/',  # File di annotazioni training
        img_prefix=data_root + 'train/images/',                   # Cartella immagini training
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/',
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/',
        pipeline=val_pipeline
    )
)

# Configurazione del modello TopDown
model = dict(
    type='TopDown',
    pretrained='open-mmlab://msra/hrnetv2_w32',  # Backbone pre-addestrato
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256))
        )
    ),
    head=dict(
        type='TopDownHeatmapHead',
        in_channels=32,     # Questo valore dipende dall'output del backbone
        out_channels=4,     # Numero di keypoint (es. top, bottom, outer, inner)
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1)
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=False,
        modulate_kernel=11
    )
)

# Configurazione dell'ottimizzatore e dello scheduler
optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[90, 120]
)
total_epochs = 140

evaluation = dict(
    interval=10,
    metric=['PCK'],  # Puoi usare Percentage of Correct Keypoints come metrica
    save_best='PCK'
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)

# Directory in cui salvare i checkpoint e i log
work_dir = './work_dirs/ear_landmark'
checkpoint_config = dict(interval=10, max_keep_ckpts=3)

# ... (il resto della configurazione, incluse le definizioni di work_dir, checkpoint_config, ecc.)

# Definizione dei dataloader
train_dataloader = dict(
    samples_per_gpu=data.get('samples_per_gpu', 8),
    workers_per_gpu=data.get('workers_per_gpu', 2),
    dataset=data['train']
)

val_dataloader = dict(
    samples_per_gpu=data.get('samples_per_gpu', 8),
    workers_per_gpu=data.get('workers_per_gpu', 2),
    dataset=data['val']
)

test_dataloader = dict(
    samples_per_gpu=data.get('samples_per_gpu', 8),
    workers_per_gpu=data.get('workers_per_gpu', 2),
    dataset=data['test']
)

# Definizione dell'ottimizzatore in un wrapper
optim_wrapper = dict(
    optimizer=optimizer
)

# Aggiungi queste righe per definire val_cfg e val_evaluator
val_cfg = dict()
val_evaluator = evaluation

test_cfg = dict()
test_evaluator = evaluation