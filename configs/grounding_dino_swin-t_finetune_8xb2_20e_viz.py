from os import getenv

_base_ = './grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = getenv('VIZWIZ_DATA_ROOT', 'data/vizwiz/')


class_name = ('local_newspaper', 'bank_statement', 'bills_or_receipt', 'business_card', 
           'condom_box', 'credit_or_debit_card', 'doctors_prescription', 'letters_with_address',
           'medical_record_document', 'pregnancy_test', 'empty_pill_bottle', 'tattoo_sleeve',
           'transcript', 'mortgage_or_investment_report', 'condom_with_plastic_bag', 'pregnancy_test_box')

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='augmented_images_v2/augmented_annotations.json',
        data_prefix=dict(img='augmented_images_v2/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='augmented_images_v2/augmented_annotations.json',
        data_prefix=dict(img='augmented_images_v2/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'augmented_images_v2/augmented_annotations.json')
test_evaluator = val_evaluator

max_epoch = 20



default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
