_base_ = [
    'coda_codetr_origin_dataset.py'
]
pretrained = 'models/swin_base_patch4_window12_384_22k.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        out_indices=(1, 2, 3),
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[96*2, 96*4, 96*8]))

# optimizer
optimizer = dict(weight_decay=0.05)
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)