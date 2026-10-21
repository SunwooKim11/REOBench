# [REOBench repro] Setting A for Potsdam: the released config (configs/vit_base_win/SatMAE_Potsdam.py) unchanged except
#   - schedule = paper App.: 12 epochs, batch 8 (3,456 train patches -> 432 iter/epoch -> 5,184 iter)
#     (the released config inherits schedule_160k with samples_per_gpu=2)
#   - data paths -> released REOBench Potsdam zips, extracted
#   - evaluation on the clean test split at the end (corrupted splits: tools/test.py with --cfg-options)
_base_ = ['../vit_base_win/SatMAE_Potsdam.py']
ROOT = '/data/rs/reobench/Potsdam_extracted'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(img_dir=ROOT + '/images/train', ann_dir=ROOT + '/anns/train'),
    val=dict(img_dir=ROOT + '/test/clean', ann_dir=ROOT + '/test/clean_ann'),
    test=dict(img_dir=ROOT + '/test/clean', ann_dir=ROOT + '/test/clean_ann'))
runner = dict(type='IterBasedRunner', max_iters=5184)
checkpoint_config = dict(by_epoch=False, interval=5184)
evaluation = dict(interval=5184, metric='mIoU', pre_eval=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
