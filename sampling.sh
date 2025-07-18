#["rtts", "urhi", 'etc', 'custom']
python ./dehaze_sampling.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.data.dataset "custom" \
    --config.data.test_data_root "datasets/custom" \
    --config.flow.refine_t True \
    --config.sampling.ckpt 'hazeflow.pth' \
    --config.work_dir 'samples/' \
    --config.expr 1step \
    --config.sampling.sample_N 1 \