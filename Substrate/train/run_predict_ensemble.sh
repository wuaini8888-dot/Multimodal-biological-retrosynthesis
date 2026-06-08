python predict_ensemble.py \
  --config /data/stu1/ml_project/bioec_retro1/FF/Finger3/train_stage2.yaml \
  --checkpoints \
    /data/stu1/ml_project/bioec_retro1/FF/Finger3/save/finetune/20260402_step_200000.pt \
    /data/stu1/ml_project/bioec_retro1/train1/save2/finetune_seed2/model_step_200000.pt \
    /data/stu1/ml_project/bioec_retro1/train1/save3/finetune_seed3/model_step_200000.pt \
    /data/stu1/ml_project/bioec_retro1/train1/save4/finetune_seed4/model_step_200000.pt \
  --input /data/stu1/ml_project/bioec_retro1/dataset/data_processed/tokenized/tokenized_src_test.txt \
  --output /data/stu1/ml_project/bioec_retro1/train1/prediction.txt