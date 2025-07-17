# Glo-In-One v2 Demo Results

## Successfully Processed

- 2 test images in 34 seconds (CPU)
- Generated 15-class segmentation masks
- Output: `test_1.png_preds_merged.npy` and `test_2.png_preds_merged.npy`

## Command

```bash
python testing_2D_patch_v2.py --valset_dir Test_Patch/data_list.csv --reload_path weights/model_checkpoint
```
