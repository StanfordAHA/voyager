This folder provides scripts to generate training and inference data used by the gold model, System C, and Accelerator simultation.

## MobileBERT
To dump the input data for inference or training
```bash
python mobilebert/dump_model_and_dataset.py --model_name_or_path mobilebert/model_data/mobilebert-tiny-squad-bf16 --dataset squad --output_dir ../data/mobilebert-tiny-squad --dump_model --dump_dataset
```
