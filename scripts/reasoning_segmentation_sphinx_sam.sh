export CUDA_VISIBLE_DEVICES=1
# python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fern --images images_4
# python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fern --images images_4 --sphinx_sam
# python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fern --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene flower --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene flower --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene flower --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fortress --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fortress --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fortress --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene horns --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene horns --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene horns --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene leaves --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene leaves --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene leaves --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene orchids --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene orchids --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene orchids --images images_4 --lisa

python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene room --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene room --images images_4 --sphinx_sam
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene room --images images_4 --lisa


python metrics_lisa.py --pred_output_path output/16_llff_masks/qwen_sam --tag qwen_sam_metrics
python metrics_lisa.py --pred_output_path output/16_llff_masks/lisa --tag lisa_metrics
python metrics_llff.py