export CUDA_VISIBLE_DEVICES=1
# python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fern --images images_4
# python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene flower --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fortress --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene horns --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene leaves --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene orchids --images images_4
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene room --images images_4
python metrics_llff.py