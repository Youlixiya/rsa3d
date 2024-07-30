#figurines
CUDA_VISIBLE_DEVICES=0 python train_gs.py --source_path data/gsgrouping/figurines --images images --model_path output/gsgrouping/figurines
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/gsgrouping/figurines/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/gsgrouping/figurines --images images

#ramen
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/gsgrouping/ramen --images images --model_path output/gsgrouping/ramen
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/gsgrouping/ramen/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/gsgrouping/ramen --images images

#teatime
CUDA_VISIBLE_DEVICES=2 python train_gs.py --source_path data/gsgrouping/teatime --images images --model_path output/gsgrouping/teatime
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/gsgrouping/teatime/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/gsgrouping/teatime --images images

python test_gsgrouping_qwen.py --cfg_path scripts/16_gsgrouping_test_config.json --train_views 3

python render.py --cfg_path scripts/16_gsgrouping_test_config.json --scene teatime