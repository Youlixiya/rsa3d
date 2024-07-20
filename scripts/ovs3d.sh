#sofa
CUDA_VISIBLE_DEVICES=0 python train_gs.py --source_path data/ovs3d/sofa --images images_4 --model_path output/ovs3d/sofa

CUDA_VISIBLE_DEVICES=0 python mask_refine.py --src_path data/ovs3d/sofa/masks_8 --refine_path data/ovs3d/sofa/gundam
CUDA_VISIBLE_DEVICES=0 python mask_refine.py --src_path data/ovs3d/sofa/masks_8 --refine_path data/ovs3d/sofa/gray_sofa
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/ovs3d/sofa/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/ovs3d/sofa --images images_4
CUDA_VISIBLE_DEVICES=0 python test_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json --scene sofa


#bed
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/ovs3d/bed --images images_4 --model_path output/ovs3d/bed
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/ovs3d/bed/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/ovs3d/bed --images images_4
CUDA_VISIBLE_DEVICES=1 python test_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json --scene bed

#bench
CUDA_VISIBLE_DEVICES=2 python train_gs.py --source_path data/ovs3d/bench --images images_4 --model_path output/ovs3d/bench
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/ovs3d/bench/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/ovs3d/bench --images images_4
CUDA_VISIBLE_DEVICES=2 python test_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json --scene bench

#lawn
CUDA_VISIBLE_DEVICES=3 python train_gs.py --source_path data/ovs3d/lawn --images images_4 --model_path output/ovs3d/lawn
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/ovs3d/lawn/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/ovs3d/lawn --images images_4
CUDA_VISIBLE_DEVICES=3 python test_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json --scene lawn

#room
CUDA_VISIBLE_DEVICES=4 python train_gs.py --source_path data/ovs3d/room --images images_4 --model_path output/ovs3d/room
CUDA_VISIBLE_DEVICES=4 python mask_refine.py --src_path data/ovs3d/room/masks_8 --refine_path data/ovs3d/room/background
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/ovs3d/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/ovs3d/room --images images_4
CUDA_VISIBLE_DEVICES=4 python test_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json --scene room


python test_clip_ovs3d.py --cfg_path scripts/16_ovs3d_test_config.json

python test_ovs3d_qwen.py --cfg_path scripts/16_ovs3d_test_config.json --train_views 5
python test_ovs3d_qwen.py --cfg_path scripts/16_ovs3d_test_config.json --train_views 2
python test_ovs3d_qwen.py --cfg_path scripts/16_ovs3d_test_config.json --train_views 4
python test_ovs3d_qwen.py --cfg_path scripts/16_ovs3d_test_config.json --train_views 4 --clip_type box
