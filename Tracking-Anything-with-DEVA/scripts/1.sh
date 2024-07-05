python demo/demo_automatic.py --chunk_size 4 --img_path ../data/tandt/truck/images --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/lego_real_night_radial/images --amp --temporal_setting semionline --size 480 --output ./example/masks
python demo/demo_with_text.py --chunk_size 4 --img_path ../data/lego_real_night_radial/images --amp --temporal_setting semionline --size 480 --output ./example/bucket --prompt bucket.
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/nerf_real_360/pinecone/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks_8
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/fork/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/nerf_llff_data/trex/images_4 --amp --temporal_setting semionline --size 480 --output ./example/trex --prompt dino. --DINO_THRESHOLD 0.
CUDA_VISIBLE_DEVICES=4 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/sofa/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_8 --amp --temporal_setting semionline --size 480 --output ../data/ovs3d/sofa/gundam --prompt gundam. --DINO_THRESHOLD 0.70
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_8 --amp --temporal_setting semionline --size 480 --output ../data/ovs3d/sofa/gray_sofa --prompt "gray sofa". --DINO_THRESHOLD 0.72
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/bed/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/bed/masks_8
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/bench/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/bench/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/lawn/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/lawn/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/room/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/room/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/room/images_8 --amp --temporal_setting semionline --size 480 --output "../data/ovs3d/room/background" --prompt "background." --DINO_THRESHOLD 0.72


CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/gsgrouping/figurines/images_train --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/gsgrouping/figurines/masks
CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/gsgrouping/ramen/images_train --amp --temporal_setting semionline --size 480 --output ../data/gsgrouping/ramen/masks
CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/gsgrouping/teatime/images_train --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/gsgrouping/teatime/masks