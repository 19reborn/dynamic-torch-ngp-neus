CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v1_fixed_one_bug --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v2_normalize_gradients --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v3_normalize_gradients_lr0.001 --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --lr 0.001

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v4_normalize_gradients_check_nan --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v5_normalize_gradients_check_nan_gradients_detach --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v6_tiled_encoding --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v7_tiled_encoding_deform_gradients --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v8_hash_encoding --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v9_tiled_encoding_deform_gradients --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0
CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.15/v9_tiled_encoding_deform_gradients --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --test

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v1_tiled_encoding_deform_gradients --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v2_tiled_encoding_add_all --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v3_tiled_encoding_gradients_normalized --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v4_tiled_encoding_add_all --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v5_tiled_encoding_add_all_no_ek_loss --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v6_tiled_encoding_add_all_later_ek_loss --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v7_tiled_encoding_add_all_no_ekloss_more_iters --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --iters 100000

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v7_tiled_encoding_add_all_no_ekloss_canonical_direction --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v8_tiled_encoding_add_all_no_ekloss_canonical_direction_color_canonical_pos --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v9_canonical_direction_debug --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v10_canonical_direction_debug_noviewdirs --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v11_canonical_direction_noviewdirs --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v12_canonical_direction_noviewdirs_debug --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 --canonical_direction

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v13_hash_encoding_no_ek --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v14_smooth_hash_encoding_no_ek --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v14_smooth_hash_encoding_no_ek_debug --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.16/v14_smooth_hash_encoding_no_ek_debug_1 --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 

CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.17/v1_tcnn_hash_debug --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 
CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.17/v1_tcnn_hash_debug_2 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 
CUDA_VISIBLE_DEVICES=7 python main_dnerf_neus.py data/dnerf/jumpingjacks --workspace output/neus/upsample/8.17/v1_tcnn_hash_debug_3 --fp16 --preload --bound 1.0 --scale 0.8 --dt_gamma 0 