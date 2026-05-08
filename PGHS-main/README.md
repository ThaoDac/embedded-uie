python app_console.py \
    --input Dataset/testset\(ref\)/test-EUVP/input \
    --output output \
    --gt Dataset/testset\(ref\)/test-EUVP/target \
    --model models/uw_epoch_69.pth

# Basic usage (directory)
python app_console.py \
  --input ./test_images \
  --output ./output \
  --config ./configs/uie_waterformer.yml \
  --model ./checkpoints/weights.pth

# With ground truth for PSNR/SSIM
python app_console.py \
  --input ./test_images/input \
  --output ./output \
  --gt ./test_images/target

# With tile-based processing for large images
python app_console.py \
  --input ./high_res_images \
  --output ./output \
  --tile 720 \
  --tile_overlap 32

# Video processing
python app_console.py \
  --input ./video.mp4 \
  --output ./output_frames \
  --video

# CPU mode
python app_console.py \
  --input ./test_images \
  --output ./output \
  --device cpu