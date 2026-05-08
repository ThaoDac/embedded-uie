# LU2Net 

# install from requirements.txt
pip install -r requirements.txt


### Console Application

#### Process a single image

```bash
python app_console.py -i input.jpg -o output.jpg
```

#### Process a directory of images

```bash
python app_console.py --input ./dataset/input/ --output ./results/ --gt ./dataset/gt/
```

#### Process a video

```bash
# Generate output video only
python app_console.py -v input.mp4 -o output_folder/

# Generate output video and save individual frames
python app_console.py -v input.mp4 -o output_folder/ --save-frames

# Process every 5th frame (faster processing)
python app_console.py -v input.mp4 -o output_folder/ --frame-skip 5
```

Excutorch
```bash
# Tạo môi trường mới với Python 3.12
conda create -n executorch python=3.12 -y

# Kích hoạt môi trường
conda activate executorch

# Cài đặt các thư viện cần thiết
pip install executorch torch torchvision