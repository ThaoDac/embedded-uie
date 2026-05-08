# TACL-main

Flask-based demo application 
Python 3.12 

## Setup
1. Move into the project folder:
   ```bash
   cd ARTICLE/projectThao/UIR-PolyKernel-main
   ```
2. Create & activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model checkpoints
Ensure the pre-trained model exists at: `IMPLEMENTATION/UIR-PolyKernel-main/models`
The app expects UIR-PolyKernel model under `UIR-PolyKernel-main/models/UIR_PolyKernel_epoch_311.pth`

## Running the app
Start the Flask server directly with:
```bash
python app.py
```

By default the app binds to `0.0.0.0` on port `5000`. Modify the `app.run(...)` call in `app.py` if you need a different host or port.

## Using the web UI
1. Open a browser and navigate to [http://localhost:5000](http://localhost:8090).
2. Upload an underwater image (`.jpg`, `.jpeg`, `.png`, or `.bmp`).
3. Submit the form to process the image.  
   The enhanced output, sample pixel values, image quality metrics (PSNR, SSIM, UIQM, UIQE, NIQE), and performance measurements are rendered on the results page.

Uploaded and processed images are written to the `static/` folder. Delete its contents if you need to clear previous runs.
