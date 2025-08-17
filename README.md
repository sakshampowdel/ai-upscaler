# 🖼️ AI Image Upscaler (ESRGAN)

This project is a **Python-based AI image upscaler** that enhances low-resolution images by **4×** using the pretrained **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** model.  

It allows you to take any input image and generate a sharper, higher-resolution output — all running locally on CPU (no GPU required).  

---

## 🚀 Features
- Upscales images by **4×** using **ESRGAN**  
- Runs on **CPU** (no GPU required)  
- Clean **command-line interface** (just provide an image path when prompted)  
- Outputs a **new high-resolution image** with a timestamped filename  
- Uses **PyTorch** + **PIL** for processing  

---

## 📦 Requirements
- Python 3.8+  
- [PyTorch](https://pytorch.org/get-started/locally/)  
- torchvision  
- Pillow  

Install dependencies:  
```bash
pip install torch torchvision pillow
```
---

## 🚀 Usage

1. Run the script:  
   ```bash
   python upscale.py
   ```
2. Enter the path to your image when prompted:

Where is your image located?
> ./lowres.png

3. The script will process the image and save the result in the current directory with a timestamped filename, e.g.:

17_August_2025_15.png

4. The output image will be upscaled by 4× using the pretrained ESRGAN model.


---

## 🧠 How It Works
- Loads the pretrained **RRDBNet** model (`RRDB_ESRGAN_x4.pth`)  
- Converts your input image into a PyTorch tensor  
- Passes it through the ESRGAN network  
- Outputs a **4× upscaled** version of the image  
- Saves it as a `.png`  

---

## 📸 Example
Input:  
Low-resolution 256×256 image  

Output:  
High-resolution 1024×1024 image (sharper details, reduced blur/artifacts)  

---

## ⚠️ Notes
- The model runs on **CPU by default**, which may be slower for large images.  
- For faster performance, install PyTorch with GPU support and run on CUDA.  
- Output quality depends on the pretrained weights used.  

---

## 📜 License
This project is for **educational purposes only**. The ESRGAN model and pretrained weights are credited to the [original ESRGAN authors](https://github.com/xinntao/ESRGAN).  

---


