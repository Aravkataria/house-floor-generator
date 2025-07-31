# house-floor-generator

This project is a multi-stage GAN pipeline that generates high-resolution 2D house floor plans from input parameters.  
For research and educational purposes only. Unauthorized copying, modification, or distribution is strictly prohibited.

## Features
- Room type prediction using a Random Forest Classifier with CSV input
- Conditional GAN for 256×256 floor plan generation
- Denoising GAN for final cleanup
- Takes input parameters like total area and number of rooms

## Installation
Clone the repository and install required dependencies:

1. **Prepare the environment**
   - Install Python 3.9 or later.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download datasets**
   - Download [CubiCasa5K](https://www.kaggle.com/code/qmarva/cubicasa5k-swin-transformer-mmdetection) and [Housing Dataset](https://www.kaggle.com/datasets/ashydv/housing-dataset) manually.
   - Place them in a `datasets/` folder inside the project directory.

3. **Run the pipeline**
   - Open a terminal in the project folder and execute:
     ```bash
     python main.py
     ```

4. **For deployment (Web App)**
   - To launch the Streamlit-based web interface, run:
     ```bash
     python main_webapp.py
     ```

5. **Input parameters**
   - When prompted, enter:
     - Total area (in sq ft or sq m)
     - Number of rooms

6. **View results**
   - The generated final floor plan will be saved in a folder with a timestamped filename.
