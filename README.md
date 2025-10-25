# house-floor-generator

This project is a multi-stage GAN pipeline that generates high-resolution 2D house floor plans from user-defined parameters.  
For research and educational purposes only. Use is restricted; see the License section below.

## Features
- Room type prediction using a Random Forest Classifier trained on a housing dataset
- DCGAN for 256×256 floor plan generation
- U-Net based Denoiser GAN for final image cleanup
- Generates floor plans based on total area and number of bedrooms
- Streamlit-based web interface for interactive generation

## Installation
Clone the repository and install required dependencies:

1. **Prepare the environment**
   - Install Python 3.9 or later.
   - Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download datasets**  
   - Place them in a `datasets/` folder inside the project directory.

3. **Preprocess dataset**
   - Resize and process images using:
     ```bash
     python image_processing.py
     ```

4. **Train models (optional)**
   - To train the Stage 1 GAN:
     ```bash
     python stage1_gan.py
     ```
   - To train the Stage 3 Denoising GAN:
     ```bash
     python stage3_gan.py
     ```

5. **Run the pipeline**
   - Execute the following command to generate a floor plan:
     ```bash
     python main.py
     ```
   - Enter:
     - Total area (sq ft)
     - Number of bedrooms

6. **For deployment (Web App)**
   - To launch the Streamlit web interface, run:
     ```bash
     streamlit run webapp.py
     ```

7. **View results**
   - Generated floor plans will be saved in the `final_generated_images/` or `web_generated_images/` folder with timestamped filenames (256×256 pixels).

## Sample Outputs
Sample generated floor plans are included in the `samples/` folder for demonstration.
