# house-floor-generator

This project is a multi-stage GAN pipeline that generates high-resolution 2D house floor plans from input parameters.  
For research and educational purposes only. Unauthorized copying, modification, or distribution is strictly prohibited.

## Features
- Room type prediction using a Random Forest Classifier with CSV input
- Conditional GAN for 256×256 floor plan generation
- Denoising GAN for final cleanup
- Takes input parameters like total area and number of rooms
