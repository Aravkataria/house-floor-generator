import os
import cv2

root_folder = "folder"
output_folder = os.path.join(root_folder, "processed")
os.makedirs(output_folder, exist_ok=True)

for subdir in ["high_quality_architectural", "high_quality", "colorful"]:
    full_path = os.path.join(root_folder, subdir)
    
    for plan_folder in os.listdir(full_path):
        plan_path = os.path.join(full_path, plan_folder)
        
        for file in os.listdir(plan_path):
            if file.endswith(".png"):
                img_path = os.path.join(plan_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                # Resize to 256x256
                img_resized = cv2.resize(img, (256, 256))

                # Unique output name
                out_filename = f"{subdir}_{plan_folder}_{file}"
                out_path = os.path.join(output_folder, out_filename)

                cv2.imwrite(out_path, img_resized)

print("All images processed and saved in:", output_folder)
