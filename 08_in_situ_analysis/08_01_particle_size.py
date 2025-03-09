print(f"\n--------------------------> Wellcome to EMcopilot! <--------------------------")
print(f"\n--------------------------> Importing required modules <--------------------------")
print(f"Please wait patiently, this may take dozens of seconds...")

import cv2
import csv
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from datetime import datetime
from scipy import stats
from scipy.stats import norm, skewnorm, cauchy
from PIL import Image
import os
import time
import albumentations
import torch
import hyperspy.api as hs

print(f"All modules imported successfully.")


def dm4_processor(directory, images_dir):
    """
    Process DM4 files in a specified directory, extract metadata, normalize images, and store sample names,
    dimensions, and units.

    Parameters:
        - dm4_files_directory (str): Directory containing DM4 files.
        - output_folder (str): Folder to save the normalized images.

    Returns:
        - sample_count (int): Number of processed dm files.
    """
    # Check files already processed
    sample_count = 0
    processed_files = set()
    if os.path.exists(f'{images_dir}/processed_files.txt'):
        with open(f'{images_dir}/processed_files.txt', 'r') as txt_file:
            for line in txt_file:
                processed_files.add(line.strip())
        sample_count = len(processed_files)
    else:
        sample_count = 0

    # Define a folder to save the images
    os.makedirs(images_dir, exist_ok=True)
    # Create or overwrite the text file to store sample names and dimensions
    if os.path.exists(f'{images_dir}/samples_info.txt'):
        with open(f'{images_dir}/samples_info.txt', "a") as txt_file:
            print(f"--------------------------> Starting import new DM files <--------------------------")
            # Load DM4 file
            for dm4_filename in os.listdir(directory):
                if dm4_filename.endswith(".dm4"):
                    dm4_file_name_no_ext = os.path.splitext(dm4_filename)[0]
                    if dm4_file_name_no_ext not in processed_files:
                        dm4_file_path = os.path.join(directory, dm4_filename)
                        dm4_file = hs.load(dm4_file_path)
                        try:
                            # Extract metadata
                            name = dm4_file.metadata.General.title
                            scale = dm4_file.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Scale
                            units = dm4_file.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Units
                            acquisition_time = dm4_file.original_metadata.ImageList.TagGroup0.ImageTags.DataBar[
                                'Acquisition Time']

                            # Extract image data and normalize it
                            image_data = dm4_file.data
                            normalized_data = (
                                    (image_data - image_data.min()) / (
                                     image_data.max() - image_data.min()) * 255).astype(
                                np.uint8)

                            # Save normalized image as PNG in a subfolder named after the sample name
                            sample_folder = os.path.join(images_dir, name)
                            os.makedirs(sample_folder, exist_ok=True)
                            output_path = os.path.join(sample_folder, f"{name}_img.png")
                            normalized_image = Image.fromarray(normalized_data)
                            normalized_image.save(output_path)

                            # Calculate resolution
                            dimension1_size = dm4_file.axes_manager[0].size
                            resolution = float(scale) * dimension1_size

                            # Output sample information
                            print(f"==========================> Sample Name: {name}")
                            print(f"                            Resolution: {dm4_file.data.shape}")
                            print(f"                            Scale: {scale} {units}/pixel")
                            print(f"                            Dimension: {resolution} {units}²")
                            print(f"                            Acquisition Time: {acquisition_time}")
                            print(f"                            Image saved as '{output_path}'\n ")

                            # Write sample name, dimension, and units to the text file
                            txt_file.write(f"{name}, {resolution}, {units}, {acquisition_time}\n")

                            # Increment the sample count
                            sample_count += 1

                        except AttributeError:
                            print(f"Unable to find the required metadata in the DM4 file in: {dm4_filename}")
        print(
            f" {sample_count - len(processed_files)} new DM4 files import complete. "
            f"ALl images saved as subfolders named by sample name.")
    else:
        with open(f'{images_dir}/samples_info.txt', "w") as txt_file:
            txt_file.write("Sample Name, Dimension, Units, Acquisition Time, Average Size\n")  # Write header
            print(f"\n\n\n\n--------------------------> Starting import DM files <--------------------------")
            # Load DM4 file
            for dm4_filename in os.listdir(directory):
                if dm4_filename.endswith(".dm4"):
                    dm4_file_path = os.path.join(directory, dm4_filename)
                    dm4_file = hs.load(dm4_file_path)

                    try:
                        # Extract metadata
                        name = dm4_file.metadata.General.title
                        scale = dm4_file.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Scale
                        units = dm4_file.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Units
                        acquisition_time = dm4_file.original_metadata.ImageList.TagGroup0.ImageTags.DataBar[
                            'Acquisition Time']

                        # Extract image data and normalize it
                        image_data = dm4_file.data
                        normalized_data = (
                                (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(
                            np.uint8)

                        # Save normalized image as PNG in a subfolder named after the sample name
                        sample_folder = os.path.join(images_dir, name)
                        os.makedirs(sample_folder, exist_ok=True)
                        output_path = os.path.join(sample_folder, f"{name}_img.png")
                        normalized_image = Image.fromarray(normalized_data)
                        normalized_image.save(output_path)

                        # Calculate resolution
                        dimension1_size = dm4_file.axes_manager[0].size
                        resolution = float(scale) * dimension1_size

                        # Output sample information
                        print(f"==========================> Sample Name: {name}")
                        print(f"                            Resolution: {dm4_file.data.shape}")
                        print(f"                            Scale: {scale} {units}/pixel")
                        print(f"                            Dimension: {resolution} {units}²")
                        print(f"                            Acquisition Time: {acquisition_time}")
                        print(f"                            Image saved as '{output_path}'\n ")

                        # Write sample name, dimension, and units to the text file
                        txt_file.write(f"{name}, {resolution}, {units}, {acquisition_time}\n")

                        # Increment the sample count
                        sample_count += 1

                    except AttributeError:
                        print(f"Unable to find the required metadata in the DM4 file in: {dm4_filename}")
        print(
            f" {sample_count} DM4 files import complete. "
            f"ALl images saved as subfolders named by sample name.")
    return sample_count


def segmentor(images_dir, model_path):
    """
    Segment images using the pre-trained deep learning model and save the results.

    Args:
        images_dir (str): Path to the folder containing input images and subfolders for each sample.
        model_path (str): Path to the pre-trained deep learning model.

    Returns:
        None
    """
    print(f"\n\n\n\n--------------------------> Initializing segmentation <--------------------------")

    # Define the color mapping for segmentation classes
    colors = [(0, 0, 0), (0, 0, 255)]

    # Check if a GPU is available, otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define test data augmentation operations
    test_aug = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(),
    ])

    all_particle_count = 0
    all_particle_areas = []
    processed_files = set()

    print(f"                            Loading model...")
    model = torch.load(model_path).to(device)
    model.eval()

    if os.path.exists(f'{images_dir}/processed_files.txt'):
        with open(f'{images_dir}/processed_files.txt', 'r') as txt_file:
            for line in txt_file:
                processed_files.add(line.strip())

    samples_info_path = os.path.join(images_dir, 'samples_info.txt')
    sample_names = []

    with open(samples_info_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(', ')
            if len(parts) == 4:
                sample_name, dimension, units, acquisition_time = parts
                sample_name = sample_name.strip()
                sample_names.append(sample_name)

    for sample_name in sample_names:
        if sample_name not in processed_files:
            sample_folder_path = os.path.join(images_dir, sample_name)

            if os.path.isdir(sample_folder_path):
                ori_image_path = os.path.join(sample_folder_path, f"{sample_name}_img.png")
                ori_image = cv2.imread(ori_image_path)

                # Data augmentation and preprocessing
                augmented_image = test_aug(image=ori_image)['image']
                augmented_image = np.expand_dims(np.transpose(augmented_image, axes=[2, 0, 1]), axis=0)
                augmented_image = torch.from_numpy(augmented_image).to(device).float()

                # Perform image segmentation using the deep learning model
                output = model(augmented_image).cpu().detach().numpy()[0].argmax(0)

                # Calculate the ratio of pixels for one class in the segmentation result
                ratio = np.sum(output[output == 1]) / (output.shape[0] * output.shape[1])

                # Map the segmentation result back to colors, blend with the original image
                output = np.reshape(np.array(colors, np.uint8)[np.reshape(output, [-1])], [512, 512, -1])
                output = cv2.resize(output, (ori_image.shape[1], ori_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                overlap = cv2.addWeighted(ori_image, 0.5, output, 0.5, 0)

                # calculate_particle_areas
                particle_count = 0
                particle_areas = []
                gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Noise filter
                    if area > 20:
                        particle_count += 1
                        particle_areas.append(area)

                # Get scale info
                image_height, image_width = output.shape[:2]
                dimensions_file_path = os.path.join(images_dir, 'sample_dimensions.txt')
                with open(f'{images_dir}/samples_info.txt', 'r') as file:
                    for line in file:
                        parts = line.strip().split(', ')
                        if len(parts) == 4:
                            name, dimension, units, acquisition_time = parts
                            if name == sample_name:
                                try:
                                    # Attempt to convert dimension to a float (assuming it's a number)
                                    image_width_nano = float(dimension)
                                    image_height_nano = float(dimension)
                                except ValueError:
                                    print("Dimension is not a valid number.")
                                break  # Exit the loop when the sample is found

                # Convert scale
                pixel_width_nano = image_width_nano / image_width
                pixel_height_nano = image_height_nano / image_height
                particle_areas_nano = [area * pixel_width_nano * pixel_height_nano for area in particle_areas]
                average_area = round(np.mean(particle_areas_nano), 2)

                info_path = os.path.join(images_dir, 'samples_info.txt')
                with open(info_path, 'r') as file:
                    lines = file.readlines()
                with open(info_path, 'w') as file:
                    for line in lines:
                        parts = line.strip().split(', ')
                        if parts[0] == sample_name:
                            updated_line = f"{sample_name}, {parts[1]}, {parts[2]}, {parts[3]}, {average_area} nm2\n"
                            file.write(updated_line)
                        else:
                            file.write(line)

                csv_path = os.path.join(sample_folder_path, f"{sample_name}_particles.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for area in particle_areas_nano:
                        csv_writer.writerow([area])

                output_image_path = os.path.join(sample_folder_path, f"{sample_name}_mask.png")
                overlap_image_path = os.path.join(sample_folder_path, f"{sample_name}_segmented.png")
                cv2.imwrite(output_image_path, output)
                cv2.imwrite(overlap_image_path, overlap)

                print(f"==========================> Sample Name: {sample_name}")
                print(f"                            Particle count: {particle_count}")
                print(f"                            Average particle size: {average_area} nm²")
                print(f"                            Saved: {csv_path}, {output_image_path}\n")

                all_particle_count += particle_count
                all_particle_areas.extend(particle_areas_nano)

    batch_csv_path = os.path.join(images_dir, 'original_data.csv')
    csv_mode = 'a' if os.path.exists(batch_csv_path) else 'w'
    with open(batch_csv_path, csv_mode, newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for area in all_particle_areas:
            csv_writer.writerow([area])
    print(f"Segmentation complete. Total particles: {all_particle_count}")


def analyzer(images_dir):
    """
    Incrementally analyzes new particle area data and generates statistical visualizations.

    Parameters:
        images_dir (str): The directory containing image data and particle CSV files.

    Returns:
        None. Saves statistical data and visualizations.
    """
    all_particle_areas = []  # Store all particle areas
    sample_count = 0  # Track the number of samples

    # Load the list of already processed files
    processed_files_path = os.path.join(images_dir, "processed_files.txt")
    processed_files = set()
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = set(line.strip() for line in f)

    new_processed_files = []  # Track new files to update `processed_files.txt`

    print(f"\n--------------------------> Analyzing individual samples <--------------------------")

    # Traverse directories to find and process new CSV files only
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith("_particles.csv") and file not in processed_files:
                sample_count += 1
                csv_path = os.path.join(root, file)
                new_processed_files.append(file)

                # Read particle areas from the current sample's CSV file
                particle_areas = []
                with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        particle_areas.append(float(row[0]))

                # Add the current sample's areas to the overall list
                all_particle_areas.extend(particle_areas)

                # Generate and save the histogram for the current sample
                sample_name = os.path.splitext(file)[0]
                output_path = os.path.join(root, f"{sample_name}_distribution.png")
                plt.figure()
                plt.hist(particle_areas, bins=30, density=True, alpha=0.6, color='g', label='Experimental Distribution')
                plt.xlabel('Particle Size (nm²)', fontsize=18, fontfamily='arial')
                plt.ylabel('Frequency', fontsize=18, fontfamily='arial')
                plt.title(f"{sample_name} Particle Size Distribution", fontsize=18, fontfamily='arial')

                # Fit and plot the distributions
                mu, std = norm.fit(particle_areas)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'b', linewidth=2, label='Fitted Normal Distribution')

                a, loc, scale = skewnorm.fit(particle_areas)
                p_skew = skewnorm.pdf(x, a, loc, scale)
                plt.plot(x, p_skew, 'r', linewidth=2, label='Fitted Skewed Distribution')

                loc_cauchy, scale_cauchy = cauchy.fit(particle_areas)
                p_cauchy = cauchy.pdf(x, loc_cauchy, scale_cauchy)
                plt.plot(x, p_cauchy, 'm', linewidth=2, label='Fitted Lorentzian Distribution')

                # Add legend and save the plot
                plt.legend()
                plt.savefig(output_path)
                plt.close()
                print(f"Saved {sample_name} distribution plot.")

    # Generate and save the overall histogram if new data is found
    if all_particle_areas:
        overall_output_path = os.path.join(images_dir, "histogram.png")
        plt.figure()
        plt.hist(all_particle_areas, bins=30, density=True, alpha=0.6, color='g', label='Experimental Distribution')
        plt.xlabel('Particle Size (nm²)', fontsize=18, fontfamily='arial')
        plt.ylabel('Frequency', fontsize=18, fontfamily='arial')
        plt.title("Overall Particle Size Distribution", fontsize=18, fontfamily='arial')

        # Fit and plot the distributions
        mu, std = norm.fit(all_particle_areas)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'b', linewidth=2, label='Fitted Normal Distribution')

        a, loc, scale = skewnorm.fit(all_particle_areas)
        p_skew = skewnorm.pdf(x, a, loc, scale)
        plt.plot(x, p_skew, 'r', linewidth=2, label='Fitted Skewed Distribution')

        loc_cauchy, scale_cauchy = cauchy.fit(all_particle_areas)
        p_cauchy = cauchy.pdf(x, loc_cauchy, scale_cauchy)
        plt.plot(x, p_cauchy, 'm', linewidth=2, label='Fitted Lorentzian Distribution')

        # Add legend and save the plot
        plt.legend()
        plt.savefig(overall_output_path)
        plt.close()
        print(f"Overall particle size distribution plot saved.")

    # Update the processed files list
    with open(processed_files_path, 'a') as f:
        for file in new_processed_files:
            f.write(f"{file}\n")

    # Print global statistics
    particle_count = len(all_particle_areas)
    mean_area = np.mean(all_particle_areas) if particle_count > 0 else 0
    separator = '-' * 100

    print(f"\n--------------------------> Global statistics <--------------------------")
    print(separator)
    print(f"                            Sample count: {sample_count}")
    print(f"                            Filtered Particle count: {particle_count}")
    print(f"                            Mean Particle Area: {mean_area:.2f} nm²")
    print(separator)


if __name__ == "__main__":
    print(f"\n\n\n\n--------------------------> Navigating directory <--------------------------")

    # Interact with user to initialize dm4 dir
    print(f"==========================> ACTION NEED ")
    while True:
        dm4_files_dir = input("Please enter the folder name where the pending dm4 is stored: ")
        # Check dir
        if not os.path.exists(dm4_files_dir):
            print(f"The folder '{dm4_files_dir}' does not exist. Please check the path and try again.")
            continue
        # Check dm4
        files_in_folder = os.listdir(dm4_files_dir)
        dm4_files = [file for file in files_in_folder if file.endswith(".dm4")]
        if not dm4_files:
            print(
                f"No DM4 files found in the folder '{dm4_files_dir}'. Please check the folder contents and try again.")
        else:
            print(
                f"Found {len(dm4_files)} DM4 files in the folder '{dm4_files_dir}'. Proceeding with further actions.")
            break

    # Interact with user to initialize results dir
    print(f"\n==========================> ACTION NEED ")
    while True:
        images_dir = input("Please enter the folder name to be created to save the results: ")
        # Check folder
        if not os.path.exists(images_dir):
            print(f"The folder '{images_dir}' does not exist. Creating it...")
            os.makedirs(images_dir)
            break

        # Check files containing
        files_in_folder = os.listdir(images_dir)
        if not files_in_folder:
            break
        makedir_response = input(
            f"The folder '{images_dir}' is not empty. Do you want to overwrite its contents? (yes/no): ")
        if makedir_response.lower() == 'yes':
            shutil.rmtree(images_dir)
            os.makedirs(images_dir)
            break
        elif makedir_response.lower() == 'no':
            print(f"Please enter a new folder name.")
            continue
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")
    print(f"The folder '{images_dir}' is ready for use.")

    # Interact with user to initialize model
    print(f"\n\n\n\n--------------------------> Configuring model <--------------------------")
    print(f"\n==========================> ACTION NEED ")
    # Check all .pt and .pth files
    current_directory = os.getcwd()
    selected_model = None
    model_files = glob.glob(os.path.join(current_directory, "*.pt")) + glob.glob(
        os.path.join(current_directory, "*.pth"))
    if not model_files:
        print("No .pt or .pth model files found in the current directory.")
    else:
        print("Available models in this project:")
        for i, model_file in enumerate(model_files):
            print(f"{i + 1}. {os.path.basename(model_file)}")
        # Interact to choose one
        while True:
            try:
                choice = int(input("Please enter the number of the model file you want to use: "))
                if 1 <= choice <= len(model_files):
                    selected_model = model_files[choice - 1]
                    print(f"You have selected '{os.path.basename(selected_model)}' for segmentation.")
                    break
                else:
                    print("Invalid choice. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Call the function to process DM4 files
    sample_count = dm4_processor(f"./{dm4_files_dir}", f"./{images_dir}")

    # Call the function to perform image segmentation
    segmentor(f"./{images_dir}", f"{selected_model}")

    # Call the function to analyze particle info and plot
    global fig1, fig2
    fig1 = plt.figure(figsize=(16, 9))
    fig2 = plt.figure(figsize=(16, 7))
    analyzer(f"./{images_dir}")

    # Pending
    processed_files = []
    if os.path.exists(f'{images_dir}/samples_info.txt'):
        with open(f'{images_dir}/samples_info.txt', "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(', ')
                sample_name = parts[0].strip()
                processed_files.append(sample_name)
            with open(f'{images_dir}/processed_files.txt', "w") as file:
                for line in processed_files:
                    file.write(line + '\n')
                file.close()
    else:
        print("Invalid samples_info dir.")
    while True:
        print(f"\n\n\n\n--------------------------> Pending for new data <--------------------------")
        # Plot initial
        if processed_files:
            last_processed_sample = processed_files[-1]
            sample_folder_path = os.path.join(images_dir, last_processed_sample)
            if os.path.exists(sample_folder_path):
                img_path = os.path.join(sample_folder_path, f"{last_processed_sample}_img.png")
                segmented_img_path = os.path.join(sample_folder_path, f"{last_processed_sample}_segmented.png")

                img = cv2.imread(img_path)
                segmented_img = cv2.imread(segmented_img_path)

                plt.figure(1)
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"{last_processed_sample} - Original Image")

                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
                plt.title(f"{last_processed_sample} - Segmented Image")
                plt.axis('off')
                plt.pause(1)

            else:
                print(f"Folder '{sample_folder_path}' does not exist.")
        else:
            print("No processed files found.")

        plt.figure(2)
        plt.clf()
        plt.imshow(cv2.imread(f"{images_dir}/histogram.png"))
        font_properties = {'family': 'arial', 'color': 'white', 'weight': 'normal', 'size': 20}
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(450, 0.01, f'Update Time: {current_time}', **font_properties, bbox=dict(facecolor='black', alpha=0.7))
        plt.axis('off')
        plt.pause(1)

        print(f"\n==========================> Now rechecking for new files in the folder '{dm4_files_dir}'.")
        # Read processed samples as list
        complete_files = []
        processed_files = []
        pending_files = []
        if os.path.exists(f'{images_dir}/samples_info.txt'):
            with open(f'{images_dir}/samples_info.txt', "r") as f:
                next(f)
                for line in f:
                    parts = line.strip().split(', ')
                    sample_name = parts[0].strip()
                    processed_files.append(sample_name)
                with open(f'{images_dir}/processed_files.txt', "w") as file:
                    for line in processed_files:
                        file.write(line + '\n')
                    file.close()
        else:
            print("Invalid samples_info dir.")

        # Read compete samples as list
        for filename in os.listdir(dm4_files_dir):
            if filename.endswith('.dm4'):
                file_name_without_extension = os.path.splitext(filename)[0]
                complete_files.append(file_name_without_extension)

        # Record pending samples
        pending_files = [file for file in complete_files if file not in processed_files]
        if len(pending_files) == 0:
            print("--------------------------> No pending files.")
        if len(pending_files) < 0:
            print("Error: number of samples reduced.")
        # Write pending files to a txt file (if there are any)
        if len(pending_files) > 0:
            # Write pending files
            with open(f'{images_dir}/pending_files.txt', "w") as f:
                for file in pending_files:
                    f.write(file + "\n")
            print(
                f"\n\n\n\n--------------------------> Found {len(pending_files)} new files in the folder '{dm4_files_dir}' <--------------------------")
            # Call the function to segment and analyze
            sample_count = dm4_processor(f"./{dm4_files_dir}", f"./{images_dir}")
            segmentor(f"./{images_dir}", f"{selected_model}")
            fig2.clear()
            analyzer(f"./{images_dir}")

        plt.ioff()
