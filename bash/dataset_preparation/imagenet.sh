#!/bin/bash
#SBATCH --output=imagenet.out
#SBATCH --error=imagenet.err
#SBATCH --time=2-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

echo "Starting job script..."

# Move to local disk
echo "Changing directory to SLURM temporary directory..."
cd $SLURM_TMPDIR

# Create a work directory
echo "Creating work directory..."
mkdir train
cd train

# Copy the tar file from project space to local disk
echo "Copying imagenet.tar from project space to local disk..."
cp /home/soroush1/projects/def-kohitij/soroush1/imagenet/ILSVRC2012_img_train.tar .

# Extract the tar file using pigz for parallel uncompression and verbose output
echo "Extracting imagenet.tar using pigz..."
python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/extract_script.py ILSVRC2012_img_train.tar .

# Remove the tar file to free up space
echo "Removing imagenet.tar to free up space..."
rm ILSVRC2012_img_train.tar

# Now do the computations with the extracted data...
# For example, running a training script
# Now do the computations with the extracted data...
# For example, running a training script
echo "Starting computations with the extracted data..."
python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/check_imagenet_bash_scripts.py --folder .

# Clean up
echo "Creating a tar archive of the results..."
cd $SLURM_TMPDIR

# Optionally, remove extracted data to free up space
# Optionally, remove extracted data to free up space
echo "Removing extracted data to free up space..."
rm -rf train