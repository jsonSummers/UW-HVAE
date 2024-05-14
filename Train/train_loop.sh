#!/bin/bash


# Function to create experiment folder, subfolders, and metadata file
create_experiment_folders() {
    local experiment_id="$1"
    local metadata_file="../Test/experiments/$experiment_id/metadata.txt"

    # Create experiment folder if it doesn't exist
    mkdir -p "../Test/experiments/$experiment_id/checkpoints/enhancer"
    mkdir -p "../Test/experiments/$experiment_id/evaluation_images"

    # Create metadata file and write parameter descriptions
    echo "Experiment Parameters:" > "$metadata_file"
    echo "Experiment ID: $experiment_id" >> "$metadata_file"
    echo "Learning Rate: $learning_rate" >> "$metadata_file"
    echo "Factor: $factor" >> "$metadata_file"
    echo "Patience: $patience" >> "$metadata_file"
    echo "Lambda L1: $lambda_l1" >> "$metadata_file"
    echo "Lambda Con: $lambda_con" >> "$metadata_file"
    echo "Lambda KL: $lambda_kl" >> "$metadata_file"
    echo "Use VGG: $use_vgg" >> "$metadata_file"
}

learning_rate_values=(0.001)
factor_values=(0.1)
patience_values=(5)
lambda_l1_values=(1.0)
lambda_con_values=(1.0)
lambda_kl_values=(1.0)
use_vgg_options=(false)

# Calculate the total number of experiments
total_experiments=$(( ${#learning_rate_values[@]} * ${#factor_values[@]} * ${#patience_values[@]} * ${#lambda_l1_values[@]} * ${#lambda_con_values[@]} * ${#lambda_kl_values[@]} * ${#use_vgg_options[@]} ))

# Initialize the counter
experiment_counter=0

# Loop through each combination of parameters
for learning_rate in "${learning_rate_values[@]}"; do
    for factor in "${factor_values[@]}"; do
        for patience in "${patience_values[@]}"; do
            for lambda_l1 in "${lambda_l1_values[@]}"; do
                for lambda_con in "${lambda_con_values[@]}"; do
                    for lambda_kl in "${lambda_kl_values[@]}"; do
                        for use_vgg in "${use_vgg_options[@]}"; do
                            # Increment the counter
                            ((experiment_counter++))

                            # Generate a unique experiment ID (using counter)
                            experiment_id="experiment_$experiment_counter"

                            echo "Running experiment $experiment_counter/$total_experiments: $experiment_id"
                            # Create experiment folders and metadata file
                            create_experiment_folders "$experiment_id"
                            # Run the Python script with the parameters as arguments
                            python train.py "$experiment_id" \
                                --lr "$learning_rate" \
                                --factor "$factor" \
                                --patience "$patience" \
                                --lambda_l1 "$lambda_l1" \
                                --lambda_con "$lambda_con" \
                                --lambda_kl "$lambda_kl" \
                                $(if [ "$use_vgg" = true ]; then echo "--use_vgg"; fi)
                        done
                    done
                done
            done
        done
    done
done