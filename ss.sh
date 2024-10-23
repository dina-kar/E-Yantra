#!/bin/bash

# Define the list of files to source
FILES=(
    "./eyantra_warehouse/install/setup.bash"
    "./ebot_description/install/setup.bash"
    "./ebot_nav2/install/setup.bash"
    "./ur5_control/install/setup.bash"
    "./ur_description/install/setup.bash"
    "./ur_moveit_config/install/setup.bash"
    "./ur_simulation_gazebo/install/setup.bash"
)

# Loop through the files and source them
for FILE in "${FILES[@]}"; do
    if [[ -f $FILE ]]; then
        echo "Sourcing $FILE..."
        source $FILE
    else
        echo "File $FILE not found, skipping..."
    fi
done

echo "All specified files sourced (if found)."
