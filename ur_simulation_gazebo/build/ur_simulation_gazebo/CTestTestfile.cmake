# CMake generated Testfile for 
# Source directory: /workspaces/E-Yantra/ur_simulation_gazebo
# Build directory: /workspaces/E-Yantra/ur_simulation_gazebo/build/ur_simulation_gazebo
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_test_gazebo.py "/usr/local/python/current/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/workspaces/E-Yantra/ur_simulation_gazebo/build/ur_simulation_gazebo/test_results/ur_simulation_gazebo/test_test_gazebo.py.xunit.xml" "--package-name" "ur_simulation_gazebo" "--output-file" "/workspaces/E-Yantra/ur_simulation_gazebo/build/ur_simulation_gazebo/launch_test/test_test_gazebo.py.txt" "--command" "/usr/local/python/current/bin/python3" "-m" "launch_testing.launch_test" "/workspaces/E-Yantra/ur_simulation_gazebo/test/test_gazebo.py" "--junit-xml=/workspaces/E-Yantra/ur_simulation_gazebo/build/ur_simulation_gazebo/test_results/ur_simulation_gazebo/test_test_gazebo.py.xunit.xml" "--package-name=ur_simulation_gazebo")
set_tests_properties(test_test_gazebo.py PROPERTIES  LABELS "launch_test" TIMEOUT "180" WORKING_DIRECTORY "/workspaces/E-Yantra/ur_simulation_gazebo/build/ur_simulation_gazebo" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/launch_testing_ament_cmake/cmake/add_launch_test.cmake;131;ament_add_test;/workspaces/E-Yantra/ur_simulation_gazebo/CMakeLists.txt;13;add_launch_test;/workspaces/E-Yantra/ur_simulation_gazebo/CMakeLists.txt;0;")
