# Hackathon 2024 - Submission of Group Just4Fun

Team members:
    - Max Singer
    - Johann Vérolet
    - Emilia Charlotte Wiehe

## Description
There are two methods employed to acertain the ideal gripper location within ~3 seconds.
First is an analytical approach. Here the part is converted to a binary mask using a ML
model and the gripper is placed at every (x,y) position and angle starting at the center
of mass. If no valid gripper position is found within 3 seconds the analytic approach
is exited and a more compact ML Model is started. The compact model generates random
gripper positions and classifies them as valid or not.

## How to Run
- Navigate to the 'Hackathon24' folder.
- Make sure requirements.txt and python version >3.10 is installed.
- Place the task.csv in the 'solution' folder.
- Use the following command: python solution/main.py './evaluate/task.csv' './solution/result'
- NOTE: Importing the libraries might take a few seconds. The start of the program will be announced
  in the terminal.
- The results will be compiled in solution/result.csv
- OPTIONAL: To generate the resulting images in solution/visulization use:
    - python solution/visualize_solution.py

## ... and other things you want to tell us
- Analylitic approach is extremely accurate, but very slow.
    - In 3s: ~30 mm radius around the part center of mass (COM) can be checked analytically
        - In reality the gripper has to be close to the COM anyway, so the analytic approach
        should cover all relevant cases.
- The secondary ML Model is fast, but very inaccurate and therefore serves more as a 'last
ditch effort' to still return a decent gripper position.