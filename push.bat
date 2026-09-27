@echo off
echo Initializing Git repository...
.\mingit\cmd\git.exe init

echo Adding files...
.\mingit\cmd\git.exe add .

echo Configuring Git (temporary)...
.\mingit\cmd\git.exe config user.name "Your Name"
.\mingit\cmd\git.exe config user.email "youremail@example.com"

echo Committing files...
.\mingit\cmd\git.exe commit -m "Initial commit"

echo Setting main branch...
.\mingit\cmd\git.exe branch -M main

echo Adding remote repository...
.\mingit\cmd\git.exe remote add origin https://github.com/VAIBHAV-AGARWAL11/Onion_grading_system.git

echo Pushing to GitHub...
echo.
echo =======================================================
echo A GitHub login window may pop up. Please log in!
echo =======================================================
.\mingit\cmd\git.exe push -u origin main

echo.
echo Done! Press any key to close this window.
pause
