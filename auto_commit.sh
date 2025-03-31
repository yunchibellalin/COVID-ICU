#!/bin/bash

# Add all changes to the staging area
git add .

# Commit changes
git commit -m "Initial commit with LFS tracking"

# Push to the main branch
git push -u origin main
