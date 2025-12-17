---
id: 2
title: textbook-structure-tasks
stage: tasks
date: 2025-12-14
surface: agent
model: Claude Sonnet 4.5
feature: physical-ai-humanoid-robotics
branch: master
user: Claude
command: /sp.tasks
labels: ["textbook", "docusaurus", "structure", "tasks"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - specs/physical-ai-humanoid-robotics/textbook-structure-tasks.md
tests: []
outcome: Atomic tasks created for Docusaurus textbook structure
evaluation: Tasks broken down into 30 specific, executable items organized by user stories
---

# Prompt Text

Break down the NEXT deliverable into atomic, code-safe tasks for Claude Code.

DELIVERABLE:
Design and generate the complete Docusaurus textbook structure
for "Physical AI & Humanoid Robotics".

--------------------------------
ATOMIC TASKS
--------------------------------

1. Create Docusaurus docs folder structure
   - Create base folders:
     - docs/intro
     - docs/weeks
     - docs/modules
     - docs/capstone
     - docs/hardware
     - docs/glossary
     - docs/appendix

2. Generate Intro chapter markdown
   - File: docs/intro/intro.md
   - Include:
     - Course overview
     - Why Physical AI matters
     - How to use this textbook

3. Generate Week-based chapter files (Week 1–13)
   - Files:
     - docs/weeks/week-01-introduction.md
     - docs/weeks/week-02-physical-ai-foundations.md
     - ...
     - docs/weeks/week-13-conversational-robotics.md
   - Each file includes:
     - Learning goals
     - Key concepts
     - No code execution, only explanation

4. Generate Module-based chapters
   - Files:
     - docs/modules/ros2.md
     - docs/modules/gazebo.md
     - docs/modules/unity.md
     - docs/modules/nvidia-isaac.md
     - docs/modules/vla.md
   - Focus on conceptual + architectural understanding

5. Create Capstone Project chapter
   - File: docs/capstone/autonomous-humanoid.md
   - Include:
     - Problem statement
     - System architecture
     - Simulation-to-action flow

6. Create Hardware & Lab Guide
   - File: docs/hardware/hardware-guide.md
   - Include:
     - Workstation requirements
     - Jetson kits
     - Robot options
     - Cloud vs On-Prem tradeoffs

7. Create Glossary & Index
   - File: docs/glossary/glossary.md
   - Define key terms (ROS2, VLA, SLAM, Isaac, etc.)

8. Create Appendix
   - File: docs/appendix/appendix.md
   - Environment notes (no full scripts)

--------------------------------
CONSTRAINTS
--------------------------------
- One task = one markdown file
- No free-form writing
- Each chapter aligned with course outline
- Beginner-friendly language
- No hallucinated hardware or commands

--------------------------------
OPTIONAL CLI COMMANDS (WHEN IMPLEMENTING)
--------------------------------
mkdir -p docs/intro docs/weeks docs/modules docs/capstone docs/hardware docs/glossary docs/appendix

# Response Text

Created atomic, code-safe tasks for the Docusaurus textbook structure. The breakdown includes:

1. 30 specific tasks organized by user stories
2. Directory structure creation (T001-T002)
3. Intro chapter creation (T003)
4. 13 week-based chapters (T004-T016)
5. 5 module-based chapters (T017-T021)
6. Capstone project chapter (T022)
7. Hardware & Lab Guide (T023)
8. Glossary & Index (T024)
9. Appendix (T025)
10. Quality assurance tasks (T026-T030)

Each task follows the required format with proper IDs, story labels, and file paths. The tasks are designed to be executed in parallel where possible, with clear dependencies and independent test criteria.