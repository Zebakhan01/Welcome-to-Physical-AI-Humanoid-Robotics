@echo off
echo Restructuring project architecture...

REM Move docs directory
if exist "E:\Ai-Hacakthon\docs" (
    move "E:\Ai-Hacakthon\docs" "E:\Ai-Hacakthon\frontend\"
)

REM Move src directory if it exists
if exist "E:\Ai-Hacakthon\src" (
    move "E:\Ai-Hacakthon\src" "E:\Ai-Hacakthon\frontend\"
)

REM Move static directory if it exists
if exist "E:\Ai-Hacakthon\static" (
    move "E:\Ai-Hacakthon\static" "E:\Ai-Hacakthon\frontend\"
)

REM Move docusaurus.config.js if it exists
if exist "E:\Ai-Hacakthon\docusaurus.config.js" (
    move "E:\Ai-Hacakthon\docusaurus.config.js" "E:\Ai-Hacakthon\frontend\"
)

REM Move package.json if it exists
if exist "E:\Ai-Hacakthon\package.json" (
    move "E:\Ai-Hacakthon\package.json" "E:\Ai-Hacakthon\frontend\"
)

REM Move other common Docusaurus files
if exist "E:\Ai-Hacakthon\docusaurus.config.js" (
    move "E:\Ai-Hacakthon\docusaurus.config.js" "E:\Ai-Hacakthon\frontend\"
)

if exist "E:\Ai-Hacakthon\sidebars.js" (
    move "E:\Ai-Hacakthon\sidebars.js" "E:\Ai-Hacakthon\frontend\"
)

if exist "E:\Ai-Hacakthon\babel.config.js" (
    move "E:\Ai-Hacakthon\babel.config.js" "E:\Ai-Hacakthon\frontend\"
)

if exist "E:\Ai-Hacakthon\.gitignore" (
    move "E:\Ai-Hacakthon\.gitignore" "E:\Ai-Hacakthon\frontend\"
)

echo Project restructuring completed!