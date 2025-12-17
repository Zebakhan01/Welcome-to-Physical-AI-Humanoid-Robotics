# Physical AI & Humanoid Robotics Textbook

This project creates a unified AI-native textbook titled "Physical AI & Humanoid Robotics" with a Docusaurus-based frontend.

## Features

- Comprehensive textbook on Physical AI and Humanoid Robotics
- Week-by-week learning structure (13 weeks)
- Module-based deep dives into key technologies
- Hardware and lab guides
- Capstone project integration
- AI-native content with RAG-powered chatbot (future integration)

## Structure

The textbook is organized into:

- **Intro**: Introduction to the course and learning objectives
- **Weeks**: 13 weeks of progressive learning content
- **Modules**: Deep dives into ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action systems
- **Capstone**: Project integration and implementation guidelines
- **Hardware**: Hardware guide and assembly instructions
- **Appendix**: Reference materials and setup guides

## Setup

1. Install Node.js and npm
2. Run `npm install` to install dependencies
3. Run `npm start` to start the development server

## Development

This project uses Docusaurus for documentation. To start a local development server:

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Building

To build the static website for production:

```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.