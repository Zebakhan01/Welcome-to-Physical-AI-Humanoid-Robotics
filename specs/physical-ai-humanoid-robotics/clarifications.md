# Assumptions Clarification - Physical AI & Humanoid Robotics Textbook

## Project Scope Clarifications

### 1. Book as Source of Truth for RAG Chatbot
**Assumption**: The textbook content serves as the exclusive knowledge base for the RAG chatbot.
**Clarification**: The RAG system will only be trained on content from the textbook. The chatbot will be explicitly programmed to avoid generating responses based on external knowledge or general AI training. When faced with questions outside the textbook scope, the chatbot should respond with "I can only answer questions based on the Physical AI & Humanoid Robotics textbook content."

### 2. Educational Focus vs. Practical Robot Control
**Assumption**: The project requires only educational content, not real robot control code.
**Clarification**: The project will focus on theoretical concepts, simulations, and educational examples. Any code examples will be for illustrative purposes and educational understanding rather than actual robot deployment. No interfaces to real hardware or robot control systems will be implemented.

### 3. Simulation Content Approach
**Assumption**: Simulations will be explained conceptually rather than executed within the textbook.
**Clarification**: The textbook will include conceptual explanations of simulation platforms (Gazebo, Unity, Isaac Sim) with code examples and screenshots. Actual simulation execution will not be embedded in the textbook but will be referenced as external activities for learners.

### 4. Hardware Documentation Style
**Assumption**: Hardware sections will be descriptive rather than setup tutorials.
**Clarification**: Hardware chapters will provide comprehensive overviews of different hardware components, their functions, and their applications in humanoid robotics. These sections will explain hardware concepts without requiring actual hardware setup or assembly instructions.

### 5. Bonus Features Priority
**Assumption**: Authentication, personalization, and Urdu translation are optional extensions.
**Clarification**: These features are indeed optional and will be implemented only if core functionality is stable. The priority remains on delivering excellent core educational content with a functional RAG chatbot. Bonus features can be developed in future iterations.

### 6. Target Audience Background
**Assumption**: Target audience has basic Python and AI background.
**Clarification**: Content should assume familiarity with Python programming, basic machine learning concepts, and fundamental mathematics (linear algebra, calculus). The textbook will not cover basic programming concepts but may provide brief refreshers on relevant AI/ML techniques as they apply to robotics.

## Implementation Boundaries

### In Scope
- Comprehensive textbook content covering Physical AI and Humanoid Robotics
- Functional RAG chatbot that only responds to textbook content
- Conceptual explanations of simulation platforms
- Descriptive hardware guides
- Docusaurus-based deployment on GitHub Pages
- FastAPI backend with Qdrant and Neon integration

### Out of Scope
- Real robot control interfaces
- Hardware setup tutorials
- Complex authentication systems (basic auth acceptable)
- Full Urdu localization (English only initially)
- Advanced personalization features

## Technical Constraints

### Hard Requirements
- RAG system must validate responses against textbook content
- All code examples must be educational and well-documented
- Simulation explanations must be platform-agnostic
- Hardware descriptions must be conceptual rather than prescriptive

### Soft Requirements
- Content should be modular for future translation
- Architecture should support eventual personalization
- Code structure should accommodate additional features
- Design should be accessible to beginners with Python/AI background

## Acceptance Criteria

### Must Have
- Complete textbook content covering all specified topics
- RAG chatbot that only answers from textbook content
- Proper separation between concepts, simulations, and hardware
- Deployable Docusaurus site with integrated chatbot

### Nice to Have
- Basic user authentication
- Personalization capabilities
- Multi-language support preparation
- Advanced search and filtering