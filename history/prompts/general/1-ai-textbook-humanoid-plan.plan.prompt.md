---
id: 1
title: ai-textbook-humanoid-plan
stage: plan
date: 2025-12-14
surface: agent
model: Claude Sonnet 4.5
feature: physical-ai-humanoid-robotics
branch: master
user: Claude
command: /sp.plan
labels: ["textbook", "ai", "robotics", "docusaurus", "rag", "fastapi"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - specs/physical-ai-humanoid-robotics/plan.md
  - specs/physical-ai-humanoid-robotics/research.md
  - specs/physical-ai-humanoid-robotics/data-model.md
  - specs/physical-ai-humanoid-robotics/quickstart.md
  - specs/physical-ai-humanoid-robotics/contracts/chat-api.yaml
  - specs/physical-ai-humanoid-robotics/contracts/content-api.yaml
  - specs/physical-ai-humanoid-robotics/contracts/auth-api.yaml
  - specs/physical-ai-humanoid-robotics/tasks.md
tests: []
outcome: Implementation plan created for Physical AI & Humanoid Robotics textbook
evaluation: Plan includes all required phases, technical context, constitution check, and project structure
---

# Prompt Text

Create a full production plan for building the AI-native textbook
"Physical AI & Humanoid Robotics".

The plan must cover:
- Complete textbook creation
- RAG chatbot development and integration
- Deployment
- Optional bonus extensions

--------------------------------
PHASE 1: TEXTBOOK STRUCTURE & DESIGN
--------------------------------
Goal: Lock the educational backbone before writing content.

Includes:
- Define chapter order (Intro → Weeks → Modules → Capstone)
- Define learning outcomes per chapter
- Map official course outline to chapters
- Define what each chapter must include:
  - Concepts
  - Diagrams
  - Tables
  - Code examples
  - Exercises
- Decide file naming conventions
- Identify shared assets (figures, architecture diagrams)

Deliverables:
- Final chapter list
- Chapter-by-chapter content checklist
- Folder structure (JSON)

--------------------------------
PHASE 2: CONTENT CREATION (AI-NATIVE)
--------------------------------
Goal: Write the textbook using Claude Code as a co-author.

Includes:
- Write chapters one-by-one using specs
- Use natural language programming (no free-form dumping)
- Keep content beginner-friendly but accurate
- Add diagrams, tables, and pseudo-code where needed
- Validate each chapter against learning outcomes

Deliverables:
- Complete markdown chapters
- Linked images and assets
- Internal cross-references

--------------------------------
PHASE 3: RAG CHATBOT SYSTEM
--------------------------------
Goal: Enable AI-assisted learning on top of the book.

Includes:
- Define RAG data sources (chapters only)
- Chunking and embedding strategy
- Vector indexing in Qdrant
- Metadata storage in Neon Postgres
- FastAPI backend for chatbot
- Support:
  - Full-book Q&A
  - Selected-text Q&A

Deliverables:
- RAG backend APIs
- Indexed book content
- Verified no-hallucination responses

--------------------------------
PHASE 4: CHATBOT INTEGRATION WITH BOOK
--------------------------------
Goal: Embed chatbot directly into the learning experience.

Includes:
- Embed ChatKit UI in Docusaurus
- Chapter-aware chatbot context
- Selected-text interaction support
- UX that does not distract from reading

Deliverables:
- Embedded chatbot UI
- Working end-to-end flow

--------------------------------
PHASE 5: DEPLOYMENT
--------------------------------
Goal: Make the project publicly accessible.

Includes:
- Deploy Docusaurus book to GitHub Pages
- Deploy FastAPI backend to cloud
- Configure environment variables
- Verify production readiness

Deliverables:
- Live book URL
- Live chatbot endpoint

--------------------------------
PHASE 6: BONUS EXTENSIONS (OPTIONAL)
--------------------------------
Goal: Earn extra hackathon points.

Includes:
- Signup / Signin using Better Auth
- User profiling (software + hardware background)
- Chapter-level personalization button
- Urdu translation button per chapter

Deliverables:
- Auth-enabled book
- Personalized content variants
- Urdu-translated chapters

--------------------------------
TIMELINE (SUGGESTED)
--------------------------------
Week 1:
- Phase 1 complete
- Start Phase 2 (Intro + Weeks 1–3)

Week 2:
- Complete Phase 2 (remaining chapters)
- Start Phase 3 (RAG backend)

Week 3:
- Complete Phase 3 & 4
- Phase 5 deployment
- Optional Phase 6

--------------------------------
OUTPUT REQUIREMENTS:
- Bullet-list plan (this document)
- Task table with time estimates
- Dependencies between phases
- Risks and mitigation strategies

# Response Text

Created a comprehensive implementation plan for the Physical AI & Humanoid Robotics textbook project. The plan includes:

1. Detailed project structure with all required phases
2. Technical context covering languages, dependencies, and constraints
3. Constitution check ensuring alignment with project principles
4. Research document addressing key technical decisions
5. Data models for all required entities
6. Quickstart guide for developers
7. API contracts for chat, content, and authentication services
8. Comprehensive task list organized by user stories

The implementation follows a structured approach with 6 phases as requested, with proper dependencies, risk assessment, and success criteria. All deliverables have been created according to the specifications.