---
sidebar_position: 5
---

# Week 5: Locomotion

## Learning Objectives

By the end of this week, students will be able to:
- Understand the principles of bipedal locomotion
- Analyze different walking patterns and gaits
- Implement basic walking controllers for humanoid robots
- Evaluate stability in dynamic walking systems

## Introduction to Locomotion

Locomotion is the ability to move from one place to another. For humanoid robots, locomotion presents unique challenges due to the complexity of bipedal walking, which requires precise balance, coordination, and control. Unlike wheeled robots that have continuous ground contact, bipedal robots have intermittent contact with the ground, making balance and stability critical concerns.

### Types of Locomotion

**Static Locomotion**:
- Center of mass always within support polygon
- Slow but stable movement
- Maintains balance at all times

**Dynamic Locomotion**:
- Center of mass may move outside support polygon
- Faster movement but requires active control
- Uses momentum to maintain balance

## Bipedal Walking Fundamentals

### Walking Phases

A complete walking cycle consists of:
1. **Single Support Phase**: One foot in contact with ground
2. **Double Support Phase**: Both feet in contact with ground
3. **Swing Phase**: Non-support leg moves forward

### Key Biomechanical Concepts

**Zero Moment Point (ZMP)**:
- Point where net moment of ground reaction forces is zero
- Critical for dynamic balance in walking
- Must remain within support polygon for stability

**Center of Mass (CoM)**:
- Average location of body mass
- Trajectory planning critical for stable walking
- Control of CoM movement essential for balance

**Center of Pressure (CoP)**:
- Point where ground reaction force is applied
- Moves during walking to maintain balance
- Related to ZMP concept

## Walking Pattern Generation

### Inverted Pendulum Model

The simplest model for bipedal walking:
- Body mass concentrated at single point
- Leg acts as massless rod
- Single contact point with ground
- Useful for understanding basic balance

### Linear Inverted Pendulum Model (LIPM)

Extension of inverted pendulum with constant height:
- CoM height remains constant
- Simplifies control design
- Good approximation for slow walking

### Capture Point Concept

Point where robot must step to come to stop:
- Function of current CoM state
- Critical for balance recovery
- Determines foot placement for stability

## Walking Controllers

### Preview Control

Uses future reference trajectory to compute control:
- Optimal solution for LIPM
- Requires future ZMP reference
- Provides smooth, stable walking

### Model Predictive Control (MPC)

Optimizes control over finite horizon:
- Handles constraints explicitly
- Adapts to changing conditions
- Computationally intensive but flexible

### Central Pattern Generators (CPGs)

Neural network models for rhythmic motion:
- Generate rhythmic walking patterns
- Adapt to terrain variations
- Biologically inspired approach

## Balance Control

### Feedback Control for Balance

**Cart-Table Model**:
- Simplified model of inverted pendulum
- Linear control design
- Good for small perturbations

**Pendulum Model**:
- More accurate representation
- Nonlinear control considerations
- Better for larger disturbances

### Balance Recovery Strategies

**Ankle Strategy**: Use ankle joints for small perturbations
**Hip Strategy**: Use hip joints for larger perturbations
**Stepping Strategy**: Take corrective steps when needed
**Suspension Strategy**: Use arms for balance recovery

## Gait Analysis

### Gait Parameters

**Step Length**: Distance between consecutive foot placements
**Step Width**: Lateral distance between feet
**Step Time**: Duration of single step
**Stride Length**: Distance for complete gait cycle

### Gait Stability Metrics

**Stability Margin**: Distance from ZMP to support polygon boundary
**Foot Placement Strategy**: How feet are positioned for stability
**Energy Efficiency**: Power consumption during walking

## Terrain Adaptation

### Flat Ground Walking

- Predictable environment
- Simplified control requirements
- Focus on efficiency and stability

### Rough Terrain Walking

- Variable ground height
- Obstacle negotiation
- Adaptive foot placement

### Stair Climbing

- Discrete height changes
- Different gait patterns required
- Precise foot placement critical

## Challenges in Humanoid Locomotion

### Dynamic Balance

Maintaining balance during dynamic motion:
- Continuous adjustment of control parameters
- Real-time response to disturbances
- Coordination of multiple joints

### Computational Requirements

Real-time control demands:
- Fast sensor processing
- Quick decision making
- Low-latency actuator commands

### Energy Efficiency

Minimizing power consumption:
- Optimal trajectory planning
- Efficient control algorithms
- Mechanical design considerations

### Safety Considerations

Preventing falls and damage:
- Fall detection and recovery
- Safe shutdown procedures
- Human safety during operation

## Advanced Locomotion Concepts

### Passive Dynamic Walking

Uses mechanical design for energy-efficient walking:
- Minimal active control
- Energy efficiency through design
- Limited adaptability

### Compliance Control

Incorporates compliant behavior:
- Adapts to ground irregularities
- Reduces impact forces
- Improves stability

## Week Summary

This week explored the fundamental principles of bipedal locomotion in humanoid robots. We covered walking models, balance control strategies, gait analysis, and challenges in achieving stable, efficient walking. Understanding locomotion is crucial for developing truly mobile humanoid robots.

The next week will focus on manipulation, exploring how humanoid robots grasp and manipulate objects in their environment.