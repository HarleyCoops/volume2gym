# Railroad Engineer 1959: From Rules to Reinforcement Learning

## Overview
Qwen3-RailroadEngineer1959-RL is a state-of-the-art neuro-symbolic project that translates the visual and procedural logic of the 1959 Railroad Instruction Manual into reinforcement learning reward functions.

## The Architecture: Gemini 3 Pro + SAM 3D
This project will augment the rule building extraction of the Dakota1890 project but augment that with 3D visualizations in the learning process. Not sure how this is going to work to be honest, but the math makes sense.

### 1. 3D Reconstruction: SAM 3D Objects
We use Meta's SAM 3D to lift 2D images into 3D.

*   **Model**: SAM 3D Objects
*   **Role**: "The Visual Cortex"
*   **Function**: Decomposes masked 2D signal diagrams into 3D point clouds and meshes, allowing the system to understand "depth" (e.g., a semaphore arm angle relative to the mast).

### 2. Semantic Reasoning: Gemini 3 Pro
We use Google's Gemini 3 Pro (gemini-3-pro-preview) as the exclusive cognitive engine.

*   **Role**: "The Brain"
*   **Function**:
    *   Takes the raw rulebook pages and SAM 3D reconstructions.
    *   Generates precise "Composite Signal Functions" (e.g., mapping a visual aspect to a procedural constraint).
    *   Translates 1959 operational text into formal logic for RL.

## Project Structure
Mimicking the structure of Dakota1890, the project is organized as follows:

This project demonstrates a complete pipeline for creating a specialized Reinforcement Learning (RL) environment and a synthetic dataset from a historical document: the *1959 Consolidated Code of Operating Rules*.

## 1. The Pipeline

We transformed a static PDF into a dynamic, interactive learning system for AI agents.

### Phase 1: Extraction & Digitization
*   **Source**: Scanned PDF of the 1959 Rulebook.
*   **Method**: Parallel processing using VLM (Claude 3.5 Sonnet) to extract text and structure.
*   **Result**: 
    *   **536** Structured Rules (JSON)
    *   **2,708** Evaluation Tasks (Scenarios)

### Phase 2: Environment Creation
*   **Package**: `railroad_1959` (Published on Prime Intellect Hub)
*   **Mechanism**: A `verifiers` compatible environment.
*   **Rubric**: An LLM-based examiner (Claude 3.5 Sonnet) that grades agents on:
    1.  **Safety** (Critical, 50% weight)
    2.  **Procedure** (Sequence of actions, 30% weight)
    3.  **Terminology** (Correct railroad language, 20% weight)

### Phase 3: Synthetic Data Generation (SFT)
*   **Goal**: Create a baseline model that "knows" the rules before RL training.
*   **Method**: Using the extracted rules to generate diverse, complex scenarios via Gemini Pro.
*   **Dataset**: `railroad_sft_data.jsonl`
    *   **Format**: Question (Scenario) -> Answer (Rule-based Action)
    *   **Volume**: Targeting ~2,000 - 5,000 pairs.

## 2. Training Strategy: Iterative RLAIF

We employ a "Curriculum Learning" approach:

1.  **Supervised Fine-Tuning (SFT)**:
    *   Train a base model (e.g., Qwen 2.5-0.5B or 1.5B) on the synthetic Q&A pairs.
    *   **Objective**: Minimize prediction error. Teach the model the *facts* of the rules.
    *   **Data Requirement**: ~2,000 pairs is a solid baseline for a small model (0.5B - 3B params) to learn the domain syntax and basic knowledge. For robust generalization, 5,000+ is ideal.

2.  **Reinforcement Learning (RL)**:
    *   Place the SFT model into the `railroad_1959` environment.
    *   **Objective**: Maximize the Rubric Score. Teach the model to *apply* the rules safely and precisely.
    *   **Outcome**: The model learns to reason, handle edge cases, and prioritize safety over simple text completion.

## 3. Usage

### Install Environment
```bash
prime env install harleycooper/railroad_1959
```

### Load Data for SFT
The synthetic dataset is available at:
`RailroadEngineer1959/data/railroad_sft_data.jsonl`

### Run RL Training
Use the provided `train.toml` configuration with the PrimeRL framework.

```bash
python RailroadEngineer1959/run_training.py
```
