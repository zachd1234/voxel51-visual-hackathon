# Voxel51 Visual Hackathon 🥇

This repo contains our first-place-winning project from the **Voxel51 Computer Vision Hackathon at Northeastern**.

## 🔴 Problem
Elderly individuals often struggle to find items or avoid hazards due to poor eyesight, cluttered environments, or cognitive limitations.

## 🧪 Solution
We prototyped an **AR glasses assistant** that:
- Listens for voice queries (“Where’s my white medicine bottle with the orange label?”)
- Detects objects in the scene using YOLO
- Understands and filters based on the voice prompt using Whisper + CLIPSeg
- Highlights the correct object directly in the video feed

## 🧠 Tech Stack
- 🗣️ **OpenAI Whisper** — converts speech to text
- 👁 **YOLO-world** — for zero-shot object detection
- 🎯 **CLIPSeg** — to align visual embeddings with natural language queries
- 🧱 **Python** with OpenCV + custom detection logic

## 🕶 Workflow
1. Glasses microphone records query
2. Whisper transcribes to text
3. YOLO detects candidate objects
4. CLIPSeg filters & ranks objects based on query
5. Visual overlay highlights correct object

## 🎥 Example Prompts & Results
- “Find my white remote” → highlights white remote in cluttered background  
- “Where are the yellow and blue pillows?” → highlights both items  
- “Spot the large white sofa” → targets specific furniture in frame

Videos are in `/videos/` — see:  
- `where is my remote controller?.mp4`  
- `blue pillow; not the yellow one.mp4`  
- `yellow and blue pillows.mp4`

## 🏆 Hackathon Outcome
- Built in 24 hours  
- Won **1st place** at Voxel51 Visual AI Hackathon  
- Judged on creativity, real-world utility, and technical depth

## 👥 Team
- Zach Derhake  
- Sidney Ma  
- Ronit Avadhuta  
- Nicholas Yee

Special thanks to **Voxel51** and **Daniel G.** for organizing the event!

## 🗃️ Repo Structure
