# handwritten-exam-parser
This repository contains a system for segmenting and labeling handwritten student answers from scanned, multi-page exam booklets.

Given a PDF of handwritten responses where students answer questions in arbitrary order, the system:
- Extracts student metadata (name, roll number)
- Identifies question numbers written in the margins
- Segments answers across pages
- Correctly maps answer regions to their corresponding questions, including sub-questions and continuations

The approach combines vision-language models with spatial reasoning to overcome limitations of traditional OCR and layout-based methods.

This project explores practical challenges in handwritten document understanding and multi-page reasoning.