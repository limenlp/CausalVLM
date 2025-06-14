# What’s Missing in Vision-Language Models? Probing Their Struggles with Causal Order Reasoning


## Project Overview
This project addresses a critical gap in the evaluation of vision-language models: their ability to understand and reason about causal relationships. Although VLMs have demonstrated impressive performance on many downstream tasks, it remains unclear whether they truly grasp causal relations rather than relying on object recognition or activity identification shortcuts.

To bridge this gap, we introduce two new benchmarks:
- **VQA-Causal**, **VCR-Causal**: Two new benchmarks designed to isolate and rigorously evaluate VLMs’ causal reasoning abilities.

Our key findings:
1. VLMs excel at object and activity recognition but perform poorly on causal reasoning, often only marginally better than random guessing.
2. This shortcoming is primarily due to a severe lack of explicit causal expressions in widely used training datasets.
3. Fine-tuning with hard negative cases can significantly improve a model’s causal reasoning ability while preserving downstream performance and generalization.

Our study highlights a major limitation of current VLMs and lays the groundwork for future research on causal understanding.  

## Code & Data
> Note: Our server is currently offline due to an unexpected flooding incident. The server administrators are working diligently to restore service, and we hope it will be back online soon.

Once the server is back online, more data and code will be updated. We will update and make available the following:
- Code for **causal test**  
- Code for **multi‐choice** 
- Code for **data analysis**  
- Revised **file structure** and detailed **usage instructions**
