# Echo → MRI Translation

*CycleGAN-based unpaired image-to-image translation — enhancing echocardiography images into MRI-like cardiac visuals.*

🔗 **[Live Demo](https://varunn-ranaa-echomri.hf.space)**

---

## The Problem

Cardiac MRI provides high-quality diagnostic imaging, but it is:

* Expensive
* Time-consuming
* Limited to urban hospitals

Echocardiography (ultrasound), on the other hand, is:

* Affordable
* Widely available
* But produces low-quality, blurry images

A large portion of the population in rural areas lacks access to MRI facilities, leading to compromised diagnostic quality.

---

## Our Approach

We use a **CycleGAN model** to enhance echocardiography images by learning mappings between Echo and MRI domains — **without requiring paired datasets**.

```
Echo Image → Generator G → Enhanced Image  
Enhanced Image → Generator F → Reconstructed Echo
```

* Cycle consistency ensures structural preservation
* The model improves visual clarity while retaining cardiac features

> This makes the solution scalable in real-world healthcare settings where paired medical datasets are rarely available.

---

## Architecture Overview

![CycleGAN Architecture](assets/Technical_Diagram.jpeg)

**Training Flow:**

* Echo → Generator G → Enhanced Image
* Enhanced → Generator F → Reconstructed Echo
* Discriminators ensure realism in both domains
* Cycle consistency preserves cardiac structure

---

##  Video Explaination

[![Watch Demo](assets/thumbnail.png)](https://youtu.be/yEM8OEfaFXc)

---

##  Model Architecture

| Component     | Details                         |
| ------------- | ------------------------------- |
| Generator     | ResNet-based, 6 residual blocks |
| Discriminator | PatchGAN (70×70 patches)        |
| GAN Loss      | LSGAN (MSE) — stable training   |
| Cycle Loss    | L1 × 10 — preserves structure   |
| Identity Loss | L1 × 5 — maintains contrast     |

---

##  Dataset

| Dataset                                                  | Type             | Frames |
| -------------------------------------------------------- | ---------------- | ------ |
| [EchoNet-Dynamic](https://echonet.github.io/dynamic/)    | Echocardiography | 3,583  |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cardiac MRI      | 3,583  |


---

##  Training Results

| Epoch | G Loss | D Loss |
| ----- | ------ | ------ |
| 1     | 2.259  | 0.251  |
| 3     | 1.560  | 0.231  |
| 5     | 1.481  | 0.200  |
| 7     | 1.366  | 0.198  |
| 10    | 1.322  | 0.193  |

 Generator loss decreased from **2.26 → 1.32**, indicating stable learning.

---

## Project Structure

```
├── model/
│   ├── generator.py        # CycleGAN Generator
│   └── discriminator.py    # PatchGAN Discriminator
├── data/
│   └── dataset.py          # Unpaired dataset loader
├── train.py                # Training pipeline
├── app.py                  # Gradio web app
└── requirements.txt
```

---

## Tech Stack

* PyTorch
* CycleGAN
* Gradio
* HuggingFace Spaces

---

## Team

**TechGeeks_3.0**
WTC Group 6 — Round 2

---

## Note

> Making cardiac imaging more accessible, affordable, and intelligent using AI.
