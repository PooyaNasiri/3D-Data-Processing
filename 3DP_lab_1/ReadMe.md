SGM-Census with mono-depth initial guess
========================================

---
## Instructions

**Building (Out-of-Tree)**

    mkdir build
    cd build/
    cmake ..
    make
    
**Usage (from build/ directory)**

    ./sgm <right image> <left image> <monocular_right> <gt disparity map> <output image file> <disparity range> 

**Examples**

    ./sgm Examples/Rocks1/right.png Examples/Rocks1/left.png Examples/Rocks1/right_mono.png Examples/Rocks1/rightGT.png output_disparity.png 85

---


