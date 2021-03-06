# dehazing-openvino

Image dehazing refers to procedures that attempt to remove the haze amount in a hazy image and grant the degraded image an overall sharpened appearance to obtain clearer visibility and a smooth image.
Most of the handy techniques include one-pass image filters. However, the result isn't pleasing to the human eye and degrades images upon tested on nontrivial real-life images. On the other hand, deep learning based techniques that does not depend upon prior, yields good performance in terms of aesthetics and PSNR metric. Howerver, due to high parameter and FLOPs count, tend to be slow during inference.

To find the right balance between good metric score & runtime performance, we resort to techniques like Knowledge Distillation, Feature Fusion Attention(FFA) and OpenVino's model optimization to get best of both worlds!

## Technology Used

- Python
- PyTorch
- OpenVino

## References

- https://arxiv.org/abs/1911.07559
- https://arxiv.org/abs/2106.05237
- https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html