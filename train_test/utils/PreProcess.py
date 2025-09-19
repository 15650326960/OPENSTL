from einops import rearrange


class PreProcess:
    """
    预处理函数
    """

    @staticmethod
    def reshape_patch(img_tensor, patch_size):
        B, T, C, H, W = img_tensor.shape
        # 确保高度和宽度能被patch_size整除
        assert H % patch_size == 0, f"图像高度 {H} 不能被 patch_size {patch_size} 整除"
        assert W % patch_size == 0, f"图像宽度 {W} 不能被 patch_size {patch_size} 整除"
        
        patch_tensor = rearrange(
            img_tensor,
            "b t c (ph p1) (pw p2) -> b t (c p1 p2) ph pw",
            ph=H // patch_size,
            pw=W // patch_size,
            p1=patch_size,
            p2=patch_size,
        )
        return patch_tensor

    @staticmethod
    def reshape_patch_back(patch_tensor, patch_size):
        img_tensor = rearrange(
            patch_tensor,
            "b t (c p1 p2) ph pw -> b t c (ph p1) (pw p2)",
            p1=patch_size,
            p2=patch_size,
        )
        return img_tensor
