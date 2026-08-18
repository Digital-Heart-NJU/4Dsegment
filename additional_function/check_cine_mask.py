import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import math

def check_cine_mask(cine_path, mask_path):
    if not os.path.exists(cine_path):
        raise FileNotFoundError(f"cine file not found: {cine_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    # -------------------------------
    # 2. 读取数据
    # -------------------------------
    cine_img = nib.load(cine_path)
    mask_img = nib.load(mask_path)

    cine_data = cine_img.get_fdata()   # (x, y, z, t)
    mask_data = mask_img.get_fdata()   # (x, y, z) 或 (x, y, z, t)

    print("cine shape:", cine_data.shape)
    print("mask shape:", mask_data.shape)

    # -------------------------------
    # 3. 选择切片
    # -------------------------------
    z_slice = cine_data.shape[2] // 2
    # z_slice = 4
    cine_frames = cine_data[:, :, z_slice, :]  # (x, y, t)

    if mask_data.ndim == 4:  # mask 随时间变化
        mask_frames = mask_data[:, :, z_slice, :]
    else:  # 静态 mask，复制到所有帧
        mask_frames = np.repeat(mask_data[:, :, z_slice][:, :, None],
                                cine_frames.shape[-1], axis=-1)

    # -------------------------------
    # 4. 动态显示：cine + 半透明 mask
    # -------------------------------
    fig, ax = plt.subplots()

    im = ax.imshow(cine_frames[:, :, 0].T, cmap="gray", animated=True)

    # 定义多标签颜色映射表（R,G,B,A）
    label_colors = {
        0: (1, 0, 0, 0.4),   # 红色，透明度 0.4
        1: (0, 1, 0, 0.4),   # 绿色
        2: (0, 0, 1, 0.4),   # 蓝色
        4: (1, 1, 0, 0.4),   # 黄色
        5: (0, 1, 1, 0.4),   # 青色
        6: (1, 0, 1, 0.4)    # 品红
    }
    overlay = ax.imshow(np.zeros_like(cine_frames[:, :, 0]), cmap="jet", alpha=0.0, animated=True)

    def update(frame):
        # 更新 cine 背景
        im.set_array(cine_frames[:, :, frame].T)

        # 创建一个 RGBA 图层
        overlay_img = np.zeros((*cine_frames.shape[:2], 4))  # (x, y, RGBA)

        # 遍历每个标签，填充颜色
        labels = np.unique(mask_frames[:, :, frame])
        for lbl in labels:
            if lbl == 0:
                continue  # 跳过背景
            color = label_colors.get(int(lbl), (1, 1, 1, 0.4))  # 默认白色
            overlay_img[mask_frames[:, :, frame].T == lbl] = color

        # 更新叠加层
        overlay.set_data(overlay_img)
        overlay.set_alpha(1.0)

        ax.set_title(f"Frame {frame} / {cine_frames.shape[-1]}")
        return [im, overlay]

    ani = animation.FuncAnimation(fig, update, frames=cine_frames.shape[-1], interval=100, blit=False)

    plt.show()

def check_cine_mask_all_slices_auto(cine_path, mask_path):
    if not os.path.exists(cine_path):
        raise FileNotFoundError(f"cine file not found: {cine_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    # -------------------------------
    # 1. 读取数据
    # -------------------------------
    cine_img = nib.load(cine_path)
    mask_img = nib.load(mask_path)

    cine_data = cine_img.get_fdata()  # (X,Y,Z,T)
    mask_data = mask_img.get_fdata()  # (X,Y,Z) or (X,Y,Z,T)

    print("cine shape:", cine_data.shape)
    print("mask shape:", mask_data.shape)

    X, Y, Z, T = cine_data.shape

    # 2. mask 维度统一
    if mask_data.ndim == 3:
        mask_data = np.repeat(mask_data[..., None], T, axis=-1)

    if mask_data.shape != cine_data.shape:
        raise ValueError("Mask shape does not match cine shape")

    # 3. 自动计算子图布局 ⭐
    n_cols = math.ceil(math.sqrt(Z))
    n_rows = math.ceil(Z / n_cols)

    print(f"Auto layout: {n_rows} rows × {n_cols} cols")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows)
    )

    axes = np.atleast_1d(axes).flatten()

    ims = []
    overlays = []

    label_colors = {
        1: (1, 0, 0, 0.4),  # 红色，透明度 0.4
        2: (0, 1, 0, 0.4),  # 绿色
        3: (0, 0, 1, 0.4),  # 蓝色
        # 4: (1, 1, 0, 0.4),  # 黄色
        # 4: (0, 1, 1, 0.4),  # 青色
        # 6: (1, 0, 1, 0.4)  # 品红
    }

    # -------------------------------
    # 4. 初始化每个 Z 切面
    # -------------------------------
    for z in range(Z):
        ax = axes[z]
        ax.axis("off")

        cine_frame = cine_data[:, :, z, 0]
        mask_frame = mask_data[:, :, z, 0]

        im = ax.imshow(cine_frame.T, cmap="gray", animated=True)

        overlay = ax.imshow(
            np.zeros((*cine_frame.T.shape, 4)),
            animated=True
        )

        ax.set_title(f"Z = {z}")

        ims.append(im)
        overlays.append(overlay)

    # 多余的子图隐藏
    for ax in axes[Z:]:
        ax.axis("off")

    # -------------------------------
    # 5. 动画更新
    # -------------------------------
    def update(frame):
        for z in range(Z):
            cine_frame = cine_data[:, :, z, frame]
            mask_frame = mask_data[:, :, z, frame]

            ims[z].set_array(cine_frame.T)

            overlay_img = np.zeros((*cine_frame.T.shape, 4))
            for lbl in np.unique(mask_frame):
                if lbl == 0:
                    continue
                color = label_colors.get(int(lbl), (1, 1, 1, 0.4))
                overlay_img[mask_frame.T == lbl] = color

            overlays[z].set_data(overlay_img)

        fig.suptitle(f"Frame {frame+1}/{T}", fontsize=14)
        return ims + overlays

    ani = animation.FuncAnimation(
        fig, update,
        frames=T,
        interval=100,
        blit=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 指定 NIfTI 文件路径

    cine_path = r"/home/mengao/Code/4Dsegment/data/subject_1/example.nii"
    mask_path = r"/home/mengao/Code/4Dsegment/data/subject_1/segs/seg_lvsa_ED.nii.gz"

    # check_cine_mask(cine_path, mask_path)
    check_cine_mask_all_slices_auto(cine_path, mask_path)