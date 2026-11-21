import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def Check_sax(cine_path):
    direction = "axial"  # 'axial', 'sagittal', 'coronal'
    # -------------------------------
    # 读取数据
    # -------------------------------
    if not os.path.exists(cine_path):
        raise FileNotFoundError(f"cine file not found: {cine_path}")

    cine_img = nib.load(cine_path)
    cine_data = cine_img.get_fdata()   # (256,256,8,30)

    print("cine shape:", cine_data.shape)

    # -------------------------------
    # 选择方向
    # -------------------------------
    axis_map = {"axial": 2, "sagittal": 0, "coronal": 1}
    axis = axis_map.get(direction.lower(), 2)

    num_slices = cine_data.shape[axis]
    slice_indices = np.arange(num_slices)

    # -------------------------------
    # 获取切片函数
    # -------------------------------
    def get_slice(data, slice_idx, axis):
        if axis == 0:      # sagittal
            return data[slice_idx, :, :, :]   # (y,z,t)
        elif axis == 1:    # coronal
            return data[:, slice_idx, :, :]   # (x,z,t)
        else:              # axial
            return data[:, :, slice_idx, :]   # (x,y,t)

    cine_slices = [get_slice(cine_data, idx, axis) for idx in slice_indices]

    # -------------------------------
    # 绘图布局
    # -------------------------------
    cols = 4
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    # >>> 新增：全局时间帧显示文本（放在图像上方）
    time_text = fig.text(
        0.5, 0.95, "",            # 位置：居中靠上
        ha="center", va="center",
        fontsize=14, color="black"
    )
    axes = axes.flatten()

    ims = []
    for i, ax in enumerate(axes):
        if i >= num_slices:
            ax.axis('off')
            continue

        cine_slice = cine_slices[i]  # (H,W,T)
        im = ax.imshow(cine_slice[:, :, 0].T, cmap="gray", animated=True)
        ax.set_title(f"Slice {slice_indices[i]}")
        ax.axis('off')

        ims.append((im, cine_slice))

    num_frames = cine_data.shape[-1]

    # -------------------------------
    # 动画更新函数
    # -------------------------------
    def update(frame):
        for im, cine_slice in ims:
            im.set_array(cine_slice[:, :, frame].T)
            # # ===== 新增：更新时间显示 =====
            # time_text.set_text(
            #     f"Frame {frame + 1} / {num_frames}    "
            #     f"Cardiac phase: {time_percent[frame]:.1f}%"
            # )
        # >>> 新增：更新时间帧编号
        time_text.set_text(f"Frame: {frame + 1} / {num_frames}")

        return [im for im, _ in ims] + [time_text]

    # -------------------------------
    # 播放动画
    # -------------------------------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=100,   # 100 ms / frame
        blit=False
    )

    plt.tight_layout()
    plt.show()

def Check_lax(cine_path):
    if not os.path.exists(cine_path):
        raise FileNotFoundError(f"cine file not found: {cine_path}")

    # -------------------------------
    # 2. 读取数据
    # -------------------------------
    cine_img = nib.load(cine_path)
    cine_data = cine_img.get_fdata()  # shape: (x, y, z, t)

    print("cine shape:", cine_data.shape)

    # -------------------------------
    # 3. 选择切片
    # -------------------------------
    # # z_slice = cine_data.shape[2] // 2   # 取中间层
    # cine_frames = cine_data[:, :, :]  # (x, y, t)
    z_slice = 0  # 现在 z 维度只有 1
    cine_frames = cine_data[:, :, z_slice, :]

    # -------------------------------
    # 4. 动态显示 cine
    # -------------------------------
    fig, ax = plt.subplots()
    im = ax.imshow(cine_frames[:, :, 0].T, cmap="gray", animated=True)

    def update(frame):
        im.set_array(cine_frames[:, :, frame].T)
        ax.set_title(f"Frame {frame} / {cine_frames.shape[-1]}")
        return [im]

    ani = animation.FuncAnimation(fig, update,
                                  frames=cine_frames.shape[-1],
                                  interval=100, blit=False)

    plt.show()

if __name__ == "__main__":
    # -------------------------------
    # 用户配置
    # -------------------------------

    # cine_path = r"/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/group2/subject001/Instance_2/SAX.nii.gz"
    # cine_path = r"/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/group3/subject002/Instance_2/LAX_4Ch.nii.gz"

    # cine_path = r"/media/mengao/T7/Dataset_XI-AN_XINSIJUN/normal_select/A YUN GAK1092085/SAX.nii.gz"
    cine_path = r"/media/mengao/T7/Dataset_XI-AN_XINSIJUN/normal_select/A YUN GAK1092085/LAX_4Ch.nii.gz"
    Check_sax(cine_path)
    # Check_lax(cine_path)
