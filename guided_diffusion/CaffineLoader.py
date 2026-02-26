import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
from matplotlib import pyplot as plt

Train_list = [
    'sub-021', 'sub-022', 'sub-028', 'sub-029', 'sub-031', 'sub-032', 'sub-033', 'sub-036', 'sub-038',
    'sub-041', 'sub-042', 'sub-044', 'sub-049', 'sub-050', 'sub-051', 'sub-054', 'sub-055', 'sub-059',
    'sub-062', 'sub-065', 'sub-068', 'sub-075', 'sub-077', 'sub-078', 'sub-082', 'sub-084', 'sub-085',
    'sub-086', 'sub-087', 'sub-093', 'sub-094', 'sub-095', 'sub-097', 'sub-098', 'sub-099', 'sub-100',
    'sub-101', 'sub-105', 'sub-106', 'sub-110', 'sub-111', 'sub-112', 'sub-115', 'sub-117',
    'sub-145', 'sub-092', 'sub-120', 'sub-144', 'sub-060', 'sub-109', 'sub-053', 'sub-056',
    'sub-102', 'sub-016', 'sub-015', 'sub-020', 'sub-034', 'sub-138', 'sub-136', 'sub-104',
    'sub-023', 'sub-128', 'sub-072', 'sub-047', 'sub-040', 'sub-088', 'sub-114', 'sub-132',
    'sub-074', 'sub-076', 'sub-027', 'sub-063', 'sub-141', 'sub-108', 'sub-083', 'sub-069',
    'sub-137', 'sub-131', 'sub-161', 'sub-052', 'sub-043', 'sub-139', 'sub-046', 'sub-124',
    'sub-048', 'sub-149', 'sub-066', 'sub-119', 'sub-026', 'sub-058', 'sub-064', 'sub-071',
    'sub-057', 'sub-073', 'sub-080', 'sub-061', 'sub-107', 'sub-081', 'sub-103', 'sub-039',
]
Test_list = [
    'sub-122', 'sub-123', 'sub-125', 'sub-126', 'sub-129', 'sub-130', 'sub-133', 'sub-134', 'sub-146',
    'sub-147', 'sub-153', 'sub-154', 'sub-156', 'sub-159', 'sub-160', 'sub-118',
    'sub-070', 'sub-045', 'sub-035', 'sub-024', 'sub-017', 'sub-030', 'sub-091', 'sub-019', 'sub-025',
    'sub-140', 'sub-067', 'sub-127', 'sub-018'
]
def getLR(hr_data, scaling_factor):
    imgfft = np.fft.fftn(hr_data)
    imgfft = np.fft.fftshift(imgfft)

    x, y, z = imgfft.shape
    diff_x = x // (scaling_factor * 2)
    diff_y = y // (scaling_factor * 2)
    diff_z = z // (scaling_factor * 2)

    x_centre = x // 2
    y_centre = y // 2
    z_centre = z // 2
    mask = np.zeros(imgfft.shape)
    mask[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y,
    z_centre - diff_z: z_centre + diff_z] = 1
    imgfft = imgfft * mask

    imgifft = np.fft.ifftshift(imgfft)
    imgifft = np.fft.ifftn(imgifft)
    img_out = abs(imgifft)
    return img_out


class CaffineVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None, LR=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1n, t1c, t2w, t2f, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.coord_cache = None
        self.seqtypes = {"AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI'],
                         "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
                         "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA']}
        self.x = ['AD', 'FA', 'MD', 'RD']
        self.y = ['fit_FWF', 'fit_NDI', 'fit_ODI', 'DI', 'F', 'MSD', 'MSK', 'uFA']
        # self.seqtypes = {"AMICO/NODDI": ['fit_FWF'],
        #                  "FWDTI": ['FA', 'MD', 'RD', 'FW']}
        self.LR = LR
        print(self.directory)
        data_path = "/data/csx/Caffine/Caffine"
        self.database = []
        if mode == "train":
            for i in Train_list:
                path = i
                datapoint = dict()
                for k, v in self.seqtypes.items():
                    for v_i in v:
                        datapoint[v_i] = os.path.join(data_path,path, k, f"{v_i}.nii.gz")
                self.database.append(datapoint)
        else:
            for i in Test_list:
                path = i
                datapoint = dict()
                for k, v in self.seqtypes.items():
                    for v_i in v:
                        datapoint[v_i] = os.path.join(data_path,path, k, f"{v_i}.nii.gz")
                self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'
        # return_dict = {}
        # return_dict['file'] = filedict['FA'].split('/')[5]
        # return_dict['suhb'] = 'dummy_string'
        # print(filedict['FA'])
        number = filedict['FA'].split('/')[5]
        image = None
        label = None
        for k, v in self.seqtypes.items():
            for v_i in v:
                if v_i in filedict:
                    data_v_i = nibabel.load(filedict[v_i]).get_fdata()
                    data_v_i_clean = np.zeros((128, 128, 64))
                    data_v_i_clean[:, :, 7:-7] = data_v_i
                    # return_dict[v_i] = data_v_i_clean_pad.float()
                    if v_i in self.x:
                        if self.LR:
                            data_v_i_clean = getLR(data_v_i_clean, 2)
                        data_v_i_clean = clip_and_normalize(data_v_i_clean)
                        data_v_i_clean = torch.tensor(data_v_i_clean).unsqueeze(0)
                        if image is None:
                            image = data_v_i_clean.float()
                        else:
                            image = torch.cat((image, data_v_i_clean.float()), dim=0)
                    elif v_i in self.y:
                        data_v_i_clean = clip_and_normalize(data_v_i_clean)
                        data_v_i_clean = torch.tensor(data_v_i_clean).unsqueeze(0)
                        if label is None:
                            label = data_v_i_clean.float()
                        else:
                            label = torch.cat((label, data_v_i_clean.float()), dim=0)

        out_dict = {}
        weak_label = 1
        return image, out_dict, weak_label, label, number

    def __len__(self):
        return len(self.database)


def clip_and_normalize(img):
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))

    return img_normalized

def plot(data):
    # 配置参数
    slice_dim = -1  # 按最后一个维度切片
    rows, cols = 5, 5  # 图像矩阵的行列数
    total_slices = data.shape[slice_dim]
    selected_slices = 25  # 选择 25 个切片 (5x5)

    # 从 160 个切片中均匀选择 25 个
    step = max(1, total_slices // selected_slices)
    slice_indices = np.arange(0, total_slices, step)[:selected_slices]

    # 创建画布和子图
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # 绘制每个子图
    for idx, ax in enumerate(axes.flat):
        if idx < len(slice_indices):
            slice_idx = slice_indices[idx]
            img = data[:, :, slice_idx]  # 切片操作
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.set_title(f'Slice {slice_idx}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')  # 隐藏多余的子图

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()
#
if __name__ == '__main__':
        #     # testing
        #     # split = make_split()
        #     # print(split['validation'])
        #     p = 3 * [torch.linspace(-1, 1, 128)]
        #     print(len(p))
        #     print(p[0].shape)
        #     w = torch.stack(p, dim=0)
        #     print(w.shape)
        #     dim = 3  # 2d or 3d
        #     coord_cache = torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, 128)], indexing='ij'), dim=0)
        #     print(coord_cache.shape)
    image_size = 128
    half_res_crop = False

    ds = CaffineVolumes(
        directory='/data/csx/HCP_microstructure/HCP_microstructure',
        mode='train',
        LR=False
    )

    image, out_dict, weak_label, label, number = ds[0]
    print(image.shape, label.shape)
    print(number)
    data = image.numpy()
    plot(data[0])

