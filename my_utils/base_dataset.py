import numpy as np
from my_utils import get_calib
from PIL import Image
import os


# 加载kitti 插入人后的obj点云 返回points, faces
def load_obj_points(obj_path):
    with open(obj_path) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3]), float(strs[4])))
            if strs[0] == "f":
                faces.append((int(strs[1]), int(strs[2]), int(strs[3]), float(strs[4])))
    points = np.array(points)

    return points


def obj_img(obj_path,dataroot):
    in_points = load_obj_points(obj_path)
    calib_txt = dataroot + '/calib/' + obj_path.split('/')[-1].split('.')[0].split('_')[0] + '.txt'

    image_size = (384, 512)  # (1280,384)

    # 8、雷达坐标 转 图像坐标
    calib = get_calib.Calibration(calib_txt)
    in_points_img_3, in_points_img_depth = calib.lidar_to_img(in_points[:,:3])
    in_points_img_4 = np.hstack((                                               # x，y,，深度，反射强度
                                  in_points_img_3[:,0].reshape(-1,1),
                                  in_points_img_3[:,1].reshape(-1,1),
                                  in_points_img_depth.reshape(-1,1),
                                  in_points[:,3].reshape(-1,1)
                                  ))

    # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    img_in = np.zeros((image_size[1],image_size[0],1), dtype=np.uint8)    # 图像模板
    img_gt = np.zeros((image_size[1],image_size[0],1), dtype=np.uint8)    # 图像模板


    box_center = (in_points_img_4[:,0].mean(),in_points_img_4[:,1].mean())

    for index_point in range(len(in_points_img_4)):
        point = in_points_img_4[index_point]
        x_index = int(point[0] + image_size[0]/2 - box_center[0] + 0.5)
        y_index = int(point[1] + image_size[1]/2 - box_center[1] + 0.5)
        try:
            img_gt[y_index,x_index,:] = 255

            kernal = 7
            zero = kernal//2
            for i in range(kernal):
                for j in range(kernal):
                    img_in[y_index-zero+i, x_index-zero+j,:] = 255
        except:
            pass

    return img_in,img_gt





def save_results(results_path,epoch,iteration,img_ins,img_gts,outputs,paths):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'epoch_{}'.format(epoch))
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, 'iteration_{}'.format(iteration))
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    img_id = paths[0].split('/')[-1].split('.')[0]


    img_in = Image.fromarray(((img_ins.data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_gt = Image.fromarray(((img_gts.data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    output = Image.fromarray(((outputs.data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))

    img_in_path = '{}/{}_img_in.png'.format(results_path,img_id)
    img_gt_path = '{}/{}_img_gt.png'.format(results_path,img_id)
    output_path = '{}/{}_output.png'.format(results_path,img_id)

    img_in.save(img_in_path)
    img_gt.save(img_gt_path)
    output.save(output_path)



if __name__ == '__main__':
    pass

    # import os
    # from PIL import Image
    #
    # def make_dataset2(dir, max_dataset_size=float("inf")):
    #     images = []
    #     assert os.path.isdir(dir), '%s is not a valid directory' % dir
    #     for root, _, fnames in sorted(os.walk(dir)):
    #         for fname in fnames:
    #             path = os.path.join(root, fname)
    #             images.append(path)
    #     return images[:min(max_dataset_size, len(images))]
    #
    #
    #
    # A_paths = sorted(make_dataset2('/code/mix-pe/cycle_gan/trainA'))
    # B_paths = sorted(make_dataset2('/code/mix-pe/cycle_gan/trainB'))
    #
    # dataroot = '/code/mix-pe/cycle_gan'
    #
    #
    # for index in range(len(A_paths)):
    # # for index in range(2):
    #     A_path = A_paths[index % len(A_paths)]  # make sure index is within then range
    #
    #
    #     img = obj_img(A_path, dataroot)
    #
    #     img = img * 255 + 10
    #     img = img.astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img.save('/code/mix-pe/cycle_gan/resultsA/{}.png'.format(A_path.split('/')[-1].split('.')[0]))
    #
    # for index in range(len(B_paths)):
    # # for index in range(2):
    #     B_path = B_paths[index % len(B_paths)]  # make sure index is within then range
    #
    #
    #     img = obj_img(B_path, dataroot)
    #
    #     img = img * 255 + 10
    #     img = img.astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img.save('/code/mix-pe/cycle_gan/resultsB/{}.png'.format(B_path.split('/')[-1].split('.')[0]))