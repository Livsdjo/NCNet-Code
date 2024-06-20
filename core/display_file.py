import numpy as np
import cv2

def draw_matches(img0, img1, kpts0, kpts1, match_idx, label, mask,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):

    # Args:
    #     img: color image.
    #     kpts: Nx2 numpy array.
    #     match_idx: Mx2 numpy array indicating the matching index.
    # Returns:
    #     display: image with drawn matches.

    print("标签", label, label.sum())

    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    if 1:
        count = 0
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            if int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 1:

                # print("正确", label_temp[val[0]], mask[val[0]])
                # if(count % 10 != 0):
                #     continue
                cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                # print(pt0, pt1)
                """
                if count >= 30:
                    break
                """
            elif int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 0:
                """
                count += 1
                if count >= 500:
                    continue
                """
                if 1:
                    cv2.circle(display, pt0, radius, (255, 0, 0), thickness)
                    cv2.circle(display, pt1, radius, (255, 0, 0), thickness)
                    cv2.line(display, pt0, pt1, (255, 0, 0), thickness)      # color
                else:
                    pass
            else:
                pass

    display /= 255
    return display



def draw_matches_shuzhi(img0, img1, kpts0, kpts1, match_idx, label, mask,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):

    # Args:
    #     img: color image.
    #     kpts: Nx2 numpy array.
    #     match_idx: Mx2 numpy array indicating the matching index.
    # Returns:
    #     display: image with drawn matches.

    print("标签", label, label.sum())

    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    # 之前是行
    display = np.zeros((rows0 + rows1, max(cols0, cols1), 3))
    print("resize", resize1.shape, resize0.shape)
    display[:rows0, :cols0, :] = resize0
    display[rows0:(rows0 + rows1), :cols0, :] = resize1
    """
    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1
    """

    if 1:
        count = 0
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            # 原来是行
            # pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))
            pt1 = (int(kpts1[val[1]][0]), int(kpts1[val[1]][1]) + rows0)

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            # if int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 1:
            if int(label_temp[val[0]]) == 1:
                count += 1

                """
                if(count % 10 != 0):
                    continue
                """

                # print("正确", label_temp[val[0]], mask[val[0]])
                if 1:
                # if count % 2 == 0:
                    cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                    cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                    cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                else:
                    cv2.circle(display, pt0, radius, (0, 0, 255), thickness)
                    cv2.circle(display, pt1, radius, (0, 0, 255), thickness)
                    cv2.line(display, pt0, pt1, (0, 0, 255), thickness)      # color
                # print(pt0, pt1)
                """
                if count >= 30:
                    break
                """
            # elif int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 0:
            elif int(label_temp[val[0]]) == 0:
            # elif 0:
                """
                count += 1
                if count % 3 == 0:
                    continue
                """

                if 1:
                    cv2.circle(display, pt0, radius, (0, 0, 255), thickness)
                    cv2.circle(display, pt1, radius, (0, 0, 255), thickness)
                    cv2.line(display, pt0, pt1, (0, 0, 255), thickness)      # color
                else:
                    pass
            else:
                pass

            # name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay/" + "dirplay_" + str(count) + ".png"
            # name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay3/" + "display_10.png"
            # name = "C:/Users/Administrator.DESKTOP-3TQ2JAH/Desktop/author_huitu/paint2/Figure_4.png"

        print("个数", count)

    display = display[:, :, ::-1]
    # cv2.imwrite(name, display)
    display /= 255
    return display




def image_join(image1, image2):
    """
    水平合并两个opencv图像矩阵为一个图像矩阵
    :param image1:
    :param image2:
    :return:
    """
    h1, w1 = image1.shape[0:2]
    h2, w2 = image2.shape[0:2]

    if h1 > h2:
        margin_height = h1 - h2
        if margin_height % 2 == 1:
            # margin_top = int(margin_height / 2)
            # margin_top = margin_height
            # margin_bottom = margin_top + 1
            margin_top = 0
            margin_bottom = margin_height
        else:
            # margin_top = margin_bottom = int((h1 - h2) / 2)
            # margin_top = margin_height
            margin_top = 0
            margin_bottom = margin_height
        image2 = cv2.copyMakeBorder(image2, margin_top, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # print("hehhe", image2.shape)
    elif h2 > h1:
        margin_height = h2 - h1
        if margin_height % 2 == 1:
            # margin_top = int(margin_height / 2)
            # margin_top = margin_height
            # margin_bottom = margin_top + 1
            margin_top = 0
            margin_bottom = margin_height
        else:
            # margin_top = margin_bottom = int(margin_height / 2)
            # margin_top = margin_height
            margin_top = 0
            margin_bottom = margin_height
        image1 = cv2.copyMakeBorder(image1, margin_top, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return np.concatenate((image1, image2), axis=1)





