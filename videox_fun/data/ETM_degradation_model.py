import numpy as np
import cv2
import numpy as np
from poisson_noise import random_add_poisson_noise_pt  # 确保正确导入
import numpy as np
import cv2
import scipy.special
from scipy.signal import convolve2d


def airy_disk_psf(size=21, wavelength=550e-9, aperture_diameter=2e-3, pixel_size=1.12e-6):
    """生成基于艾里斑（Airy Disk）的 PSF"""
    x = np.linspace(-size//2, size//2, size) * pixel_size
    y = np.linspace(-size//2, size//2, size) * pixel_size
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    k = np.pi * aperture_diameter / wavelength  # 衍射比例因子
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = (2 * scipy.special.j1(k * R) / (k * R))**2
    psf[R == 0] = 1  # 处理中心值
    psf /= psf.sum()  # 归一化
    return psf

def apply_diffraction_blur(image, psf):
    """对图像应用衍射模糊"""
    blurred = cv2.filter2D(image, -1, psf)
    return blurred


def add_poisson_noise(image, poisson_level=1.0):
    """
    给输入图像添加泊松噪声，并允许控制噪声强度。

    :param image: 输入灰度图像 float32, 0-255
    :param poisson_level: 泊松噪声强度控制参数, 默认1.0(标准泊松噪声）
    :return: 添加泊松噪声后的图像
    """
    image = np.clip(image, 0, 1.0)  # 限制像素值在 0-255
    scaled_image = image * poisson_level  # 调整噪声强度
    noisy = np.random.poisson(scaled_image).astype(np.float32) / poisson_level  # 生成泊松噪声并归一化
    noisy = np.clip(noisy, 0, 1.0)  # 限制像素值范围

    return noisy


def generate_poisson_noise(img, scale=1.0):
    """Generate poisson noise.

    Reference: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.

    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """

    # round and clip image for counting vals correctly
    img = np.clip(img, 0, 255) / 255. # 限制像素值在 0-1

    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = out - img
    # noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    res = noise * scale
    res = np.clip(res * 255., 0, 255)

    return res


def add_poisson_noise_basicsr(img, scale=1.0, clip=True, rounds=True):
    """Add poisson noise.

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.

    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise(img, scale)
    out = img + noise
    if clip and rounds:
        out = np.clip(out.round(), 0, 255) 
    elif clip:
        out = np.clip(out, 0, 255)
    elif rounds:
        out = out.round()
    return out



def add_gaussian_noise(image, mean=0, var=0.1):  
    """  
    给图像添加高斯噪声  
    :param image: 输入图像（灰度图）  
    :param mean: 噪声的均值  
    :param var: 噪声的方差  
    :return: 添加噪声后的图像  
    """  
    gauss = np.random.normal(mean, var ** 0.5, image.shape)  
    gauss = gauss.reshape(image.shape).astype(np.float32)  
    noisy = cv2.add(image, gauss)  
    return np.clip(noisy, 0, 1) 

def add_salt_pepper_noise(image, salt_prob=0.00, pepper_prob=0.005):  
    """  
    向图片添加椒盐噪声  
    :param image: np, h,w,(c)  
    :param salt_prob: 添加盐噪声的概率  
    :param pepper_prob: 添加椒噪声的概率  
    :return: 添加了椒盐噪声的图片  
    """ 
    img_copy = image.copy()  # 创建一个图片的副本以避免修改原始图片  
    # 获取图片的高度和宽度  
    h, w = img_copy.shape[:2]  
      
    # 盐噪声  
    num_salt = np.ceil(salt_prob * h * w)  
    coords = [np.random.randint(0, h, int(num_salt)),  
              np.random.randint(0, w, int(num_salt))]  
    salt_mask = np.zeros(image.shape[:2], dtype=bool)  
    salt_mask[tuple(coords)] = True  
    img_copy[salt_mask] = 1  # 将选中的点设置为白色（盐）  
  
    # 椒噪声  
    num_pepper = np.ceil(pepper_prob * h * w)  
    coords = [np.random.randint(0, h, int(num_pepper)),  
              np.random.randint(0, w, int(num_pepper))]  
    pepper_mask = np.zeros(image.shape[:2], dtype=bool)  
    pepper_mask[tuple(coords)] = True  
    img_copy[pepper_mask] = 0  # 将选中的点设置为黑色（椒）  
  
    return np.clip(img_copy, 0, 1)
 
def temporal_noise(img, c_mean_range=[0.15,0.25], c_variance=0.01):
    """
    加入由于c抖动带来的噪声
    img[np.array, np.float32]: np array of a gray image, shape: (h,w), 0-1.0,
    c_mean_range[list]: float number betwenn 0-1
    c_variance: the variance of c
    """
    ## add c_variance noise
    c_mean = np.random.uniform(low=c_mean_range[0], high=c_mean_range[1])
    c = np.random.normal(loc=c_mean, scale=np.sqrt(c_variance), size=(img.shape[0], img.shape[1]))
    delta_c = c - c_mean
    noised_img = img * np.exp(delta_c)
    noised_img = np.clip(noised_img, 0, 1)

    return noised_img

def gt2etm(img, temporal_noise_FLG=True, c_mean_range=[0.15,0.25], c_variance=0.01, 
           salt_pepper_noise_FLG = True, salt_prob=0.000001, pepper_prob=0., 
           gaussian_noise_FLG = True, gaussian_mean=0., gaussian_var=0.01, 
           poisson_noise_FLG=True, poisson_level=1.0,
           diffraction_blur_FLG=True, diffraction_psf_size=21,
           return_etm_gt_FLG=False, img_format='BGR'):
    """
    img: h,w,c, c=3, range 0-1, BGR
    return: etm: h,w,1, float32, range 0-1
    """
    
    # convert to gray img
    if img_format == 'BGR':
        gray = 0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]  
    elif img_format == 'RGB':
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2] 
    elif img_format == 'GRAY' and img.ndim == 2:
        gray = img
    elif img_format == 'GRAY' and img.ndim == 3:
        gray = img[:, :, 0]
    else:
        raise ValueError(f"img_format: {img_format} is not supported.")

    if return_etm_gt_FLG:
        etm_gt = gray
        etm_gt = etm_gt[:, :, np.newaxis]
    
    ## Diffraction Blur
    if diffraction_blur_FLG:
        psf = airy_disk_psf(size=diffraction_psf_size)
        gray = apply_diffraction_blur(gray, psf)
    
    ## Temporal Noise
    if temporal_noise_FLG:
        gray = temporal_noise(gray, c_mean_range=c_mean_range, c_variance=c_variance)

    ## Salt & Pepper noise
    if salt_pepper_noise_FLG:
        gray = add_salt_pepper_noise(gray, salt_prob=salt_prob, pepper_prob=pepper_prob)

    ## Gaussian Noise
    gray = gray.astype(np.float32)
    if gaussian_noise_FLG:
        gray = add_gaussian_noise(gray, mean=gaussian_mean, var=gaussian_var)

    ## Poisson Noise（可选）
    if poisson_noise_FLG and poisson_level > 0:
        print(f'[DEBUG]: gray type: {gray.dtype}')
        print(f'[DEBUG]: gray shape: {gray.shape}')
        print(f'[DEBUG]: gray max value: {gray.max()}')

        gray = add_poisson_noise(gray, poisson_level)
        # gray = add_poisson_noise_basicsr(gray, poisson_level)
        print(f'[DEBUG]: gray type: {gray.dtype}')
        print(f'[DEBUG]: gray shape: {gray.shape}')
        print(f'[DEBUG]: gray max value: {gray.max()}')


    # Convert to h,w,1
    gray = gray[:, :, np.newaxis]
    gray = np.clip(gray, 0, 255)

    if return_etm_gt_FLG:
        return gray, etm_gt
    else:
        return gray


if __name__=='__main__':

    # 设定参数
    return_etm_gt_FLG = False
    temporal_noise_FLG = True
    c_mean_range = [0.15,0.25]
    c_variance = 0.005
    salt_pepper_noise_FLG = True
    salt_prob = 1e-5 #0.00
    pepper_prob = 0
    gaussian_noise_FLG = False # 不需要高斯噪声，不需要任何加性噪声
    gaussian_mean = 0.
    gaussian_var =0.01
    poisson_noise_FLG = False
    poisson_level = 1000 # 100 - 400
    diffraction_blur_FLG = True
    diffraction_psf_size=8 #4-10


    # 读取图像并归一化
    img_path = "./test.png" 
    img_gt = cv2.imread(img_path).astype(np.float32) / 255.0  # 归一化到 0-1 之间

    # 生成 ETM 图像
    etm = gt2etm(img_gt, temporal_noise_FLG, c_mean_range, c_variance, # temporal noise
                salt_pepper_noise_FLG, salt_prob, pepper_prob, # salt pepper noise
                gaussian_noise_FLG, gaussian_mean, gaussian_var,  # gaussian noise
                poisson_noise_FLG=poisson_noise_FLG, poisson_level=poisson_level, # poisson noise
                diffraction_blur_FLG=diffraction_blur_FLG, diffraction_psf_size=diffraction_psf_size, # diffraction blur
                return_etm_gt_FLG = return_etm_gt_FLG)
    
    # 保存 ETM 图像
    output_path = 'light_alldegradation.png'
    etm_uint8 = (etm * 255).astype(np.uint8)
    cv2.imwrite(output_path,  etm_uint8)
    print(f"ETM saving: {output_path}")
