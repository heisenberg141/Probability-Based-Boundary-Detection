import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from sklearn.cluster import KMeans


def main():
    # n_orientations = 16
    # DoG_Filter_Bank = generate_dog_filter_bank(n_orientations)
    # display_and_save_filters(DoG_Filter_Bank,3,len(DoG_Filter_Bank)/3,"DOG_Filter_Bank.png")
    
    # # LM filter bank
    # LM_small = generate_lm_filter_bank("small")
    # LM_large = generate_lm_filter_bank("large")
    # display_and_save_filters(LM_small,4,len(LM_small)/4,"LM_small.png")
    # display_and_save_filters(LM_large,4,len(LM_small)/4,"LM_large.png")

    # #Gabor Filter Bank
    # Gabor_Filter_Bank = generate_gabor_filter_bank(n_orientations)
    # display_and_save_filters(Gabor_Filter_Bank,3,len(Gabor_Filter_Bank)/3,"Gabor_Filter_Bank.png")
    angles_list = np.arange(0,360,20)
    for angle in angles_list:
        Filters.half_disk(9,angle)
    # sigma_y = 5
    # sigma_x = sigma_y 
    # k_size = 49
    # d_order = 2
    # gaussian_kernel = Filters.gaussian_2D(sigX = sigma_x, sigY =  sigma_y,kernel_shape=(k_size,k_size),d_order=d_order)
    # log_kernel = Filters.laplace_gauss_2d(sigma_y,(k_size,k_size))
    # gabor = Filters.gabor(k_size = k_size, lam=10,theta = np.pi/6, psi = 0, sigma = sigma_x,gamma = 0.5)
    # plt.imshow(gabor,cmap='gray')
    # plt.show()

def generate_half_disk_bank(radii,n_orients):
    
    left_filters = list()
    right_filters = list()
    angles_list = np.arange(0,180,360/n_orients)
    for radius in radii:
        for theta in angles_list:
            left_filters.append(cv2.bitwise_not(Filters.half_disk(radius, theta)))
            right_filters.append(cv2.bitwise_not(Filters.half_disk(radius,180+theta)))
    return [left_filters,right_filters]
            
def generate_gabor_filter_bank(n_orientations):
    filter_bank = list()
    sigmas = [2,2*2**0.5,4]
    k_size = 21
    angles = np.arange(0, 180, int(180/n_orientations)+1)
    for sigma in sigmas:
        for angle in angles:
            gabor = Filters.gabor(k_size = k_size, lam=4,theta = angle, psi = 0, sigma = sigma, gamma = 0.8)
            filter_bank.append(gabor)
    
    return filter_bank

def generate_lm_filter_bank(size):
    if size == "small":
        scales = [1, 2**0.5, 2, 2*(2**0.5)]
    else:
        scales = [ 2**0.5, 2, 2*(2**0.5),4]
    k_size = 49
    center = (k_size//2 , k_size//2 )
    n_orientations = 6
    n_scales = 3
    angles = np.arange(0, 180, int(180/n_orientations)+1)
    filter_bank = list()
    for i in range(n_scales):
        sigma_x = scales[i]
        sigma_y = sigma_x*3
        deriv_gaussian = Filters.gaussian_2D(sigX=sigma_x, sigY=sigma_y,kernel_shape=(k_size,k_size),d_order=1)
        dderiv_gaussian = Filters.gaussian_2D(sigX=sigma_x, sigY=sigma_y,kernel_shape=(k_size,k_size),d_order=2)
        for theta in angles:
            R =  cv2.getRotationMatrix2D(center, theta, scale=1)
            rotated_d = cv2.warpAffine(deriv_gaussian, R, deriv_gaussian.shape)
            filter_bank.append(rotated_d)
        for theta in angles:
            R =  cv2.getRotationMatrix2D(center, theta, scale=1)
            rotated_dd = cv2.warpAffine(dderiv_gaussian, R, dderiv_gaussian.shape)
            filter_bank.append(rotated_dd)
    
    for scale in scales:
        log1 = Filters.laplace_gauss_2d(scale, kernel_shape=(k_size,k_size))
        log3 = Filters.laplace_gauss_2d(scale*3, kernel_shape=(k_size,k_size))

        filter_bank.append(log1)
        filter_bank.append(log3)
    
    for scale in scales:
        filter_bank.append(Filters.gaussian_2D(sigX=scale, sigY=scale,kernel_shape=(k_size,k_size)))

            
    
    return filter_bank

def generate_dog_filter_bank(orientations):
    sigmas = [1,2**(1/2),2]
    k_size = 7
    sobel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    center = (k_size//2 , k_size//2 )
    filter_bank = list()
    for sigma in sigmas:
        gaussian = Filters.gaussian_2D(sigX=sigma,sigY=sigma,kernel_shape = (k_size, k_size))
        DoG_filter = cv2.filter2D(gaussian,-1,sobel_x)
        angles = np.arange(0, 180, int(180/orientations)+1)
        oriented_filters = list()
        for theta in angles: 
            R =  cv2.getRotationMatrix2D(center, theta, scale=1)
            dog_filter_rotated = cv2.warpAffine(DoG_filter, R, DoG_filter.shape)
            # oriented_filters.append(dog_filter_rotated)
            filter_bank.append(dog_filter_rotated)
    return filter_bank 

def display_and_save_filters(masks, rows, cols, file_name):
    """
    Displays generated filters together and save as an image.
    """
    for i in range(len(masks)):
        plt.subplot(rows, cols, i+1)
        plt.axis('off')
        plt.imshow(masks[i], cmap='gray')
    plt.savefig(file_name)
    plt.show()
    plt.close()  

def generate_gradient_map(map,K,half_disks):
    lefts, rights = half_disks
    
    chi_square_list = list()
    for i in range(len(lefts)):
        chi_square = np.zeros(map.shape)
        for val in range(K):
            bin_img = (map == val)
            bin_img = np.float32(bin_img)
            # cv2.imshow("bin_image",bin_img)

            gi = cv2.filter2D(src=bin_img, ddepth=-1, kernel=np.uint8(lefts[i]))
            # cv2.imshow("Gi",gi)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            hi = cv2.filter2D(src=bin_img, ddepth=-1, kernel=np.uint8(rights[i]))

            chi_square += ((gi - hi) ** 2 )/2 * (gi + hi)
        chi_square_list.append(chi_square)

    chi_square_list = np.array(chi_square_list)
    return np.mean(chi_square_list, axis=0)
            
def generate_texton_map(img, D, LM, G):
    filter_responses = list()

    for filter in D:
        conv = cv2.filter2D(img,-1, filter)
        filter_responses.append(conv)
    for filter in LM:
        conv = cv2.filter2D(img,-1, filter)
        filter_responses.append(conv)
    for filter in G:
        conv = cv2.filter2D(img,-1, filter)
        filter_responses.append(conv)
    
    filter_responses = np.array(filter_responses)  

    n, r, c = filter_responses.shape
    filter_responses = np.reshape(filter_responses,(n,r*c)).T

    km = KMeans(n_clusters=64, n_init=2)
    labels = km.fit_predict(filter_responses)
    t_map = labels.reshape([r,c])
    return t_map
    plt.imshow(t_map)
    plt.show()
    
class Filters:
    def gaussian_2D(sigX, sigY, kernel_shape, d_order = 0):
        x_size = kernel_shape[0]
        y_size = kernel_shape[1]
        gauss_x = Filters.gaussian_1d(x_size, sigX,d_order=d_order)
        gauss_y = Filters.gaussian_1d(y_size, sigY)
        gaussian_kernel = np.outer(gauss_x, gauss_y)
        if d_order==0:
            gaussian_kernel/=np.sum(gaussian_kernel)
        return gaussian_kernel
    
    def gaussian_1d( k_size,sigma,d_order = 0):
        x = np.arange(0,k_size)
        mu = k_size//2
        normalizer = 1/((2*np.pi)**0.5)/sigma
        power_n = -0.5 * ( (mu-x) / sigma)**2
        gaussian = normalizer*np.exp(power_n)
        gaussian/=np.sum(gaussian)
        if d_order == 1:
            gaussian =-gaussian*(mu-x)/(sigma**2)
        if d_order ==2:
            gaussian = gaussian*(((mu-x)**2) - sigma)/(sigma**2)    
        return gaussian
    
    def laplace_gauss_2d(sig, kernel_shape):
        x_size = kernel_shape[0]
        y_size = kernel_shape[1]
        gauss_x = Filters.gaussian_1d(x_size, sig, d_order=0)
        gauss_y = Filters.gaussian_1d(y_size, sig,d_order=2)
        gaussian_kernel1 = np.outer(gauss_y, gauss_x)
        gaussian_kernel2 = np.outer(gauss_x, gauss_y)
        gaussian_kernel = gaussian_kernel1+gaussian_kernel2
        return gaussian_kernel
    
    def gabor(k_size,lam, theta, psi, sigma, gamma = 1):
        center = k_size//2
        x, y = np.mgrid[-center:center+1, -center:center+1]
       
        x_dash = x * np.cos(theta) + y * np.sin(theta)
        y_dash = -x * np.sin(theta) + y * np.cos(theta)


        gaussian = np.exp(-1/2/(sigma**2)*(x_dash**2 + gamma**2* y_dash**2)) 
        sinusoid = np.cos(2*np.pi*x_dash/lam + psi)

        return gaussian * sinusoid
    
    def half_disk(radius, angle):
        size = 2*radius + 1
        mask = np.ones([size,size])
        for i in range(radius):
            for j in range(size):
                dist = (i-radius)**2 + (j-radius)**2
                if (dist < radius**2):
                    mask[i,j] = 0
        mask = skimage.transform.rotate(mask,angle,cval=1)
        mask = np.round(mask)
        return mask


        

def create_gaussian():
    return 


if __name__ == '__main__':
    main()
   
 
