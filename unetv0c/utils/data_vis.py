import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 4, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 4, 2)
    b.set_title('Output mask1')
    plt.imshow(mask[0,:,:])

    c = fig.add_subplot(1, 4, 3)
    c.set_title('Output mask2')
    plt.imshow(mask[1,:,:])

    d = fig.add_subplot(1, 4, 4)
    d.set_title('Output mask3')
    plt.imshow(mask[2,:,:])
    plt.show()
