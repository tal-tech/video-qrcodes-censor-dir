from video_block_qr_zone import *
import os

if __name__ == "__main__":
    '''
    video_block.get_qr(img): detect qrcode of a image
    video_block.cover_img(img, tpl, res[1]): cover an image with logo image
    '''
    video_path = "input.mp4"
    image_dir = "images"
    cmd = "ffmpeg -i %s -vf fps=1 %s/out%%d.jpg" % (video_path,image_dir)
    os.system(cmd)
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        if os.path.isfile(filepath):
            print(f'File: {filepath}')
            video_block = VideoBlock()
            image = cv2.imread(filepath)
            res = video_block.get_qr(image)
            print('qr position: ',res)
    # tpl = cv2.imread('./mask_image.jpeg')
    # for i in range(1000):
    #     _, img = video_block.cover_img(img, tpl, res[1])
    # cv2.imwrite('./res.jpg', img)