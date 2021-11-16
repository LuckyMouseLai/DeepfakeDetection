import os
from matplotlib import pyplot as plt


"""  
    检查每个视频id文件夹下的图片数量
"""
def check_img_nums(data_root):
    
    methods = os.listdir(data_root)
    for method in methods:
        min = 999999
        nums = 0
        data_root_path = os.path.join(data_root, method)
        ids = os.listdir(data_root_path)
        for id in ids:
            nums = len(os.listdir(os.path.join(data_root_path, id)))
            if nums == 0:
                print(id)
                continue
            min = nums if nums < min else min
        print(method, '=', min)
            

        ### 画图 文件夹数-图片数
        # x = np.arange(0, 1000, 1)  # 文件夹数
        # y = nums  # 图片数量
        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.set(xlabel='dir nums(x)', ylabel='img nums(y)', title=method)
        # ax.grid()
        # fig.savefig(method+'.png')
        # plt.show()

