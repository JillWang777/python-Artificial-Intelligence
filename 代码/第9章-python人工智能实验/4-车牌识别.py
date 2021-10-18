# 9.4 车牌识别
import click
from hyperlpr import pipline as pp
import time
import cv2
from hyperlpr import pipline as pp
from tkinter import filedialog
import tkinter as tk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'是为了解决CPU使用tensorflow不能兼容AVX2指令的问题
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 本实验使用了tkinter库，tkinter是Python自带的可用于GUI编程的库
# 利用tkinter可以建立一个选择车牌图片的对话框
root = tk.Tk()
root.withdraw()

# 生成对话框，选择车牌图片，将路径和文件名以字符串类型返回给filename，并打印出来
filename = filedialog.askopenfilename()
print(filename)

# 使用hyperlpr库识别刚才选择的车牌图片
image = cv2.imread(filename)
image, res = pp.SimpleRecognizePlateByE2E(image)

# 将识别的结果输出
print(res)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'是为了解决CPU使用tensorflow不能兼容AVX2指令的问题
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = tk.Tk()
root.withdraw()

# 生成对话框，选择车牌图片，将路径和文件名以字符串类型返回给filename，并打印出来
filename = filedialog.askopenfilename()
print(filename)


@click.command()
@click.option('--video', help='input video file')
def main(video):
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    stream = cv2.VideoCapture(filename)
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream
        grabbed, frame = stream.read()
        if not grabbed:
            print('No data, break.')
            break

        _, res = pp.SimpleRecognizePlate(frame)

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb = imutils.resize(frame, width = 750)
        # r = frame.shape[1] / float(rgb.shape[1])

        cv2.putText(frame, str(res), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    stream.release()


if __name__ == '__main__':
    main()
