#### 路径说明

datasets里存放训练分割网络的数据集。若首次训练，需先运行dataset.py，把标注文件*.json转为掩码*mask.png.
data里存放需处理的图片。以5000为例，5000_ori里存原图，5000_out存分割结果，5000_method1存方法1的中间结果，5000_method1.jpg为最终结果，5000_method1.bin为二进制格式，可用read_result.m打开进行后续处理。

#### 分割模型说明

在segment.py中，把parser.add_argument("--mode", type=str, default='demo')的'demo'改为'train'后，运行segment.py即可训练;
改为'eval'可评估精度，改为'demo'可生成out图。
其他训练参数也在get_train_args()里改。demo的路径在def main(args)里改，有相关注释。

#### 图像处理说明

通过segment.py运行demo，生成的结果在data/*_out. 再运行process.py即可进行处理。
process.py中，参数说明见if __name__ == '__main__'里的注释。
read_result.m里有读取结果bin的代码，可继续处理。