'''
导入模块说明：
torch / nn: 基础深度学习库。

transforms: 对图像进行缩放、张量化、归一化处理（匹配卷积输入要求）。

create_dataset.My_dataset: 自定义的数据加载类。

model64: 包含了生成器（Generator）和判别器（Discriminator）的具体结构。
'''

import torch
import torch.nn as nn
from torchvision import transforms
from create_dataset import My_dataset, save_img
from torch.utils.data import DataLoader
from model64 import Generator, Discriminator


'''
第一阶段：图像初始化
其中功能包括利用transforms模块来进行缩放统一图像大小，
图像张量化，
以及图像的归一化。
'''
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 网络设置图片大小为 64*64,保证图片大小符合网络结构要求
    transforms.ToTensor(), # 图像的张量化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 图像的归一化
])

'''
第二阶段：准备数据集与超参数
注：超参数（Hyperparameters）是在训练前手动指定的训练控制量，不能通过模型自动学习得到。 
'''
dataset = My_dataset(r'C:\Users\ap\Desktop\image', transform=transform)   # 这一步是准备数据集，让后续训练可以按批次读取图像。
batch_size, epochs = 32, 500   # 设置两个超参数：batch_size指每一批喂入多少张图像，epochs指的是总共训练多少轮
'''
DataLoader的作用是包装数据集，让它能被自动分批次地加载。
其中的参数代表的含义是：
dataset=dataset----使用哪个数据集
batch_size=batch_size----每次加载多少张图片
shuffle=True----是否打乱顺序（是，为了防止模型记住图片顺序）
drop_last=True----是否丢掉最后不足一整批的数据（是，避免最后一个 batch < 32 出错）
'''
my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)



'''
第三阶段：模型初始化 + 优化器设置
'''
discriminator = Discriminator()  # 新建一个判别器对象
generator = Generator()  # 新建一个生成器对象

if torch.cuda.is_available():   # 如果你有 GPU，则
    discriminator = discriminator.cuda()  # 将判别器转为GPU运行
    generator = generator.cuda()  # 将生成器转为GPU运行

'''
使用 Adam 优化器（比 SGD 收敛更快）
作用对象是判别器 discriminator.parameters() 中所有可训练参数
lr=1e-4：学习率（超参数），代表步长
betas=(0.5, 0.99)：Adam 的动量参数，常用于训练 GAN 更稳定。两个β值都是滑动平均的权重，用于决定过去数据和现有数据的参考比值
'''
d_optimizer = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.99), lr=1e-4)  # betas为adam算法两个动量参数
g_optimizer = torch.optim.Adam(generator.parameters(), betas=(0.5, 0.99), lr=1e-4)
criterion = nn.BCELoss()  # 损失函数（二元交叉熵）

'''
第四阶段：开始训练
'''

for epoch in range(epochs):

    for i, img in enumerate(my_dataloader):

        '''准备训练判断器的数据'''
        noise = torch.randn(batch_size, 100).cuda()  # 随机噪声作为输入
        real_img = img.cuda()  # 真图像
        fake_img = generator(noise)  # 利用随机噪声和生成器生成一个假图像

        real_out = discriminator(real_img)  # 判断器对真图像的输出
        fake_out = discriminator(fake_img)  # 判断器对假图像的输出
        real_label = torch.ones_like(real_out).cuda()  # 真实图片标签为1
        fake_label = torch.zeros_like(fake_out).cuda()  # 假图片标为0
        real_loss = criterion(real_out, real_label)  # 这是判别器 在真图像上的损失，如果 real_out 很小（误判为假），那损失就大，D 就会被罚。
        fake_loss = criterion(fake_out, fake_label)  # 这是判别器 在假图像上的损失，如果 fake_out 很高（误判为真），那损失就大。

        '''训练判断器'''
        d_loss = real_loss + fake_loss  # 判别器这一轮的总损失
        d_optimizer.zero_grad()  # 清空判别器旧的梯度

        d_loss.backward()  # 反向传播
        d_optimizer.step()  # 更新判别器D的参数

        '''准备训练生成器的函数'''
        noise = torch.randn(batch_size, 100).cuda()  # 根据图像规格随机生成噪声
        fake_img = generator(noise)  # 利用生成器和随机噪声生成假图片
        output = discriminator(fake_img)  # 接收判断器对于该图片的判断

        '''训练生成器'''
        g_loss = criterion(output, real_label)   # 通过判别器对假图像的判断和真图像计算损失
        g_optimizer.zero_grad()  # 清空生成器优化器上的旧梯度

        g_loss.backward()  # 反向传播
        g_optimizer.step()  # 更新生成器G的参数

        if (i + 1) % 5 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D_real: {:.6f},D_fake: {:.6f}'.format(
                epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                real_out.data.mean(), fake_out.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0 and i == len(my_dataloader) - 1:          # 保存真实图像
            save_img(img[:64, :, :, :], './sample/real_images.png')
        if (epoch+1) % 50 == 0 and i == len(my_dataloader)-1:             # 每50个epoch保存一次预测图像
            save_img(fake_img[:64, :, :, :], './sample/fake_images_{}.png'.format(epoch + 1))

torch.save(generator.state_dict(), './generator.pth')        # 保存权重文件
torch.save(discriminator.state_dict(), './discriminator.pth')
