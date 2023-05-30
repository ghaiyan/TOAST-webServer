import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import hdbscan
import os
import zipfile
from flask import Flask, render_template, request, send_file

mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


app = Flask(__name__, static_folder='static')
@app.route('/', methods=['GET', 'POST'])
# 默认的接触矩阵文件和分辨率

def index():
    if request.method == 'POST':
        # 获取输入的分辨率和接触矩阵文件
        resolution = request.form['resolution']
        #resolution = request.form['resolution'] or default_resolution
        contact_matrix = request.files['contact_matrix']
        
        #contact_matrix = request.files['matrix_file'] or default_matrix_file

        # 保存接触矩阵文件
        contact_matrix.save(contact_matrix.filename)
           # 判断文件是否为nxn格式的矩阵文件

        # 处理输入的数据并生成批量文件的压缩包
        generated_files = process_matrix_file(contact_matrix.filename,resolution)
        zip_filename = create_zip(generated_files,"zip_files.zip")
        download_link = "/zip_files.zip"  # 下载链接
        os.remove(contact_matrix.filename)
        return render_template('index.html', download_link=download_link)

    # 渲染页面
    return render_template('index.html', download_link=None)


#创建zip文件
def create_zip(files, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for file in files:
            zip_file.write(file)

    # 删除生成的单个文件
    for file in files:
        os.remove(file)

# 定义函数，用于将一个大矩阵分割成多个指定大小的子矩阵
def split_matrix(matrix, submatrix_size):
    # 获取大矩阵的大小
    n = matrix.shape[0]
    # 计算子矩阵的个数
    num_submatrices = int(np.ceil(n/submatrix_size))
    # 定义一个空列表，用于存储所有的子矩阵
    submatrices = []
    # 循环遍历每个子矩阵
    for i in range(num_submatrices):
        # 计算当前子矩阵在大矩阵中的起始索引和结束索引
        start_idx = i * submatrix_size
        end_idx = min((i+1) * submatrix_size, n)
        # 从大矩阵中提取当前子矩阵，并将其加入到子矩阵列表中
        submatrix = matrix[start_idx:end_idx, start_idx:end_idx]
        submatrices.append(submatrix)
    # 返回所有子矩阵组成的列表
    return submatrices


# 构建GAN模型
class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, adj_matrix):
        embedding = self.encoder(adj_matrix)
        adj_hat = self.decoder(embedding)
        return embedding, adj_hat

def train_gae(adj_matrix,lr):
    input_dim = adj_matrix.shape[0]
    hidden_dim = 64
    embedding_dim = 16
    num_epochs = 200

    model = GAE(input_dim, hidden_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    adj_matrix = torch.FloatTensor(adj_matrix)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _, adj_hat = model(adj_matrix)
        loss = criterion(adj_hat, adj_matrix)
        loss.backward()
        optimizer.step()

    embedding, _ = model(adj_matrix)
    return embedding.detach().numpy()



def boundaryPlot(labels):
    n = len(labels)
    boundary = np.zeros(n)
    i = 0
    label = -1
    start = 0
    while i < n:
        if labels[i] == label:
            boundary[i] = start
        else:
            start = i
            label = labels[i]
            boundary[i] = i
        i = i + 1
    return boundary

#get TAD file
def getTAD(tadfile,res,label):
    boundaries=boundaryPlot(label)
    print(boundaries)
    
    i = 0
    with open(tadfile, "w") as out:
        while i < len(boundaries):
            if boundaries[i] < i:
                start = i - 1
                while i<len(boundaries) and boundaries[i] == start:
                    end = i
                    i = i + 1
                if end-start>=(2):
                    startbin = int(start) * int(res)
                    endbin = int(end) * int(res)
                    out.write("\t".join((str(start), str(startbin), str(end), str(endbin))) + "\n")
                else:
                    start=start-1
            i = i + 1
        out.close()

def readTAD(tadfile):
    #tads = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/{}.txt".format(tadsname)
    f = open(tadfile)
    line=f.readline()
    start=[]
    end=[]
    while line:
        line = line.split()
        start1 = int(line[0])
        end1 = int(line[2])
        start.append(start1)
        end.append(end1)
        line=f.readline()
    f.close()
    return start, end

def getLabel(hicfile,start, end):
    hic=np.load(hicfile)
    n = len(hic)
    labels = np.zeros(n)
    for i in range(n):
        labels[i] = 0
    for j in range(len(start)+1):
        s=start[j-1]
        m=end[j-1]
        labels[s]=2
        labels[m]=2
        for k in range(s+1,m):
            labels[k]=1
    return labels


def getlist(tadfile,ctcf):
    #tad = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/{}.txt".format(name)  
    #tad = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/compare/{}.txt".format(name)
    distances = []
    with open(tadfile) as tad:
        for num, line in enumerate(tad):
            line = line.split()
            start = int(line[1])
            end = int(line[2])
            dist_start = calcuDist(ctcf, start)
            dist_end = calcuDist(ctcf, end)
            if abs(dist_start) <= abs(dist_end):
                distances.append(dist_start)
            else:
                distances.append(dist_end)
        tad.close()
    return list(set(distances))


#plot TAD boundaries
def plot_TAD(hic,tadfile):
    start, end = readTAD(tadfile)
    lentad=len(start)
    palette=sns.color_palette("bright",10)
    #print(labels)
    plt.figure(figsize=(10.5,10))
    start1=1
    end1=399
    sns.heatmap(data=hic[start1:end1, start1:end1], robust=True,cmap="OrRd")
    for i in range(0,lentad):
        if start1<start[i]<end1 and start1<end[i]<end1:
            #print(start[i])
            plt.hlines(y=start[i]-start1,xmin=start[i]-start1,xmax=end[i]-start1)
            plt.vlines(x=end[i]-start1,ymin=start[i]-start1,ymax=end[i]-start1)
    plt.title('TAD boundary')
    plt.savefig(tadfile+".pdf", format='pdf', bbox_inches='tight')
    plt.show()

def tadQuality(tadFile,hic):
    """TAD quality"""
    n = len(hic)
    tad = np.loadtxt(tadFile)
    intra = 0
    intra_num = 0
    for n in range(len(tad)):
        for i in range(int(tad[n,0]),int(tad[n,2]+1)):
            for j in range(int(tad[n,0]),int(tad[n,2]+1)):
                intra = intra + hic[i,j]
                intra_num = intra_num + 1

    if intra_num!=0:
        intra = intra / intra_num
        print("intra TAD: %0.3f" % intra)
    else:
        intra = 0
    
    inter = 0
    inter_num = 0
    for n in range(len(tad) - 1):
        for i in range(int(tad[n,0]),int(tad[n,2]+1)):
            for j in range(int(tad[n+1,0]),int(tad[n+1,2]+1)):
                inter = inter + hic[i,j]
                inter_num = inter_num + 1
    if inter_num!=0:
        inter = inter / inter_num
        print("inter TAD: %0.3f" % inter)
    else:
        inter = 0
    print("quality: %0.3f" % (intra - inter))
    quality=intra - inter
    return quality

def process_matrix_file(hic,res):
    generated_files = []
    # 读取矩阵文件并处理
    # 假设矩阵文件每行包含逗号分隔的数字
    matrix = np.loadtxt(hic)
    num_rows, num_cols = matrix.shape
    num = (num_cols - 1) // 400 + 1
    # 划分子矩阵
    sub_matrices = split_matrix(matrix,400)
    qualityFile='Toast_quality.txt'
    #保存所有子矩阵
    for i in range(len(sub_matrices)):
        matFile='sub_matrix_{}.txt'.format(i)
        generated_files.append(matFile)
        # 保存为txt格式
        np.savetxt(matFile, sub_matrices[i], fmt='%f')
        # 读取400x400的对称正定矩阵
        # 加载邻接矩阵
        adj_matrix = np.loadtxt(matFile)
        submatricNumber=i
        # 随机初始化节点特征向量
        feature_matrix = torch.eye(400)
        # 训练图自编码器并得到节点特征嵌入
        lr=0.001
        embedding = train_gae(adj_matrix,lr)
        # 对节点特征进行HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(metric="euclidean",gen_min_span_tree=True)
        node_labels = clusterer.fit_predict(embedding)
        print(node_labels)

        #get TAD file
        tadfile="{}_{}.tad".format(submatricNumber, res)
        generated_files.append(tadfile)

        getTAD(tadfile,res,node_labels)
        #plot TADs
        plot_TAD(adj_matrix,tadfile)
        figureTAD="{}_{}.tad.pdf".format(submatricNumber, res)
        generated_files.append(figureTAD)
        start, end = readTAD(tadfile)
        lentad=len(start)
        if int(lentad)>2:
            quality=tadQuality(tadfile,adj_matrix)
            with open(qualityFile,'a+') as f:
                    f.write("\t".join((str("submatricNumber="),str(submatricNumber),str("TAD number="),str(lentad),str("TADQuality="),str(quality)))+'\n')
                    f.close()    
    generated_files.append(qualityFile)
    return generated_files

@app.route('/download_sample')
def download_sample():
    # 提供样本文件的下载
    sample_file = 'sample/4noise.hic'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample1')
def download_sample1():
    # 提供样本文件的下载
    sample_file = 'sample/40_TADlike_alpha_50_set6.mat'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample2')
def download_sample2():
    # 提供样本文件的下载
    sample_file = 'sample/chr19_kr_50kb.txt'
    return send_file(sample_file, as_attachment=True)

@app.route('/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run()