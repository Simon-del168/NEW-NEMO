# 使用本地基础镜像
FROM tensorflow/tensorflow:2.10.1-gpu

LABEL description="for nemo"

# 安装必要的软件包
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    ffmpeg \
    vim \
    tmux \
    yasm \
    android-tools-adb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 Anaconda
ENV PATH=/opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda config --add channels defaults

# 安装 awesomevim
RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime && \
    sh ~/.vim_runtime/install_awesome_vimrc.sh && \
    echo "stty -ixon" >> ~/.bashrc

# 安装 youtube-dl
#RUN curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl && \
#    chmod a+rx /usr/local/bin/youtube-dl

# install yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
RUN chmod a+rx /usr/local/bin/yt-dlp

# 创建 conda 环境
ADD environment1.yml /environment1.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f /environment1.yml
   # conda run -n nemo_py3.10 pip install tensorflow==2.12.0

ADD environment2.yml /environment2.yml
RUN conda env create -f /environment2.yml

# 下载 Android NDK
#RUN wget https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip -O /root/android-ndk-r14b.zip && \
#    unzip /root/android-ndk-r14b.zip -d /root/ && \
#    ln -s /root/android-ndk-r14b/toolchains/llvm/prebuilt/linux-x86_64 /android-ndk-r14b/toolchains/llvm/prebuilt/linux-x86 && \
#    ln -s /root/android-ndk-r14b/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 /android-ndk-r14b/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86
RUN wget https://dl.google.com/android/repository/android-ndk-r26c-linux.zip -O /root/android-ndk-r26c.zip 
RUN unzip /root/android-ndk-r26c.zip
RUN ln -s /root/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64 /android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86
#RUN ln -s /android-ndk-r26c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 /android-ndk-r26c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86
RUN ln -s /root/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64 /android-ndk-r26c/toolchains/aarch64-linux-android


# 设置环境变量
ENV NEMO_CODE_ROOT=/workspace
ENV NEMO_DATA_ROOT=/workspace/nemo-data
ENV SNPE_ROOT=$NEMO_CODE_ROOT/third_party/snpe
ENV PYTHONPATH=${NEMO_CODE_ROOT}:${SNPE_ROOT}/lib/python:${PYTHONPATH:-/default/path}
ENV PATH=/android-ndk-r26c/build/:$PATH

# 设置工作目录
WORKDIR /workspace

# 使用 bash 作为运行命令的 shell
SHELL ["/bin/bash", "-c"]
(base) root@guest-Super-Ser
