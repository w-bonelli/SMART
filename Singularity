Bootstrap: docker
From: ubuntu:18.04

%labels
	Maintainer: Suxing Liu

%setup
	mkdir ${SINGULARITY_ROOTFS}/opt/code/

%files
	./* /opt/code/

%post
	apt update && \
	apt install -y \
		build-essential \
		python3-setuptools \
		python3-pip \
		python3-numexpr \
		libgl1-mesa-glx \
		libsm6 \
		libxext6 \
		libfontconfig1 \
		libxrender1

	pip3 install numpy \
		Pillow \
		scipy \
		scikit-image \
		scikit-learn \
		matplotlib \
		opencv-python \
		openpyxl

	mkdir /lscratch /db /work /scratch
  
	chmod -R a+rwx /opt/code/
  
%environment
	PYTHONPATH=$PYTHONPATH:/opt/code/
	export PATH
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/code/
	export LD_LIBRARY_PATH

%runscript
	echo "Arguments received: $*"
	exec /usr/bin/python "$@"
  
