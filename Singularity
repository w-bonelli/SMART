BootStrap: docker
From: conda/miniconda3

%labels
  Maintainer: Suxing Liu

%setup
  mkdir ${SINGULARITY_ROOTFS}/opt/code/

%files
  ./* /opt/code/

%post
  conda install -c anaconda numpy
  conda install -c anaconda pillow
  conda install -c anaconda scipy
  conda install -c anaconda scikit-image
  conda install -c anaconda scikit-learn
  conda install -c conda-forge matplotlib
  conda install -c wheeler-microfluidics opencv-python
  conda install -c anaconda openpyxl

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
  
