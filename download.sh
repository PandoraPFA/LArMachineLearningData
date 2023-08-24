function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

if [ -z $1 ]
then
  echo "Error: Expected usage source download.sh <experiment>"
  echo "   where experiment in [ dune sbnd uboone]"
  return 1
fi

if [ -z $MY_TEST_AREA ]
then
  echo "MY_TEST_AREA is not set, can't download the files"
  return 1
fi

if [ ! -d $MY_TEST_AREA/LArMachineLearningData/ ]
then
  echo "LArMachineLearningData does not exist in MY_TEST_AREA: $MY_TEST_AREA, Not downloading the files"
  return 1
fi

### PandoraMVAData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData

# MicroBooNE
if [[ "$1" == "sbnd" ]]
then
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" -O PandoraSvm_v03_11_00.xml && rm -rf /tmp/cookies.txt
fi

# DUNE

if [[ "$1" == "dune" ]]
then
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox" -O PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq" -O PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu" -O PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt

  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19eRtRnbqinLo_90hKMD9wjaHmpXU1m6j' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19eRtRnbqinLo_90hKMD9wjaHmpXU1m6j" -O PandoraBdt_PfoCharacterisation_DUNEFD_HD_v04_06_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RSKy9hGO6BMVQL6NJMcHIk0mAP3MnqOS' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RSKy9hGO6BMVQL6NJMcHIk0mAP3MnqOS" -O PandoraBdt_PfoCharacterisation_DUNEFD_VD_v04_06_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT" -O PandoraBdt_Vertexing_DUNEFD_v03_27_00.xml && rm -rf /tmp/cookies.txt
fi

### PandoraMVAs
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs

# SBND
if [[ "$1" == "sbnd" ]]
then
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7" -O PandoraBdt_v09_32_00_SBND.xml && rm -rf /tmp/cookies.txt
fi



### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

# DUNE

if [[ "$1" == "dune" ]]
then
  download "1kd2QqW2hivCTlKD2pgVCQifsgie8QwD9" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_U_v04_06_00.pt"
  download "16_PRz7Flch9rKyv2b3z4pli4WyhUXKqO" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_V_v04_06_00.pt"
  download "1-_hxLuNO3q59BTIiuiECI2Fq_MMLctEj" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_W_v04_06_00.pt"
  download "1GSWeLcZZ1xBWB9qRFfWbUeqgBw4oJu7o" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_U_v04_06_00.pt"
  download "1v4KpSSekvuhQAKV1S32u_yNCCaQQXWYK" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_V_v04_06_00.pt"
  download "1UGIp1mcIADZujzCW7vxiryanWYF77wjm" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_W_v04_06_00.pt"

  download "15y8Nn66mP8lVO9NP4jZTKNdTTEfJQSrA" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_U_v04_06_00.pt"
  download "1PakLUma5gJ34hXEEvgSMfV0_kky4_IMB" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_V_v04_06_00.pt"
  download "1s2DEXlgNkjTNWUyUXIJm5718OMuj3a4p" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_W_v04_06_00.pt"
  download "1K2zGwPKfPNI0WyArR_RBO3by0rpUijr3" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_U_v04_06_00.pt"
  download "1RTaWO_ilYdqTd-qpjGBAmOKuMobG0tIM" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_V_v04_06_00.pt"
  download "1867oCU1ZxPLiRMQE5SlZDswzfhz8rgSM" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_W_v04_06_00.pt"

  download "1BYX6Y0sirNzBw0qsj6Xsa-qOAdqn5PG7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_U_v04_03_00.pt"
  download "1AWywC9LsCkuNo6lVo5VAWvlM2n-FWTnx" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_V_v04_03_00.pt"
  download "1CLV64endpentEDSoRFTu6obWoYL2Bx-a" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_W_v04_03_00.pt"
  download "1HYhF3xGcZXOVxXrjDnYgV_34-q5d58UC" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_U_v04_03_00.pt"
  download "1DDqVFhnRBPy6ZIehd9u1SwyiKaWC3hSF" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_V_v04_03_00.pt"
  download "1JLr5GfjSNt6vO81ZCv42UHH09hjvtFw7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_W_v04_03_00.pt"
fi

cd $MY_TEST_AREA/LArMachineLearningData