function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

function curl_download() {
   curl -L "https://drive.usercontent.google.com/download?id=$1&confirm=xxx" -o $2
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

if [ -z $1 ]
then
  echo "Error: Expected usage source download.sh <experiment> <workflow>"
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
if [[ "$1" == "uboone" ]]
then
  curl_download "1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" "PandoraSvm_v03_11_00.xml"
fi

# DUNE

if [[ "$1" == "dune" ]]
then
  curl_download "1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox" "PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml"
  curl_download "1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq" "PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml"
  curl_download "12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu" "PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml"

  curl_download "19eRtRnbqinLo_90hKMD9wjaHmpXU1m6j" "PandoraBdt_PfoCharacterisation_DUNEFD_HD_v04_06_00.xml"
  curl_download "1RSKy9hGO6BMVQL6NJMcHIk0mAP3MnqOS" "PandoraBdt_PfoCharacterisation_DUNEFD_VD_v04_06_00.xml"
  curl_download "17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT" "PandoraBdt_Vertexing_DUNEFD_v03_27_00.xml"
fi

### PandoraMVAs
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs

# SBND
if [[ "$1" == "sbnd" ]]
then
  curl_download "1RpIanzW7Z8Blv7ImdPOJkqib2VmIC2Zq" "PandoraBdt_v09_67_00_SBND.xml"
fi

### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

# DUNE

if [[ "$1" == "dune" ]]
then
  if [ -z $2 ]
  then
    echo "Error: Expected usage source download.sh dune <workflow>"
    echo "   where workflow in [ lbl atmos lowe ]"
    return 1
  fi

  if [[ "$2" == "lbl" ]]
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

    download "13oGLdQKE5tFuvh2c0_06xwU8-8qjUQxH" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_U_v04_13_00.pt"
    download "1fmDwr9kJ5RNfIl3u3Dq5P9T5mIy3_ZbB" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_V_v04_13_00.pt"
    download "1XOLRg-xydjPx-7i7OCsk88dRzpCgrMyN" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_W_v04_13_00.pt"

    download "1Kvo8A0fWA7521ywTQ8JWFs8MFRRbss0d" "PandoraNet_Hierarchy_DUNEFD_HD_S_Class_v014_15_00.pt"
    download "1GX9xmA4ECMWUK584yIdMZV0ZJsST7kVx" "PandoraNet_Hierarchy_DUNEFD_HD_T_Class_v014_15_00.pt"
    download "170onYjTWoqouWOXjBmYDglc0x_bRGX9G" "PandoraNet_Hierarchy_DUNEFD_HD_T_Edge_v014_15_00.pt"
    download "15pKwjsy6ms9YxD3_wk2hZx_qF-RFswqu" "PandoraNet_Hierarchy_DUNEFD_HD_TS_Class_v014_15_00.pt"
    download "115_FD4SBYyVCXORTByPHbzymNkEgrGpf" "PandoraNet_Hierarchy_DUNEFD_HD_TS_Edge_v014_15_00.pt"
    download "1oX9JeGvi_hk6BMB1MbGFwddeiXztSeVp" "PandoraNet_Hierarchy_DUNEFD_HD_TT_Class_v014_15_00.pt"
    download "15uQbdDub85_3cvFvO8YL-y_0cRoqHWq9" "PandoraNet_Hierarchy_DUNEFD_HD_TT_Edge_v014_15_00.pt"
  fi

  if [[ "$2" == "atmos" ]]
  then
    download "1BYX6Y0sirNzBw0qsj6Xsa-qOAdqn5PG7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_U_v04_03_00.pt"
    download "1AWywC9LsCkuNo6lVo5VAWvlM2n-FWTnx" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_V_v04_03_00.pt"
    download "1CLV64endpentEDSoRFTu6obWoYL2Bx-a" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_W_v04_03_00.pt"
    download "1HYhF3xGcZXOVxXrjDnYgV_34-q5d58UC" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_U_v04_03_00.pt"
    download "1DDqVFhnRBPy6ZIehd9u1SwyiKaWC3hSF" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_V_v04_03_00.pt"
    download "1JLr5GfjSNt6vO81ZCv42UHH09hjvtFw7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_W_v04_03_00.pt"
  fi
fi

# DUNE ND

if [[ "$1" == "dunend" ]]
then
  download "1rFR3zYxTgXNzqvdQQ7e7PM9d_dtDyMTj" "PandoraNet_Vertex_DUNEND_Accel_1_U_v04_06_00.pt"
  download "10mVZEUxstmUMALhK4ja9gDEtWXOd6SZ-" "PandoraNet_Vertex_DUNEND_Accel_1_V_v04_06_00.pt"
  download "1Sq4YHhhH9gOEIZipEu9lXgcjw8PlfGGi" "PandoraNet_Vertex_DUNEND_Accel_1_W_v04_06_00.pt"
  download "1i3AYNEM0liddEzZp1ST9QR5rcORElAXN" "PandoraNet_Vertex_DUNEND_Accel_2_U_v04_06_00.pt"
  download "1ZpmskwhfeorVSRQPhzRoDRkzZ5TmOPz1" "PandoraNet_Vertex_DUNEND_Accel_2_V_v04_06_00.pt"
  download "1pEQ7d2OLDMkx6-s0l0WRyC-F86i_N-mo" "PandoraNet_Vertex_DUNEND_Accel_2_W_v04_06_00.pt"
fi

cd $MY_TEST_AREA/LArMachineLearningData
