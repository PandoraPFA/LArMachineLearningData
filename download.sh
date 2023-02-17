function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

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
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" -O PandoraSvm_v03_11_00.xml && rm -rf /tmp/cookies.txt

# DUNE

wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox" -O PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq" -O PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu" -O PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt

wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT" -O PandoraBdt_PfoCharacterisation_DUNEFD_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT" -O PandoraBdt_Vertexing_DUNEFD_v03_27_00.xml && rm -rf /tmp/cookies.txt

### PandoraMVAs
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs

# SBND
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7" -O PandoraBdt_v09_32_00_SBND.xml && rm -rf /tmp/cookies.txt




### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

# DUNE

download "1ahhvfyIfkhw1qisWoq56_1fXwKfwK2oK" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_U_v04_03_00.pt"
download "1iSwz4hPbbQLjtSBMIheyghdUL_jQkttI" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_V_v04_03_00.pt"
download "1B-Hf32cWXXdJsuEKm5O_xjv4wGjpgDJ_" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_W_v04_03_00.pt"
download "1xoJnjllqqeQG1kRj4k3r-R-p2T67GxJ0" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_U_v04_03_00.pt"
download "1wjkDjgqxEEFvZwjsU2eFFhRC-pGNDJh2" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_V_v04_03_00.pt"
download "1s3SyZii34b8s-2cE6q25apyboUa5cbv3" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_W_v04_03_00.pt"

download "1BYX6Y0sirNzBw0qsj6Xsa-qOAdqn5PG7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_U_v04_03_00.pt"
download "1AWywC9LsCkuNo6lVo5VAWvlM2n-FWTnx" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_V_v04_03_00.pt"
download "1CLV64endpentEDSoRFTu6obWoYL2Bx-a" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_W_v04_03_00.pt"
download "1HYhF3xGcZXOVxXrjDnYgV_34-q5d58UC" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_U_v04_03_00.pt"
download "1DDqVFhnRBPy6ZIehd9u1SwyiKaWC3hSF" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_V_v04_03_00.pt"
download "1JLr5GfjSNt6vO81ZCv42UHH09hjvtFw7" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_W_v04_03_00.pt"
